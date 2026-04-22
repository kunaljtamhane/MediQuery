"""
Person A — LLM Generation
Wraps OpenAI, AWS Bedrock, or the locally fine-tuned LLaMA-3 LoRA adapter
for streaming medical answer generation.

Provider selection via LLM_PROVIDER env var:
  "openai"  — GPT-4o-mini (default, for development)
  "bedrock" — Claude via AWS Bedrock
  "local"   — Fine-tuned LLaMA-3 with LoRA adapter loaded from LORA_ADAPTER_PATH
"""
import os
from typing import Generator

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./services/fine_tuning/lora_adapter")

MEDICAL_SYSTEM_PROMPT = (
    "You are MediQuery, a domain-specific medical research assistant. "
    "Answer questions using ONLY the provided source passages. "
    "Cite every factual claim with its source number, e.g. [1], [2]. "
    "If the passages do not contain sufficient evidence to answer, say exactly: "
    "'Insufficient evidence in retrieved sources. Please consult a licensed medical professional.' "
    "Never speculate, extrapolate, or generate information not grounded in the provided context."
)


def build_prompt(query: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] (Source: {p['metadata'].get('title', p['doc_id'])})\n{p['text']}"
        for i, p in enumerate(passages)
    )
    return f"{MEDICAL_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"


def generate_stream(query: str, passages: list[dict]) -> Generator[str, None, None]:
    """Yield answer tokens one at a time (streaming)."""
    prompt = build_prompt(query, passages)

    if LLM_PROVIDER == "openai":
        yield from _openai_stream(prompt)
    elif LLM_PROVIDER == "bedrock":
        yield from _bedrock_stream(prompt)
    elif LLM_PROVIDER == "local":
        yield from _local_stream(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _openai_stream(prompt: str) -> Generator[str, None, None]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


def _bedrock_stream(prompt: str) -> Generator[str, None, None]:
    import boto3
    import json
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = client.invoke_model_with_response_stream(
        modelId=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-5"),
        body=body,
    )
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk.get("type") == "content_block_delta":
            yield chunk["delta"].get("text", "")


def _local_stream(prompt: str) -> Generator[str, None, None]:
    """
    Load the LoRA-adapted LLaMA-3 from LORA_ADAPTER_PATH and run inference.
    The adapter is loaded once; subsequent calls reuse the cached model.
    Falls back to OpenAI if the adapter path does not exist.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    from peft import PeftModel
    import threading

    adapter_path = LORA_ADAPTER_PATH
    if not os.path.isdir(adapter_path):
        import logging
        logging.getLogger(__name__).warning(
            f"LoRA adapter not found at {adapter_path} — falling back to OpenAI"
        )
        yield from _openai_stream(prompt)
        return

    if not hasattr(_local_stream, "_model"):
        base_model_id = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _local_stream._model = PeftModel.from_pretrained(base, adapter_path)
        _local_stream._tokenizer = tokenizer
        _local_stream._model.eval()

    model = _local_stream._model
    tokenizer = _local_stream._tokenizer

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    thread = threading.Thread(
        target=model.generate,
        kwargs={
            **inputs,
            "streamer": streamer,
            "max_new_tokens": 512,
            "do_sample": False,
        },
    )
    thread.start()
    for token in streamer:
        yield token
    thread.join()
