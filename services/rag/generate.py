"""
Person A / Person D — LLM Generation (Weeks 3-4)
Wraps OpenAI or AWS Bedrock for streaming answer generation.
"""
import os
from typing import Generator

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" | "bedrock"


def build_prompt(query: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] (Source: {p['metadata'].get('title', p['doc_id'])})\n{p['text']}"
        for i, p in enumerate(passages)
    )
    return f"""You are a research assistant. Answer the question using ONLY the provided context.
Cite sources by number, e.g. [1], [2]. If you cannot answer from the context, say so.

Context:
{context}

Question: {query}
Answer:"""


def generate_stream(query: str, passages: list[dict]) -> Generator[str, None, None]:
    """Yield answer tokens one at a time (streaming)."""
    prompt = build_prompt(query, passages)

    if LLM_PROVIDER == "openai":
        yield from _openai_stream(prompt)
    elif LLM_PROVIDER == "bedrock":
        yield from _bedrock_stream(prompt)
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
    import boto3, json
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
