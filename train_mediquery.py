from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# max_seq_length = 2048 
max_seq_length = 1024
dtype = None 
load_in_4bit = True 

print("[1/5] Loading Base 4-bit Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("[2/5] Attaching LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# 3. Data Formatting (ChatML format)
print("[3/5] Formatting Dataset...")
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly precise medical research assistant. Answer the user's question using ONLY the provided contexts. Do not invent information. If the contexts contradict, defer to the PEER-REVIEWED LITERATURE.<|eot_id|><|start_header_id|>user<|end_header_id|>
Context data:
{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>"""

def format_prompts(examples):
    texts = []
    for ctx, q, a in zip(examples["context"], examples["query"], examples["answer"]):
        text = prompt_template.format(context=ctx, question=q, answer=a)
        texts.append(text)
    return texts # <-- FIX 1: Return a simple list of strings, not a dictionary

# Load the CSV
dataset = load_dataset("csv", data_files="data/processed/train.csv", split="train")
# <-- FIX 2: Removed dataset.map(). The Trainer will do it automatically.

# 4. Training Initialization
print("[4/5] Initializing Trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    formatting_func = format_prompts, # <-- FIX 3: Pass the function directly into the Trainer
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 1, # <-- REDUCED to save VRAM
        gradient_accumulation_steps = 8, # <-- INCREASED to maintain learning rate
        warmup_steps = 5,
        max_steps = 60, 
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

print("\n--- Starting Training Loop ---\n")
trainer_stats = trainer.train()

print("\n[5/5] Training Complete. Merging and Exporting to GGUF format...")
model.save_pretrained_gguf("mediquery_llama3", tokenizer, quantization_method = "q4_k_m")
print("\nSuccess! The mediquery_llama3-unsloth.Q4_K_M.gguf file is ready in your folder.")