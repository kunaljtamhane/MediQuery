from unsloth import FastLanguageModel

# 1. Load the model you ALREADY trained (saved in the mediquery_llama3 folder)
print("Loading trained model from disk...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "mediquery_llama3", 
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True,
)

# 2. Run the GGUF export step
print("Starting GGUF conversion...")
model.save_pretrained_gguf("mediquery_llama3_final", tokenizer, quantization_method = "q4_k_m")
print("Success! Conversion complete.")