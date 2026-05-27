import pandas as pd
from datasets import load_dataset
import os

def ingest_and_sanitize_pubmedqa():
    print("Downloading PubMedQA dataset...")
    # Load the expert-labeled subset of PubMedQA using the correct namespace
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    
    # Convert to Pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset)
    
    print("Sanitizing data...")
    # PubMedQA provides context, question, and long_answer
    # We will concatenate the context list into a single string
    df['context_str'] = df['context'].apply(lambda x: " ".join(x['contexts']))
    
    # Standardize column names for our pipeline
    df = df.rename(columns={'question': 'query', 'long_answer': 'answer'})
    
    # Drop any rows with missing critical data
    df = df.dropna(subset=['query', 'answer', 'context_str'])
    
    # Remove excessive whitespace to ensure clean tokenization downstream
    for col in ['query', 'answer', 'context_str']:
        df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    print(f"Total sanitized records: {len(df)}")
    return df

def split_and_save(df, output_dir="data/processed"):
    print("Splitting dataset into 70/15/15...")
    
    # Shuffle the dataset with a fixed random state for reproducibility
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_end = int(len(df_shuffled) * 0.70)
    val_end = int(len(df_shuffled) * 0.85)
    
    df_train = df_shuffled.iloc[:train_end]
    df_val = df_shuffled.iloc[train_end:val_end]
    df_test = df_shuffled.iloc[val_end:]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df_train.to_csv(f"{output_dir}/train.csv", index=False)
    df_val.to_csv(f"{output_dir}/val.csv", index=False)
    df_test.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"Data saved to {output_dir}/")
    print(f"Training: {len(df_train)} | Validation: {len(df_val)} | Test: {len(df_test)}")

if __name__ == "__main__":
    sanitized_data = ingest_and_sanitize_pubmedqa()
    split_and_save(sanitized_data)