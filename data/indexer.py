import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

# Configuration
DATA_PATH = "data/processed/train.csv"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
COLLECTION_NAME = "pubmedqa_pairs"
BATCH_SIZE = 64

def initialize_qdrant():
    print("Connecting to local Qdrant instance...")
    client = QdrantClient(url="http://localhost:6333")
    
    # Create collection if it doesn't exist
    # PubMedBERT outputs 768-dimensional vectors
    # We use EUCLID (L2 distance) to align with the RAG-BioQA paper's FAISS setup
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Creating new collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.EUCLID),
        )
    return client

def generate_embeddings(text_list, tokenizer, model, device):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply mean pooling over token embeddings as specified in the methodology
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    
    return mean_pooled.cpu().numpy()

def index_data():
    print("Loading data and initializing models...")
    df = pd.read_csv(DATA_PATH)
    
    # Set up hardware acceleration (MPS for Apple Silicon, CUDA for Nvidia, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    client = initialize_qdrant()
    
    points = []
    
    print("Generating embeddings and pushing to Qdrant...")
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        
        # Concatenate query and answer for the embedding engine
        qa_pairs = (batch_df['query'].astype(str) + " " + batch_df['answer'].astype(str)).tolist()
        embeddings = generate_embeddings(qa_pairs, tokenizer, model, device)
        
        # Build the Qdrant payload (metadata)
        for j, (_, row) in enumerate(batch_df.iterrows()):
            points.append(
                models.PointStruct(
                    id=int(row['pubid']),  # Using pubid as the unique vector ID
                    vector=embeddings[j].tolist(),
                    payload={
                        "query": str(row['query']),
                        "answer": str(row['answer']),
                        "context": str(row['context_str']),
                        "final_decision": str(row['final_decision'])
                    }
                )
            )
            
        # Push batch to Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        points = [] # Clear the batch list
        
    print("Indexing complete! Vector database is populated.")

if __name__ == "__main__":
    index_data()