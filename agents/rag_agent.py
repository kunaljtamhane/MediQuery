import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http import models  # <-- Added this import
import uuid

class RAGAgent:
    def __init__(self):
        # 1. Change to a brand new collection name
        self.collection_name = "mediquery_uploads" 
        self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[RAG Agent] Initializing on device: {self.device}")
        
        self.client = QdrantClient(url="http://localhost:6333")
        
        # 2. Automatically create the new database if it doesn't exist
        try:
            self.client.get_collection(self.collection_name)
            print(f"[RAG Agent] Database '{self.collection_name}' is ready.")
        except Exception:
            print(f"[RAG Agent] Building fresh '{self.collection_name}' database...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=768, # PubMedBERT vector size
                    distance=models.Distance.COSINE
                )
            )

        print("[RAG Agent] Loading PubMedBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        print("[RAG Agent] Ready.")

    def _embed_query(self, query):
        """Translates the text query into a 768-dimensional vector."""
        inputs = self.tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling to match our indexer logic
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        return mean_pooled.cpu().numpy()[0]

    def retrieve(self, query, top_k=4):
        """Searches Qdrant and returns the top_k most relevant contexts."""
        query_vector = self._embed_query(query)
        
        # Execute the vector search in Qdrant using the updated API
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k
        )
        
        # Safely extract points whether the client returns a list or a QueryResponse object
        points = search_result.points if hasattr(search_result, 'points') else search_result
        
        # Extract the payloads (metadata) to return to the Supervisor
        retrieved_contexts = []
        for hit in points:
            retrieved_contexts.append({
                "pubid": hit.id,
                "score": hit.score,
                "context": hit.payload.get("context"),
                "answer": hit.payload.get("answer"),
                "final_decision": hit.payload.get("final_decision")
            })
        
        return retrieved_contexts

    # MOVED THIS INSIDE THE CLASS
    def ingest_chunks(self, chunks: list, source: str = "Uploaded PDF"):
        """Embeds raw text chunks and pushes them into the Qdrant collection."""
        print(f"[RAG Agent] Ingesting {len(chunks)} chunks from {source}...")
        
        points = []
        for i, chunk in enumerate(chunks):
            # 1. Use your existing embed logic for each chunk
            vector = self._embed_query(chunk).tolist() 
            
            # 2. Format the payload for Qdrant
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "context": chunk,
                        "source": source,
                        "final_decision": "User Upload"
                    }
                )
            )
            
        # 3. Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("[RAG Agent] Ingestion complete.")


# Execution block for local testing
if __name__ == "__main__":
    agent = RAGAgent()
    
    test_query = "What is the sensitivity of rapid prescreening for detecting glandular cell abnormalities?"
    print(f"\nSearching for: '{test_query}'\n")
    
    results = agent.retrieve(test_query)
    
    for i, res in enumerate(results):
        print(f"--- Result {i+1} (Score: {res['score']:.4f}) ---")
        print(f"Decision: {res['final_decision']}")
        print(f"Context Snippet: {res['context'][:250]}...\n")