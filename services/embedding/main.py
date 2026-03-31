"""
Person A — Embedding Service (Weeks 1-2)
FastAPI service that converts text chunks into vector embeddings
and stores/retrieves them from ChromaDB.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import os
import logging

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

app = FastAPI(title="Embedding Service")

# TODO Week 1: Try these three models and pick the best one for your corpus:
#   - "all-MiniLM-L6-v2"       (fast, 384-dim, good baseline)
#   - "all-mpnet-base-v2"       (slower, 768-dim, better quality)
#   - "BAAI/bge-small-en-v1.5"  (strong for retrieval tasks)
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
model = SentenceTransformer(MODEL_NAME)

chroma_client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "localhost"),
    port=int(os.getenv("CHROMA_PORT", 8000)),
)
collection = chroma_client.get_or_create_collection(
    name="papers",
    metadata={"hnsw:space": "cosine"},
)

log.info(f"Embedding service started with model={MODEL_NAME}")


# ── Request/Response Schemas ──────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int


class IndexRequest(BaseModel):
    doc_id: str
    text: str
    metadata: dict = {}


class QueryRequest(BaseModel):
    text: str
    n_results: int = 10


class QueryResult(BaseModel):
    doc_id: str
    text: str
    score: float
    metadata: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """Return raw embedding vector for a piece of text."""
    vec = model.encode(req.text).tolist()
    return EmbedResponse(embedding=vec, dim=len(vec))


@app.post("/index")
def index_document(req: IndexRequest):
    """Embed a text chunk and store it in ChromaDB."""
    vec = model.encode(req.text).tolist()
    collection.upsert(
        ids=[req.doc_id],
        embeddings=[vec],
        documents=[req.text],
        metadatas=[req.metadata],
    )
    log.info(f"Indexed doc_id={req.doc_id}")
    return {"status": "indexed", "doc_id": req.doc_id}


@app.post("/query", response_model=list[QueryResult])
def query(req: QueryRequest):
    """Find the top-N most similar chunks to a query."""
    vec = model.encode(req.text).tolist()
    results = collection.query(
        query_embeddings=[vec],
        n_results=req.n_results,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for i in range(len(results["ids"][0])):
        output.append(QueryResult(
            doc_id=results["ids"][0][i],
            text=results["documents"][0][i],
            score=1 - results["distances"][0][i],  # cosine distance → similarity
            metadata=results["metadatas"][0][i],
        ))
    return output
