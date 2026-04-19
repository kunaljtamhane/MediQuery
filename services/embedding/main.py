"""
Person A — Embedding Service (Weeks 1-2)
FastAPI service that converts text chunks into vector embeddings
and stores/retrieves them from Qdrant.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import time
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

COLLECTION_NAME = "papers"
VECTOR_DIM = 384  # matches all-MiniLM-L6-v2; update if you change the model


def connect_qdrant(retries=10, delay=5):
    host = os.getenv("CHROMA_HOST", "qdrant")
    port = int(os.getenv("CHROMA_PORT", 6333))
    for attempt in range(1, retries + 1):
        try:
            client = QdrantClient(host=host, port=port)
            # Create collection if it doesn't exist
            existing = [c.name for c in client.get_collections().collections]
            if COLLECTION_NAME not in existing:
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                )
                log.info(f"Created Qdrant collection '{COLLECTION_NAME}'")
            log.info("Connected to Qdrant")
            return client
        except Exception as e:
            log.warning(f"Qdrant not ready (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError("Could not connect to Qdrant after multiple attempts")


qdrant = connect_qdrant()

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
    """Embed a text chunk and store it in Qdrant."""
    vec = model.encode(req.text).tolist()
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=abs(hash(req.doc_id)) % (2**63),
            vector=vec,
            payload={"doc_id": req.doc_id, "text": req.text, **req.metadata},
        )],
    )
    log.info(f"Indexed doc_id={req.doc_id}")
    return {"status": "indexed", "doc_id": req.doc_id}


@app.post("/query", response_model=list[QueryResult])
def query(req: QueryRequest):
    """Find the top-N most similar chunks to a query."""
    vec = model.encode(req.text).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=req.n_results,
        with_payload=True,
    )
    return [
        QueryResult(
            doc_id=hit.payload.get("doc_id", str(hit.id)),
            text=hit.payload.get("text", ""),
            score=hit.score,
            metadata={k: v for k, v in hit.payload.items() if k not in ("doc_id", "text")},
        )
        for hit in results
    ]
