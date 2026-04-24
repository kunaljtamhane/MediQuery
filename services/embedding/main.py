"""
Person A — Embedding Service
FastAPI service that converts text chunks into dense vector embeddings using
BioMedBERT and stores/retrieves them from QdrantDB.

BioMedBERT is the designated encoder for MediQuery:
  - Generates 768-dim dense embeddings for QdrantDB retrieval
  - Used by the knowledge graph pipeline for NER signals via SciSpaCy
  - Base weights are frozen; it is never fine-tuned
"""
import os
import time
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger(__name__)

app = FastAPI(title="Embedding Service")

# BioMedBERT is the designated encoder — 768-dim, frozen, medical domain
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
model = SentenceTransformer(MODEL_NAME)

COLLECTION_NAME = "papers"
VECTOR_DIM = 768  # BioMedBERT output dimension


def connect_qdrant(retries: int = 10, delay: int = 5) -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "qdrant")
    port = int(os.getenv("QDRANT_PORT", 6333))
    for attempt in range(1, retries + 1):
        try:
            client = QdrantClient(host=host, port=port)
            existing = [c.name for c in client.get_collections().collections]
            if COLLECTION_NAME not in existing:
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                )
                log.info(f"Created Qdrant collection '{COLLECTION_NAME}'")
            log.info(f"Connected to Qdrant at {host}:{port}")
            return client
        except Exception as e:
            log.warning(f"Qdrant not ready (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError("Could not connect to Qdrant after multiple attempts")


qdrant = connect_qdrant()

log.info(f"Embedding service started — model={MODEL_NAME} dim={VECTOR_DIM}")


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
    return {"status": "ok", "model": MODEL_NAME, "dim": VECTOR_DIM}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    vec = model.encode(req.text).tolist()
    return EmbedResponse(embedding=vec, dim=len(vec))


@app.post("/index")
def index_document(req: IndexRequest):
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
