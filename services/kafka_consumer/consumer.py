"""
Person C — Kafka Consumer / Ingestion Pipeline (Weeks 3-4)
Reads from the "documents" topic, chunks text, embeds each chunk,
and stores it in ChromaDB via the embedding service.

Run locally:  python consumer.py
In Docker:    handled by docker-compose
"""
import os
import json
import time
import logging
import httpx
from kafka import KafkaConsumer

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_DOCUMENTS", "documents")
EMBEDDING_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding:8001")

# TODO Week 3: Tune chunk size. 512 tokens ≈ 400 words is a good starting point.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Naive word-boundary chunking with overlap.
    TODO Week 3: Compare with sentence-level chunking using NLTK/spaCy.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def index_document(doc: dict):
    """Chunk a document and send each chunk to the embedding service for indexing."""
    doc_id = doc.get("doc_id", "unknown")
    text = doc.get("text", "")
    metadata = {
        "title": doc.get("title", ""),
        "authors": doc.get("authors", ""),
        "published": doc.get("published", ""),
        "url": doc.get("url", ""),
        "doc_id": doc_id,
    }

    if not text.strip():
        log.warning(f"Empty text for doc_id={doc_id}, skipping")
        return

    chunks = chunk_text(text)
    log.info(f"Indexing doc_id={doc_id} → {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        try:
            resp = httpx.post(
                f"{EMBEDDING_URL}/index",
                json={"doc_id": chunk_id, "text": chunk, "metadata": {**metadata, "chunk_index": i}},
                timeout=30.0,
            )
            resp.raise_for_status()
        except Exception as e:
            log.error(f"Failed to index chunk {chunk_id}: {e}")


def run():
    """Main consumer loop — retries connection until Kafka is ready."""
    log.info(f"Connecting to Kafka at {KAFKA_BOOTSTRAP}, topic={KAFKA_TOPIC}")

    # Retry loop so consumer waits for Kafka to be ready on startup
    consumer = None
    for attempt in range(10):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id="ingestion-pipeline",
                auto_offset_reset="earliest",
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                enable_auto_commit=True,
            )
            log.info("Kafka consumer connected")
            break
        except Exception as e:
            log.warning(f"Kafka not ready (attempt {attempt+1}/10): {e}")
            time.sleep(5)

    if consumer is None:
        raise RuntimeError("Could not connect to Kafka after 10 attempts")

    log.info("Listening for documents...")
    for message in consumer:
        try:
            doc = message.value
            log.info(f"Received doc_id={doc.get('doc_id')} from partition={message.partition} offset={message.offset}")
            index_document(doc)
        except Exception as e:
            log.error(f"Error processing message: {e}")


if __name__ == "__main__":
    run()
