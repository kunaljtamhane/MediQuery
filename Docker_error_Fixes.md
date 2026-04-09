# Docker Stack Fixes & Lessons Learned

Reference doc for debugging the local Docker stack. If something breaks, check here first.

---

## 1. ChromaDB → Replaced with Qdrant

**Problem:** `chromadb/chroma:latest` image has no `curl`, `wget`, or `python` in PATH — Docker health checks always failed.

**Fix:** Replaced with `qdrant/qdrant:latest` in `docker-compose.yml`.
- Port mapping: `8000:6333` (host 8000 → container internal 6333)
- Health check: `bash -c "echo > /dev/tcp/localhost/6333"`
- `services/embedding/requirements.txt`: `chromadb==0.5.3` → `qdrant-client==1.9.1`
- `services/embedding/main.py`: swapped ChromaDB client for Qdrant client. Collection = `papers`, vector size = 384, distance = Cosine.

---

## 2. .env Must Use Docker Service Names (Not localhost)

**Problem:** Inside a Docker container, `localhost` refers to that container itself — not the host machine or other containers.

**Fix:** Updated `.env` to use Docker Compose service names:

| Variable | Wrong | Correct |
|---|---|---|
| `CHROMA_HOST` | `localhost` | `chromadb` |
| `CHROMA_PORT` | `8000` | `6333` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | `kafka:9092` |
| `EMBEDDING_SERVICE_URL` | `http://localhost:8001` | `http://embedding:8001` |
| `RAG_SERVICE_URL` | `http://localhost:8002` | `http://rag:8002` |
| `REWARD_MODEL_URL` | `http://localhost:8003` | `http://reward_model:8003` |
| `AGENT_SERVICE_URL` | `http://localhost:8004` | `http://agents:8004` |
| `REDIS_HOST` | `localhost` | `redis` |

> `SPRING_BOOT_URL=http://localhost:8080` stays as `localhost` — it's only used from your Mac (browser/scripts), not from inside a container.

---

## 3. Embedding Service Crashed on Startup (No Retry)

**Problem:** `services/embedding/main.py` connected to Qdrant at import time with no retry. Qdrant wasn't fully ready when embedding started, causing immediate crash.

**Fix:** Added `connect_qdrant()` retry loop (10 attempts, 5s delay) in `main.py`.

---

## 4. Docker Health Checks — curl Not Available in Slim Images

**Problem:** `python:3.11-slim` and `eclipse-temurin:21-jre-alpine` don't have `curl` installed.

**Fix per service:**

| Service | Image | Health Check |
|---|---|---|
| `embedding` | python:3.11-slim | `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"` |
| `spring_boot_api` | eclipse-temurin:21-jre-alpine | `wget --no-verbose --tries=1 --spider http://localhost:8080/health` |
| `chromadb` (Qdrant) | qdrant/qdrant | `bash -c "echo > /dev/tcp/localhost/6333"` |

---

## 5. Spring Boot JSON Escaping Was Broken

**Problem:** `IngestController.toJson()` used manual string formatting. Didn't escape backslashes first, so papers with `\` in full_text produced invalid JSON. Kafka consumer crashed with `JSONDecodeError: Invalid \escape`.

**Fix:** Updated `escape()` in `IngestController.java` — backslash must be escaped before all other characters:
```java
private String escape(String s) {
    if (s == null) return "";
    return s.replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t");
}
```

---

## 6. Kafka Consumer Crashed on PDF Control Characters

**Problem:** PDF-extracted text contains hidden control characters. `json.loads()` rejects these by default, crashing the entire consumer process.

**Fix:** Added `_safe_deserialize()` in `services/kafka_consumer/consumer.py`:
```python
def _safe_deserialize(v):
    if v is None:
        return None
    try:
        return json.loads(v.decode("utf-8"), strict=False)
    except json.JSONDecodeError as e:
        log.error(f"Skipping unparseable message: {e}")
        return None
```
And added a `None` check in the consumer loop to skip bad messages without crashing.

---

## Useful Debug Commands

```bash
# Check which containers are running and their health
docker compose ps

# Check logs for a specific service
docker compose logs --tail 30 <service>
# e.g. docker compose logs --tail 30 kafka_consumer

# Check how many vectors are indexed in Qdrant
curl -s http://localhost:8000/collections/papers | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('Points:', d['result']['points_count'])"

# Check what tools are available inside a container
docker exec <container> sh -c "ls /bin /usr/bin /usr/local/bin"

# Restart a single service after a code change
docker compose build <service> && docker compose up -d <service>
```

---

## Bulk Ingestion

Script: `data/collection/bulk_ingest.py`

- Reads `data/annotation/papers.jsonl`
- Filters to 500 papers where `full_text_extracted=True`
- POSTs each to `http://localhost:8080/ingest` (Spring Boot → Kafka → consumer → embedding → Qdrant)

```bash
cd data/collection
python bulk_ingest.py
```
