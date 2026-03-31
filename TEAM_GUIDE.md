# Capstone Project — Team Guide
## Domain-Specific Agentic AI Research Platform

**300 Hours | 4 People | 10 Weeks | March–June 2026**
**Strategy:** Build locally (Weeks 1–8) → Deploy to AWS (Weeks 9–10)

---

## Quick Start (Everyone does this first)

```bash
# 1. Clone the repo
git clone https://github.com/kunaljtamhane/End-to-End-Research-Tool-with-Multi-Agent-System-and-Document-Vectorization
cd End-to-End-Research-Tool-with-Multi-Agent-System-and-Document-Vectorization

# 2. Set up your environment file
cp .env.template .env
# Open .env and fill in OPENAI_API_KEY (or AWS keys for Bedrock)

# 3. Start the full local stack
make up

# 4. Verify everything is running
docker compose ps
```

---

## How the System Fits Together

```
User
 └─► Streamlit Frontend (Person D)
       └─► Spring Boot API — POST /search, POST /ingest (Person C)
             ├─► LangGraph Agent Pipeline (Person A + D)
             │     ├─► RAG Agent → Embedding → ChromaDB → Reward Model → LLM
             │     └─► Research Agent → arXiv API (live lookup)
             └─► Kafka Topic "documents" → Python Consumer → Embed → Index
                                          (Person C)               (Person A)
```

**Local infrastructure** (Person B owns `docker-compose.yml`):
- **Kafka** (KRaft, single broker) — message bus for document ingestion
- **ChromaDB** — vector database for storing embeddings
- **Redis** — session deduplication cache

---

---

# Person A — ML & RAG Lead

**Role:** Build the intelligence of the system — embeddings, retrieval, reward model, agent orchestration.
**Total hours:** 75 | **Files owned:** `services/embedding/`, `services/rag/`, `services/reward_model/`, `services/agents/`

---

## What Has Been Built For You

### `services/embedding/` — Embedding Service
**File:** `main.py`
A fully working **FastAPI service** that:
- Loads `all-MiniLM-L6-v2` (sentence-transformers) on startup
- Exposes three endpoints:
  - `POST /embed` → returns a 384-dim vector for any text
  - `POST /index` → embeds a chunk and stores it in ChromaDB
  - `POST /query` → finds top-N most similar chunks via cosine similarity
- Runs in Docker, connects to ChromaDB automatically

**You can test it immediately:**
```bash
make up
curl -X POST http://localhost:8001/embed -H "Content-Type: application/json" \
     -d '{"text": "What is retrieval-augmented generation?"}'
# Returns: {"embedding": [...384 floats...], "dim": 384}
```

---

### `services/rag/` — RAG Pipeline
**Files:** `retrieval.py`, `rerank.py`, `generate.py`, `main.py`, `dedup.py`

| File | What it does | Status |
|---|---|---|
| `retrieval.py` | Dense search (ChromaDB) + BM25 (rank_bm25) + RRF merge | Ready |
| `rerank.py` | Calls reward model, gracefully falls back to RRF scores | Ready |
| `generate.py` | Streaming LLM wrapper for OpenAI and AWS Bedrock | Ready |
| `main.py` | FastAPI: `POST /query` → stream sources + answer tokens | Ready |
| `dedup.py` | Redis cosine-similarity cache (used in Weeks 5–6) | Ready |

**End-to-end query flow** (once papers are indexed):
```
POST /query {"query": "How does RLHF work?"}
  → hybrid_search() → RRF merge of dense + BM25
  → rerank() → reward model scores top-15 candidates
  → generate_stream() → streams answer tokens as SSE
```

**Test the full RAG pipeline:**
```bash
# First index a test document
curl -X POST http://localhost:8001/index \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "test_001", "text": "RLHF uses human feedback to align language models...", "metadata": {"title": "RLHF Paper"}}'

# Then query it
curl -X POST http://localhost:8002/query \
     -H "Content-Type: application/json" \
     -d '{"query": "How does RLHF work?", "top_k": 3}'
```

---

### `services/reward_model/` — Cross-Encoder Reward Model
**Files:** `train.py`, `evaluate.py`, `serve.py`

| File | What it does |
|---|---|
| `train.py` | Fine-tunes `ms-marco-MiniLM-L-6-v2` on your labeled triples |
| `evaluate.py` | Computes NDCG@5, Precision@5, MRR on a held-out benchmark |
| `serve.py` | FastAPI: `POST /rerank` → scores (query, passage) pairs |

The serving endpoint is already wired into `rerank.py` above — it will automatically start using your fine-tuned model once training is complete.

---

### `services/agents/` — LangGraph Agent Graph
**Files:** `graph.py`, `supervisor.py`, `rag_agent.py`, `research_agent.py`, `main.py`

| File | What it does |
|---|---|
| `graph.py` | Defines the LangGraph `StateGraph`: Supervisor → RAG Agent or Research Agent |
| `supervisor.py` | GPT-4o-mini classifies query intent and sets the route |
| `rag_agent.py` | Calls the RAG service and collects the streamed response |
| `research_agent.py` | Live arXiv API lookup for papers not in the corpus |
| `main.py` | FastAPI: `POST /query` → runs the graph and returns answer + sources |

---

## What You Need To Do (Week by Week)

### Weeks 1–2: Embedding Baseline
- [ ] Run the embedding service and test all 3 candidate models (see `TODO` in `main.py`)
- [ ] Pick the best model for your corpus (test on 10 sample queries)
- [ ] Update `EMBEDDING_MODEL` in `.env` with your choice
- [ ] Confirm Docker image builds and `POST /embed` returns sensible vectors

### Weeks 3–4: Hook Up Hybrid Search
- [ ] Once Person D has papers indexed, run `POST /query` end-to-end
- [ ] Verify BM25 + dense + RRF all return different top results (they should!)
- [ ] Confirm `POST /query` streams a sourced answer through the full pipeline

### Weeks 5–6: Train the Reward Model
- [ ] Work with Person D to collect 200+ labeled triples (`data/annotation/triples.jsonl`)
- [ ] Run `python train.py --data_path ../../data/annotation/triples.jsonl`
- [ ] Run `python evaluate.py` and record NDCG@5 before and after training
- [ ] Confirm re-ranking is visibly improving result quality

### Weeks 7–8: Wire Up LangGraph Agents
- [ ] Test the Supervisor routing — try "What is attention?" vs "Find latest 2025 papers on LLMs"
- [ ] Confirm RAG Agent returns sourced answers
- [ ] Confirm Research Agent returns live arXiv results for novel queries
- [ ] Test `POST /query` on the agent service end-to-end

### Weeks 9–10: Paper — ML Sections
- [ ] Write Section 3 (Methodology): reward model architecture, training procedure
- [ ] Write Section 5 (Results): NDCG@5 tables, before/after reranking comparison
- [ ] Create result tables showing precision@5, NDCG@5, MRR

---

## Key Commands
```bash
# Start just your services
docker compose up embedding rag reward_model agents

# Check logs
make logs s=embedding
make logs s=rag

# Run reward model training (after annotation is done)
cd services/reward_model
python train.py --data_path ../../data/annotation/triples.jsonl --output_dir ./model

# Evaluate reward model
python evaluate.py --model_dir ./model --benchmark_path ../../data/annotation/benchmark.jsonl
```

---

---

# Person B — Infrastructure Lead

**Role:** Make it run everywhere — locally in Docker, in CI, and on AWS.
**Total hours:** 75 | **Files owned:** `docker-compose.yml`, `infra/terraform/`, `infra/kubernetes/`, `Makefile`

---

## What Has Been Built For You

### `docker-compose.yml` — Local Stack
A complete Docker Compose file that starts **all 8 services** in the right order:

| Service | Port | Health Check | Depends On |
|---|---|---|---|
| kafka | 9092 | topic list | — |
| chromadb | 8000 | `/api/v1/heartbeat` | — |
| redis | 6379 | `ping` | — |
| embedding | 8001 | `/health` | chromadb |
| rag | 8002 | — | embedding, redis |
| reward_model | 8003 | — | chromadb |
| agents | 8004 | `/health` | rag, reward_model |
| kafka_consumer | — | — | kafka, chromadb, embedding |
| spring_boot_api | 8080 | `/health` | kafka, agents |
| frontend | 8501 | — | spring_boot_api |

All services use `.env` for secrets and `env_file: .env` in their container config.

---

### `Makefile` — Developer Shortcuts
```bash
make env      # copy .env.template → .env (run once)
make build    # build all Docker images
make up       # start all services detached
make down     # stop all services
make logs     # tail logs (make logs s=embedding)
make test     # run tests in containers
make restart  # down + build + up
make clean    # remove containers, volumes, images
```

---

### `infra/terraform/` — AWS Infrastructure as Code

| File/Module | What it provisions |
|---|---|
| `main.tf` | Root module — wires VPC, EKS, MSK, S3, ECR together |
| `variables.tf` | All tunable settings (instance types, region, names) |
| `outputs.tf` | Prints cluster name, MSK brokers, ECR registry after apply |
| `modules/vpc/` | VPC, 2 public + 2 private subnets, Internet Gateway |
| `modules/eks/` | EKS cluster + managed node group (2× t3.small) |
| `modules/msk/` | MSK Kafka broker (kafka.t3.small, single AZ) |

**S3 bucket** and **ECR repos** (one per service) are created directly in `main.tf`.

> **Important:** The Terraform backend (S3 state) is commented out. You must create the S3 bucket + DynamoDB table manually in Week 9 before uncommenting it.

---

### `infra/kubernetes/` — K8s Deployment Manifests

| File | Service covered |
|---|---|
| `configmap.yml` | Shared config for all pods (URLs, ports, Kafka topic) |
| `embedding-deployment.yml` | Embedding service Deployment + ClusterIP Service |
| `agents-deployment.yml` | Agents service Deployment + ClusterIP Service |
| `spring-boot-deployment.yml` | Spring Boot API Deployment + LoadBalancer Service |

Each manifest includes: liveness probes, resource requests/limits, `envFrom` for ConfigMap + Secrets.

> **Note:** You still need to create manifests for `rag`, `reward_model`, `kafka_consumer`, and `frontend` — use the existing ones as templates.

---

## What You Need To Do (Week by Week)

### Weeks 1–2: Docker Compose + Repo Setup
- [ ] Run `make up` and confirm all containers start clean
- [ ] Fix any port conflicts or dependency issues on your machine
- [ ] Verify each teammate can clone and run `make up` successfully
- [ ] Add a `CONTRIBUTING.md` with setup instructions if needed
- [ ] Set up the `.gitignore` (already done — verify it covers your IDE files)

### Weeks 3–4: Terraform + K8s Manifests
- [ ] Run `terraform init` and `terraform plan` — fix any errors (costs $0)
- [ ] Create the remaining 3 K8s manifests (rag, reward_model, kafka_consumer) using existing ones as templates
- [ ] Write `deployment-guide.md` — step-by-step AWS deployment instructions
- [ ] Add `docker-compose` health checks for any service missing them

### Weeks 5–6: CI/CD + Monitoring
- [ ] Create `.github/workflows/ci.yml` — run `docker compose build` + basic linting on every PR
- [ ] Add structured JSON log format to any service missing it (check `make logs s=<service>`)
- [ ] Verify `make test` runs and passes for all services

### Weeks 7–8: Minikube Testing + AWS Prep
- [ ] Install minikube locally and run `kubectl apply -f infra/kubernetes/`
- [ ] Confirm all services start and connect in minikube
- [ ] Prepare `terraform.tfvars` file with your AWS account details (do NOT commit it)
- [ ] Write the full `deployment-guide.md` — the runbook your team will follow in Week 9

### Weeks 9–10: AWS Deployment
- [ ] Create S3 bucket + DynamoDB table for Terraform state (manually via AWS console)
- [ ] Uncomment the `backend "s3"` block in `infra/terraform/main.tf`
- [ ] Run `terraform apply` — capture the output (cluster name, MSK brokers, ECR registry)
- [ ] Push Docker images to ECR: `docker tag ... && docker push ...`
- [ ] Run `kubectl apply -f infra/kubernetes/` against the EKS cluster
- [ ] Take screenshots: EKS dashboard, running pods, Kafka topics, CloudWatch
- [ ] **Run `terraform destroy` every night** to stay under $100 budget

---

## Key Commands
```bash
# Local stack
make up
make down
make logs s=kafka

# Terraform (Weeks 3-4: plan only — don't apply yet)
cd infra/terraform
terraform init
terraform plan          # costs $0, shows what would be created
terraform validate      # syntax check

# Week 9: Apply and destroy
terraform apply -var-file=terraform.tfvars
terraform destroy -var-file=terraform.tfvars   # ← run every night!

# Kubernetes (minikube, Weeks 7-8)
minikube start
kubectl apply -f infra/kubernetes/
kubectl get pods
kubectl logs deployment/embedding
```

---

---

# Person C — Backend & Integration Lead

**Role:** Build the Java API layer, wire Kafka, and integrate all Python services into a working pipeline.
**Total hours:** 75 | **Files owned:** `services/spring_boot_api/`, `services/kafka_consumer/`, `services/rag/dedup.py`

---

## What Has Been Built For You

### `services/spring_boot_api/` — Java Spring Boot REST API

**Project structure:**
```
spring_boot_api/
├── pom.xml                          Maven build (Java 21, Spring Boot 3.3)
├── Dockerfile                       Multi-stage build (Maven → JRE slim)
└── src/main/java/com/capstone/api/
    ├── ApiApplication.java          Entry point
    ├── controller/
    │   ├── HealthController.java    GET  /health  → {"status":"ok"}
    │   ├── IngestController.java    POST /ingest  → publishes to Kafka
    │   └── SearchController.java    POST /search  → streams from agent service
    └── service/
        └── KafkaProducerService.java  Sends JSON to "documents" Kafka topic
```

**Endpoints:**

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status":"ok"}` |
| `POST` | `/ingest` | Accepts a document, publishes it to Kafka |
| `POST` | `/search` | Forwards query to Python agent service, streams response back |

**Test it:**
```bash
make up

# Health check
curl http://localhost:8080/health

# Ingest a document
curl -X POST http://localhost:8080/ingest \
     -H "Content-Type: application/json" \
     -d '{"docId":"paper_001","title":"Attention Is All You Need","text":"The transformer architecture..."}'
# Kafka console should show the message appearing in the "documents" topic

# Search (requires agents service to be running)
curl -X POST http://localhost:8080/search \
     -H "Content-Type: application/json" \
     -d '{"query":"What is a transformer?","sessionId":"test-session"}'
```

---

### `services/kafka_consumer/consumer.py` — Python Ingestion Pipeline

Reads from the `documents` Kafka topic and runs each document through:
1. **Chunking** — splits text into 512-word overlapping windows
2. **Embedding** — calls `POST http://embedding:8001/index` for each chunk
3. **Storage** — embedding service stores chunks in ChromaDB

The consumer retries Kafka connection on startup (so it waits for Kafka to be ready).

**Test the full pipeline:**
```bash
make up
# Ingest a document via Spring Boot
curl -X POST http://localhost:8080/ingest \
     -H "Content-Type: application/json" \
     -d '{"docId":"paper_001","title":"Test Paper","text":"Large language models use self-attention..."}'

# Wait 5 seconds, then query for it
curl -X POST http://localhost:8001/query \
     -H "Content-Type: application/json" \
     -d '{"text":"self-attention language models","n_results":3}'
# Should return paper_001 chunks in the results
```

---

### `services/rag/dedup.py` — Session Deduplication Engine

Prevents the same passages from being shown again in a multi-turn conversation:
- Each passage embedding is stored in Redis per session
- Before returning a passage, checks cosine similarity against all previously served embeddings
- If similarity > 0.85, the passage is skipped and the next candidate is used
- Session cache expires after 1 hour

> **When to use:** Wire this into `services/rag/main.py` in Week 5. The `TODO` comment is already there.

---

## What You Need To Do (Week by Week)

### Weeks 1–2: Spring Boot Scaffold
- [ ] Build and run the Spring Boot service: `docker compose up spring_boot_api`
- [ ] Verify `GET /health` returns `{"status":"ok"}`
- [ ] Verify `POST /ingest` publishes to Kafka — check with Kafka console consumer:
  ```bash
  docker exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
    --bootstrap-server localhost:9092 --topic documents --from-beginning
  ```
- [ ] Verify the service appears healthy in `docker compose ps`

### Weeks 3–4: Kafka Consumer Pipeline
- [ ] Run `docker compose up kafka_consumer` and check its logs
- [ ] Ingest a document via `/ingest` and confirm it appears in ChromaDB after ~5 seconds
- [ ] Test the full pipeline: ingest → wait → query via embedding service
- [ ] Tune `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env` if needed

### Weeks 5–6: Deduplication Engine
- [ ] Wire `dedup.py` into `services/rag/main.py` (the `TODO` is on line ~40)
- [ ] Test: ask 3 follow-up questions in the same session, verify no passages repeat
- [ ] Adjust `DEDUP_THRESHOLD` in `.env` if needed (0.85 is a good starting point)
- [ ] Write the integration test: multi-turn conversation where each turn returns new passages

### Weeks 7–8: Research Agent + Full Integration
- [ ] Verify `POST /search` in Spring Boot routes through to the agent service
- [ ] Test the Research Agent: query "Find latest 2025 papers on diffusion models"
  — should call arXiv API and return live results
- [ ] Run an end-to-end integration test: `Streamlit → Spring Boot → Agents → RAG → LLM`
- [ ] Write paper section on Kafka integration and Spring Boot API design

### Weeks 9–10: Paper — Architecture Section
- [ ] Write Section 4 (Architecture): system diagram, Kafka flow, Spring Boot API design
- [ ] Create architecture diagram using draw.io or Mermaid
- [ ] Prepare and rehearse the 5-minute live demo script

---

## Key Commands
```bash
# Build Spring Boot JAR locally (without Docker)
cd services/spring_boot_api
mvn package -DskipTests
java -jar target/*.jar

# Build and run via Docker
docker compose up spring_boot_api kafka_consumer

# Check Kafka messages
docker exec kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 --topic documents --from-beginning

# Check Kafka topics
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list

# Check ChromaDB (verify documents are indexed)
curl http://localhost:8000/api/v1/collections
```

---

---

# Person D — Data & Frontend Lead

**Role:** Build the corpus, label training data, and create the user-facing demo.
**Total hours:** 75 | **Files owned:** `data/`, `services/frontend/`, `services/agents/research_agent.py`

---

## What Has Been Built For You

### `data/collection/arxiv_scraper.py` — arXiv Paper Scraper

Scrapes papers from cs.CL, cs.AI, cs.LG and outputs a JSONL file:
```json
{"doc_id": "2310.01234", "title": "...", "abstract": "...", "authors": [...], "published": "2023-10-05", "url": "...", "pdf_url": "...", "category": "cs.CL", "full_text": null}
```

**Run it:**
```bash
cd data/collection
pip install -r requirements.txt
python arxiv_scraper.py --output ../raw/papers.jsonl --max_results 3000
# Takes ~1-2 hours (respects arXiv's 3-second rate limit)
```

---

### `data/collection/pdf_extractor.py` — PDF Full-Text Extractor

Downloads arXiv PDFs and extracts full text using PyMuPDF:
```bash
python pdf_extractor.py \
  --input ../raw/papers.jsonl \
  --output ../raw/papers_with_text.jsonl \
  --limit 500
# Downloads 500 PDFs (the rest use abstract only)
```

After this step, 500 papers will have `full_text` populated. The remaining 2,500 papers will use their abstract as the indexed text.

---

### `data/annotation/annotate.py` — Reward Model Annotation Tool

Interactive CLI for labeling (query, passage, relevance) triples:

```bash
cd data/annotation
python annotate.py --output triples.jsonl --rag_url http://localhost:8002
```

**Labeling scale:** 0 = Irrelevant, 1 = Partially relevant, 2 = Highly relevant, 3 = Perfect

For each query it fetches the top-10 passages from the RAG service, you label each one, and it writes pairwise triples (positive vs negative) to `triples.jsonl`.

**Target:** 200+ triples by end of Week 5, 300+ by end of Week 6.

---

### `services/frontend/app.py` — Streamlit Research Assistant

A full multi-turn research assistant UI:

| Feature | Implementation |
|---|---|
| Streaming response | Server-Sent Events from Spring Boot → rendered token by token |
| Source attribution | Right panel shows paper title, relevance score, link |
| Multi-turn history | `st.session_state.messages` persists across turns |
| Dedup indicator | Sources labeled **NEW** (green) or **seen** (gray) per session |
| Session stats | Turn count + unique sources seen displayed live |
| Clear button | Resets conversation and session ID |

**Run locally (without Docker):**
```bash
cd services/frontend
pip install -r requirements.txt
SPRING_BOOT_URL=http://localhost:8080 streamlit run app.py
# Opens at http://localhost:8501
```

---

### `services/agents/research_agent.py` — Live arXiv Research Agent

Queries the arXiv API live for papers not in your corpus. Already integrated into the LangGraph graph — the Supervisor routes "find latest papers on X" type queries here automatically.

You own this file jointly with Person A. Your job is to test it and ensure it returns clean, well-formatted results to the frontend.

---

## What You Need To Do (Week by Week)

### Weeks 1–2: Corpus Collection
- [ ] Run `arxiv_scraper.py` — target 3,000 papers in `data/raw/papers.jsonl`
- [ ] Run `pdf_extractor.py` — extract full text from 500 PDFs
- [ ] Verify output: open the JSONL and spot-check 10 entries for correctness
- [ ] Share the completed JSONL with the team (upload to shared Google Drive or S3)
- [ ] Add 30 seed queries to `data/annotation/annotate.py` (the `SEED_QUERIES` list)

### Weeks 3–4: Bulk Indexing
- [ ] Send all 3,000 papers through the ingestion pipeline:
  ```bash
  # For each paper in the JSONL, POST to /ingest
  python bulk_ingest.py  # you need to write this small script
  ```
- [ ] Verify all papers are searchable: run 5 test queries via the embedding service
- [ ] Test LLM generation end-to-end (set `OPENAI_API_KEY` or AWS Bedrock creds in `.env`)

### Weeks 5–6: Annotation + Evaluation Benchmark
- [ ] Run the annotation tool and label 200+ triples
- [ ] Create `data/annotation/benchmark.jsonl` — 50 queries with known-relevant passages
  (format: `{"query": "...", "candidates": [{"text": "...", "label": 2}, ...]}`)
- [ ] Share `benchmark.jsonl` with Person A for reward model evaluation

### Weeks 7–8: Streamlit Frontend
- [ ] Run the frontend and test it against the live system
- [ ] Test multi-turn conversation — confirm dedup indicator works (green NEW → gray seen)
- [ ] Polish the UI: add paper links, author info, publication date to source panel
- [ ] Test the Research Agent routing ("find latest papers on X")

### Weeks 9–10: Paper + Demo Recording
- [ ] Write Abstract, Introduction, Related Work, Conclusion sections
- [ ] Format paper in IEEE style (or your required format)
- [ ] Record a screen-capture demo video with narration (~5 minutes)
- [ ] Do final Streamlit polish for the live presentation

---

## Key Commands
```bash
# Data collection
cd data/collection
python arxiv_scraper.py --output ../raw/papers.jsonl --max_results 3000
python pdf_extractor.py --input ../raw/papers.jsonl --output ../raw/papers_with_text.jsonl --limit 500

# Annotation (Weeks 5-6, after RAG pipeline is running)
cd data/annotation
python annotate.py --output triples.jsonl --rag_url http://localhost:8002

# Frontend
cd services/frontend
streamlit run app.py
# or via Docker:
docker compose up frontend
# Open http://localhost:8501
```

---

---

## Shared Responsibilities (Everyone)

### Paper Writing (15 hrs total, spread across team)
| Section | Owner |
|---|---|
| Abstract + Introduction | Person D |
| Related Work | Person D |
| Methodology (reward model, RAG) | Person A |
| Architecture (Kafka, Spring Boot, K8s) | Person C |
| Results + Evaluation tables | Person A |
| AWS Deployment section | Person B |
| Conclusion + Future Work | Person D |
| Formatting + final pass | Everyone |

### Testing & Integration (5 hrs total)
- Every service has a `/health` endpoint — run `curl http://localhost:<PORT>/health` as a smoke test
- Full end-to-end test: ingest a paper → wait → query for it → see it in Streamlit
- Integration test before each weekly sync: `make up && make test`

---

## Weekly Sync Checklist

Run every **Friday** (30 min):
1. Everyone does a `git pull` and `make up`
2. Each person demos what they built this week (2–3 min each)
3. Identify any blockers for next week
4. Check the weekly milestone (see execution plan)

**Golden rule:** Every work session ends with a `git commit && git push`. Even a WIP commit is better than lost work.

---

## File Ownership Summary

| Path | Owner | Week Active |
|---|---|---|
| `services/embedding/` | Person A | 1–2 |
| `services/rag/retrieval.py` | Person A | 3–4 |
| `services/rag/rerank.py` | Person A | 3–6 |
| `services/rag/generate.py` | Person A | 3–4 |
| `services/rag/dedup.py` | Person C | 5–6 |
| `services/reward_model/` | Person A | 5–6 |
| `services/agents/graph.py` | Person A | 7–8 |
| `services/agents/supervisor.py` | Person A | 7–8 |
| `services/agents/rag_agent.py` | Person A | 7–8 |
| `services/agents/research_agent.py` | Person C + D | 7–8 |
| `services/kafka_consumer/` | Person C | 3–4 |
| `services/spring_boot_api/` | Person C | 1–2, 7–8 |
| `services/frontend/` | Person D | 7–8 |
| `data/collection/` | Person D | 1–2 |
| `data/annotation/` | Person D | 5–6 |
| `docker-compose.yml` | Person B | 1–2 |
| `Makefile` | Person B | 1–2 |
| `infra/terraform/` | Person B | 3–4, 9–10 |
| `infra/kubernetes/` | Person B | 3–4, 7–8 |
| `shared/schemas/` | Everyone | ongoing |
