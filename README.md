# End-to-End Research Tool with Multi-Agent System and Document Vectorization

## Contributors
- Rohit Goutam Maity  
- Rashi Jain  
- Jaya Prakash Gorla  
- Kunal Jatin Tamhane  

---

## Overview

This project presents an end-to-end research platform powered by a multi-agent system that enables efficient document parsing, vector storage, and intelligent querying.

By leveraging Docling, Pinecone, and LangGraph, the system supports retrieval-augmented generation (RAG) along with external research capabilities. The application is containerized and includes a Streamlit-based interface for seamless user interaction.

---

## Features

- **Document Parsing and Vectorization**
  - Parse and process documents using Docling  
  - Store embeddings in Pinecone for efficient similarity search  

- **Multi-Agent Research System**
  - RAG Agent for document-specific question answering  
  - Research Agent for retrieving live academic research from arXiv and PubMed  
  - Web Search Agent for broader contextual information  

- **Intelligent Querying**
  - Select processed documents  
  - Perform context-aware searches  
  - Receive accurate and grounded responses  

- **Report Generation**
  - Export research results in PDF format  
  - Generate Codelabs-style instructional outputs  

- **User Interface**
  - Interactive interface built with Streamlit  
  - Session-based research summaries  

---

## Problem Statement

### Objective
Develop a research platform that integrates document processing, vector storage, retrieval-augmented generation, and multi-agent reasoning.

### Challenges
- Efficient storage and retrieval of document vectors  
- Ensuring contextual accuracy and relevance in responses  
- Combining internal document knowledge with external data sources  

### Goal
Create a scalable and user-friendly research system that streamlines knowledge discovery and interaction.

---

## Tech Stack

- Document Processing: Docling  
- Vector Database: Pinecone  
- Orchestration: LangGraph  
- Frontend: Streamlit  
- Containerization: Docker  

---

## System Workflow

### 1. Document Parsing and Vectorization
Documents are parsed using Docling, converted into embeddings, and stored in Pinecone for similarity-based retrieval.

### 2. Multi-Agent Research
Users select documents and interact with multiple agents:
- RAG Agent provides document-grounded answers  
 - Research Agent retrieves live academic insights from arXiv and PubMed  
- Web Search Agent supplements with external information  

### 3. User Interaction
Users interact through the Streamlit interface, where responses are generated using combined agent outputs. Research sessions can be saved and exported.

---

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/kunaljtamhane/End-to-End-Research-Tool-with-Multi-Agent-System-and-Document-Vectorization.git

# Navigate to project directory
cd End-to-End-Research-Tool-with-Multi-Agent-System-and-Document-Vectorization

# Copy and fill in environment variables
cp .env.template .env

# Build and run using Docker
docker-compose up --build
```

---

## Changelog

### April 21, 2026 — MediQuery Architecture Refactor (Aligned with Finalized Proposal)

This update aligns the entire codebase with the finalized MediQuery project proposal (Option A: Supervised Fine-Tuning). All references to the old reward model, ChromaDB, and two-agent system have been removed or replaced.

#### Deleted
- **`services/reward_model/`** — Entire folder removed. The reward model concept was not part of the finalized pipeline. The cross-encoder is repositioned as a passage reranker (see below), and LLaMA-3 is fine-tuned via direct LoRA SFT — no reinforcement learning loop.

#### New Files Added
- **`services/reranker/Dockerfile`** and **`services/reranker/requirements.txt`** — Docker build support for the cross-encoder reranker service (ms-marco-MiniLM, fine-tuned on MedAESQA triples).
- **`services/fine_tuning/train.py`** — LoRA/PEFT supervised fine-tuning of LLaMA-3 8B on 28,136 MedAESQA triples. Config: r=16, alpha=32, attention matrices only, 3 epochs, early stopping on 15% held-out MedAESQA split. Fallback models: BioMedLM (2.7B), Meditron-7B.
- **`services/agents/web_search_agent.py`** — Web Search Agent using DuckDuckGo (no API key required). Handles queries that go beyond indexed medical literature.
- **`services/agents/knowledge_graph.py`** — NetworkX + SciSpaCy knowledge graph module. Extracts medical entities (Disease, Drug, Gene, Cell, Organism) from ingested documents, builds a co-occurrence graph, and injects first- and second-degree neighbor relationships as structured context into LLM prompts. Nodes with ≤1 co-occurrence edge are flagged as low-confidence and excluded from query-time context.

#### Updated Files

**Embedding Service (`services/embedding/main.py`)**
- Switched encoder from `all-MiniLM-L6-v2` (384-dim) to **BioMedBERT** (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, 768-dim) — the designated frozen encoder for MediQuery.
- Updated environment variable names from `CHROMA_HOST`/`CHROMA_PORT` to `QDRANT_HOST`/`QDRANT_PORT`.
- Default host updated from `chromadb` to `qdrant`.

**RAG Retrieval (`services/rag/retrieval.py`)**
- Removed `chromadb` import and client entirely.
- Added `qdrant_client` with `scroll()` for BM25 corpus construction.
- Added **Maximal Marginal Relevance (MMR)** reranking after RRF merge to reduce redundancy in retrieved passages (per proposal spec).
- Updated env vars to `QDRANT_HOST`/`QDRANT_PORT`.

**Reranker (`services/rag/rerank.py`)**
- Renamed env var from `REWARD_MODEL_URL` to `RERANKER_URL`.
- Updated default URL from `reward_model:8003` to `reranker:8003`.
- Updated all comments from "reward model" to "cross-encoder reranker".

**LLM Generation (`services/rag/generate.py`)**
- Added `local` provider option that loads the LoRA-adapted LLaMA-3 from `LORA_ADAPTER_PATH` and runs streaming inference via HuggingFace `TextIteratorStreamer`. Falls back to OpenAI if the adapter is not found.
- Replaced generic system prompt with a medical domain prompt that enforces citation-grounded answers and refuses to speculate.

**Medical Papers Agent (`services/agents/research_agent.py`)**
- Added **medRxiv** as a third live source (via `api.biorxiv.org`), alongside PubMed and arXiv.
- Migrated all three API calls to run **concurrently** using `asyncio.gather` (was sequential `ThreadPoolExecutor`).
- Added cross-encoder reranking of all returned papers before returning to Supervisor.
- Renamed primary function to `medical_papers_agent_node`; kept `research_agent_node` as a backward-compatible alias.

**Supervisor Agent (`services/agents/supervisor.py`)**
- Updated routing from 2 routes (`rag_agent`, `research_agent`) to **3 routes**: `rag_agent`, `medical_papers_agent`, `web_search_agent`.
- Updated routing prompt to medical domain context.
- Added **`finalize_node`**: runs after all agents complete, applies the retrieval-confidence guardrail (returns a no-result response if fewer than 2 passages have score > 0.60), and merges all agent outputs via MMR synthesis before passing to the LLM.

**Agent Graph (`services/agents/graph.py`)**
- Added `medical_papers_agent`, `web_search_agent`, and `finalize_node` nodes.
- Updated `AgentState` schema with new fields: `web_results`, `knowledge_graph_context`, `guardrail_triggered`.
- All three agents now converge into `finalize_node` before `END`.

**Agent Service Entry Point (`services/agents/main.py`)**
- Updated initial state to include new fields: `web_results`, `knowledge_graph_context`, `guardrail_triggered`.
- Response now includes `guardrail_triggered` flag.

**Requirements (`services/agents/requirements.txt`)**
- Added: `duckduckgo-search`, `networkx`, `scispacy`.

**Requirements (`services/rag/requirements.txt`)**
- Removed: `chromadb`.
- Added: `qdrant-client`, `peft`, `transformers`.

**Docker Compose (`docker-compose.yml`)**
- Renamed `reward_model` service → `reranker` (build path, container name, port 8003).
- Renamed `chromadb` container → `qdrant`; updated volume from `chroma_data` to `qdrant_data`.
- Added explicit `QDRANT_HOST`, `QDRANT_PORT`, `EMBEDDING_MODEL`, and `RERANKER_URL` environment variables to relevant services.
- Updated `agents` `depends_on` from `reward_model` → `reranker`.

**Architecture Diagram (`ARCHITECTURE.md`)**
- Full rewrite reflecting the finalized 6-layer architecture:
  - QdrantDB replaces ChromaDB throughout.
  - BioMedBERT (768-dim) replaces all-MiniLM-L6-v2 (384-dim) as the encoder.
  - Cross-encoder reranker replaces reward model label.
  - Three agents (RAG, Medical Papers, Web Search) replace the old two-agent system.
  - Knowledge graph (NetworkX + SciSpaCy) added as a dedicated layer.
  - Four-stage fine-tuning pipeline diagram added.
  - Retrieval-confidence guardrail and MMR merge documented in query flow.
