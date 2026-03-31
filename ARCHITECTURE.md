# System Architecture

## Full System Diagram

```mermaid
flowchart TD
    User(["👤 User"])

    %% ─── Frontend ────────────────────────────────────────────────────────────
    subgraph PersonD_UI ["🖥️  Person D — Frontend"]
        Streamlit["Streamlit UI · :8501
        ───────────────────
        • Search bar
        • Streaming answer
        • Source attribution panel
        • NEW / seen dedup indicator
        • Multi-turn session history"]
    end

    %% ─── API Gateway ─────────────────────────────────────────────────────────
    subgraph PersonC_API ["☕  Person C — API Gateway & Ingestion"]
        SpringBoot["Spring Boot API · :8080
        ───────────────────
        GET  /health
        POST /ingest
        POST /search"]

        Kafka[["Kafka Broker · :9092
        ───────────────────
        topic: documents"]]

        KConsumer["Python Kafka Consumer
        ───────────────────
        read → chunk → embed → index"]

        SpringBoot -->|"publish JSON message"| Kafka
        Kafka -->|"consume"| KConsumer
    end

    %% ─── Agent Orchestration ─────────────────────────────────────────────────
    subgraph PersonAD_Agents ["🤖  Person A + D — LangGraph Agent Pipeline"]
        Supervisor["Supervisor Agent
        ───────────────────
        GPT-4o-mini
        Classifies query intent
        Routes to specialist"]

        RAGAgent["📚 RAG Agent
        Retrieval + Rerank
        + Generate"]

        ResearchAgent["🔍 Research Agent
        Live arXiv API lookup
        for out-of-corpus queries"]

        Supervisor -->|"route: rag_agent"| RAGAgent
        Supervisor -->|"route: research_agent"| ResearchAgent
    end

    %% ─── RAG + ML Pipeline ───────────────────────────────────────────────────
    subgraph PersonA_RAG ["🧠  Person A — RAG + ML Pipeline"]
        direction LR
        Dense["Dense Search
        ChromaDB
        vector similarity"]

        BM25["BM25 Search
        keyword matching
        rank-bm25"]

        RRF["⚖️ RRF Merge
        Reciprocal
        Rank Fusion
        k = 60"]

        RewardModel["🏆 Reward Model
        ms-marco-MiniLM
        cross-encoder
        NDCG@5 reranking"]

        Dedup["🔄 Dedup Filter
        Redis cosine cache
        blocks sim > 0.85
        per session"]

        LLM["✨ LLM Generation
        OpenAI / AWS Bedrock
        streaming tokens
        with citations"]

        Dense --> RRF
        BM25  --> RRF
        RRF   --> RewardModel
        RewardModel --> Dedup
        Dedup --> LLM
    end

    %% ─── Embedding Service ───────────────────────────────────────────────────
    subgraph PersonA_Emb ["🔢  Person A — Embedding Service"]
        EmbSvc["Embedding Service · :8001
        ───────────────────
        POST /embed  → 384-dim vector
        POST /index  → store in ChromaDB
        POST /query  → top-N similar chunks
        Model: all-MiniLM-L6-v2"]
    end

    %% ─── Storage ─────────────────────────────────────────────────────────────
    subgraph PersonB_Store ["🗄️  Person B — Storage (Docker / AWS)"]
        ChromaDB[("ChromaDB · :8000
        Vector store
        cosine similarity index")]

        Redis[("Redis · :6379
        Session dedup cache
        embedding history
        TTL: 1 hour")]
    end

    %% ─── Infra ───────────────────────────────────────────────────────────────
    subgraph PersonB_Infra ["⚙️  Person B — Infrastructure"]
        direction LR
        DC["Docker Compose
        Local dev stack
        Weeks 1–8"]

        TF["Terraform
        AWS: VPC · EKS · MSK
        S3 · ECR
        Weeks 9–10"]

        K8s["Kubernetes
        Deployment manifests
        ConfigMap · Services
        Liveness probes"]
    end

    %% ─── Data Collection ─────────────────────────────────────────────────────
    subgraph PersonD_Data ["📄  Person D — Data Collection & Annotation"]
        Scraper["arXiv Scraper
        3,000 papers
        cs.CL · cs.AI · cs.LG
        JSONL output"]

        PDFExtract["PDF Extractor
        500 full texts
        PyMuPDF"]

        AnnotTool["Annotation Tool
        300+ labeled triples
        relevance 0–3
        → reward model training"]
    end

    ArxivAPI(["🌐 arXiv API
    external"])

    %% ════════════════════════════════════════════════════════════════════════
    %% QUERY FLOW  (user asks a question)
    %% ════════════════════════════════════════════════════════════════════════
    User          -->|"1 · search query"| Streamlit
    Streamlit     -->|"2 · POST /search"| SpringBoot
    SpringBoot    -->|"3 · POST /query"| Supervisor
    RAGAgent      -->|"4a · dense"| Dense
    RAGAgent      -->|"4b · sparse"| BM25
    Dense         -->|"5 · POST /query"| EmbSvc
    EmbSvc        -->|"6 · cosine search"| ChromaDB
    BM25          -->|"6 · keyword scan"| ChromaDB
    Dedup         -->|"7 · check cache"| Redis
    LLM           -->|"8 · stream tokens"| SpringBoot
    SpringBoot    -->|"9 · SSE stream"| Streamlit

    %% ════════════════════════════════════════════════════════════════════════
    %% INGESTION FLOW  (document enters the system)
    %% ════════════════════════════════════════════════════════════════════════
    KConsumer     -->|"POST /index chunk"| EmbSvc
    EmbSvc        -->|"upsert vectors"| ChromaDB

    %% ════════════════════════════════════════════════════════════════════════
    %% DATA COLLECTION FLOW
    %% ════════════════════════════════════════════════════════════════════════
    Scraper       <-->|"API calls\n3-sec rate limit"| ArxivAPI
    PDFExtract    -->|"bulk POST /ingest"| SpringBoot
    ResearchAgent <-->|"live search"| ArxivAPI
    AnnotTool     -->|"triples.jsonl"| RewardModel

    %% ════════════════════════════════════════════════════════════════════════
    %% STYLES
    %% ════════════════════════════════════════════════════════════════════════
    classDef personA fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a,rx:8
    classDef personB fill:#dcfce7,stroke:#16a34a,color:#14532d,rx:8
    classDef personC fill:#fef9c3,stroke:#ca8a04,color:#713f12,rx:8
    classDef personD fill:#fce7f3,stroke:#db2777,color:#831843,rx:8
    classDef external fill:#fff7ed,stroke:#ea580c,color:#7c2d12,rx:8

    class Dense,BM25,RRF,RewardModel,Dedup,LLM,RAGAgent,ResearchAgent,Supervisor,EmbSvc personA
    class ChromaDB,Redis,DC,TF,K8s personB
    class SpringBoot,Kafka,KConsumer personC
    class Streamlit,Scraper,PDFExtract,AnnotTool personD
    class ArxivAPI external
```

---

## Query Flow — Step by Step

```mermaid
sequenceDiagram
    actor User
    participant ST  as Streamlit<br/>:8501
    participant SB  as Spring Boot<br/>:8080
    participant SUP as Supervisor<br/>Agent
    participant RA  as RAG Agent
    participant EMB as Embedding<br/>Service :8001
    participant DB  as ChromaDB
    participant RM  as Reward Model<br/>:8003
    participant RD  as Redis
    participant LLM as LLM<br/>OpenAI/Bedrock

    User->>ST: types search query
    ST->>SB: POST /search {query, sessionId}
    SB->>SUP: POST /query
    SUP->>SUP: classify intent<br/>(GPT-4o-mini)
    SUP->>RA: route → rag_agent

    par Dense Search
        RA->>EMB: POST /query {text, n=15}
        EMB->>DB: cosine similarity search
        DB-->>EMB: top-15 vectors
        EMB-->>RA: ranked passages
    and BM25 Search
        RA->>DB: keyword scan
        DB-->>RA: BM25 ranked passages
    end

    RA->>RA: RRF merge<br/>(dense + BM25)
    RA->>RM: POST /rerank {query, candidates}
    RM-->>RA: relevance scores
    RA->>RA: sort by rerank score<br/>→ top 5 passages
    RA->>RD: check dedup cache<br/>(cosine sim > 0.85?)
    RD-->>RA: duplicates flagged
    RA->>LLM: generate answer<br/>with context + citations
    LLM-->>SB: stream tokens (SSE)
    SB-->>ST: forward SSE stream
    ST-->>User: render answer + sources live
```

---

## Ingestion Flow — Step by Step

```mermaid
sequenceDiagram
    actor PersonD as Person D
    participant SC  as arXiv Scraper
    participant PDF as PDF Extractor
    participant SB  as Spring Boot<br/>:8080
    participant KF  as Kafka<br/>documents topic
    participant KC  as Python Consumer
    participant EMB as Embedding<br/>Service :8001
    participant DB  as ChromaDB

    PersonD->>SC: python arxiv_scraper.py
    SC->>SC: fetch 3,000 papers<br/>(cs.CL, cs.AI, cs.LG)
    SC->>PDF: papers.jsonl
    PDF->>PDF: download 500 PDFs<br/>extract full text (PyMuPDF)
    PDF->>SB: POST /ingest per paper

    loop for each document
        SB->>KF: publish JSON message<br/>{doc_id, title, text, ...}
        KF->>KC: consume message
        KC->>KC: chunk text<br/>512 words, 64 overlap
        loop for each chunk
            KC->>EMB: POST /index<br/>{chunk_id, text, metadata}
            EMB->>EMB: encode → 384-dim vector
            EMB->>DB: upsert(chunk_id, vector, text)
        end
    end

    Note over DB: 3,000 papers × ~8 chunks avg<br/>≈ 24,000 vectors indexed
```

---

## AWS Deployment Architecture (Weeks 9–10)

```mermaid
flowchart TD
    Internet(["🌐 Internet"])

    subgraph AWS ["AWS · us-east-1  (Person B — Terraform)"]

        subgraph VPC ["VPC  10.0.0.0/16"]

            subgraph Public ["Public Subnets"]
                IGW["Internet Gateway"]
                LB["LoadBalancer\nfrontend + API"]
            end

            subgraph Private ["Private Subnets"]

                subgraph EKS ["EKS Cluster  (2× t3.small)"]
                    PodEmb["Pod: embedding"]
                    PodRAG["Pod: rag"]
                    PodRM["Pod: reward_model"]
                    PodAgents["Pod: agents"]
                    PodConsumer["Pod: kafka_consumer"]
                    PodSB["Pod: spring_boot_api"]
                    PodFE["Pod: frontend"]
                end

                MSK[["MSK Kafka\nkafka.t3.small\n~$0.082/hr"]]
            end
        end

        ECR[("ECR\nDocker image\nregistry")]
        S3[("S3\nmodel artifacts\nterraform state")]
        CW["CloudWatch\nlogs + metrics"]
    end

    Internet -->|HTTPS| LB
    LB --> PodFE
    LB --> PodSB
    PodSB --> MSK
    MSK --> PodConsumer
    PodConsumer --> PodEmb
    PodSB --> PodAgents
    PodAgents --> PodRAG
    PodRAG --> PodEmb
    PodRAG --> PodRM
    EKS --> ECR
    EKS --> S3
    EKS --> CW

    style AWS fill:#f0f9ff,stroke:#0284c7
    style VPC fill:#e0f2fe,stroke:#0284c7
    style EKS fill:#dbeafe,stroke:#3b82f6
    style Public fill:#fef9c3,stroke:#ca8a04
    style Private fill:#dcfce7,stroke:#16a34a
```

---

## Component Ownership at a Glance

```mermaid
flowchart LR
    subgraph A ["🔵 Person A — ML & RAG Lead"]
        A1["Embedding Service\nservices/embedding/"]
        A2["RAG Pipeline\nservices/rag/"]
        A3["Reward Model\nservices/reward_model/"]
        A4["Agent Graph\nservices/agents/graph.py\nsupervisor.py · rag_agent.py"]
    end

    subgraph B ["🟢 Person B — Infrastructure Lead"]
        B1["Docker Compose\ndocker-compose.yml"]
        B2["Terraform\ninfra/terraform/"]
        B3["Kubernetes\ninfra/kubernetes/"]
        B4["Makefile + CI\n.github/workflows/"]
    end

    subgraph C ["🟡 Person C — Backend & Integration"]
        C1["Spring Boot API\nservices/spring_boot_api/"]
        C2["Kafka Consumer\nservices/kafka_consumer/"]
        C3["Dedup Engine\nservices/rag/dedup.py"]
        C4["Research Agent\nservices/agents/research_agent.py"]
    end

    subgraph D ["🩷 Person D — Data & Frontend"]
        D1["arXiv Scraper\ndata/collection/"]
        D2["Annotation Tool\ndata/annotation/"]
        D3["Streamlit UI\nservices/frontend/"]
        D4["Paper Writing\nabstract · intro · conclusion"]
    end
```
