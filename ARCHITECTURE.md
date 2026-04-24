# MediQuery — System Architecture

## Full System Diagram

```mermaid
flowchart TD
    User(["👤 User"])

    %% ─── Frontend (Jaya) ────────────────────────────────────────────────────
    subgraph PersonD_UI ["🖥️  Jaya — Streamlit Frontend"]
        Streamlit["Streamlit UI · :8501
        ───────────────────
        • PDF upload
        • Natural language query
        • Streaming answer + citations
        • Document comparison view
        • Knowledge graph visualization
        • Source provenance tags (Uploaded / PubMed / arXiv / medRxiv / Web)"]
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

        SpringBoot -->|"publish JSON"| Kafka
        Kafka -->|"consume"| KConsumer
    end

    %% ─── Agent Orchestration (Kunal) ─────────────────────────────────────────
    subgraph PersonAD_Agents ["🤖  Kunal — LangGraph Multi-Agent Pipeline"]
        Supervisor["Supervisor Agent
        ───────────────────
        GPT-4o-mini routing
        Classifies intent → 3 routes
        Guardrail: ≥2 passages > 0.60
        MMR merge of agent outputs"]

        RAGAgent["📚 RAG Agent
        QdrantDB retrieval (MMR)
        BM25 hybrid + RRF
        Cross-encoder rerank"]

        MedPapersAgent["🔬 Medical Papers Agent
        PubMed + arXiv + medRxiv
        Concurrent async calls
        Cross-encoder rerank"]

        WebSearchAgent["🌐 Web Search Agent
        DuckDuckGo retrieval
        General context lookup"]

        FinalizeNode["⚖️ Finalize Node
        Confidence guardrail
        MMR source merge
        KG context prepend"]

        Supervisor -->|"rag_agent"| RAGAgent
        Supervisor -->|"medical_papers_agent"| MedPapersAgent
        Supervisor -->|"web_search_agent"| WebSearchAgent
        RAGAgent --> FinalizeNode
        MedPapersAgent --> FinalizeNode
        WebSearchAgent --> FinalizeNode
    end

    %% ─── RAG + ML Pipeline ───────────────────────────────────────────────────
    subgraph PersonA_RAG ["🧠  Person A — RAG Pipeline"]
        direction LR
        Dense["Dense Search
        QdrantDB
        BioMedBERT 768-dim
        cosine similarity"]

        BM25["BM25 Search
        in-memory keyword
        rank-bm25"]

        RRF["⚖️ RRF Merge
        Reciprocal Rank Fusion
        k = 60"]

        MMR["🎯 MMR Reranking
        Maximal Marginal
        Relevance
        λ = 0.5"]

        Reranker["🏆 Cross-Encoder Reranker
        ms-marco-MiniLM
        fine-tuned on MedAESQA
        NDCG@5 reranking · :8003"]

        Dedup["🔄 Dedup Filter
        Redis cosine cache
        per session · TTL 1hr"]

        LLM["✨ LLM Generation
        Provider: openai | bedrock | local
        local = LLaMA-3 8B + LoRA adapter
        streaming tokens + citations"]

        Dense --> RRF
        BM25  --> RRF
        RRF   --> MMR
        MMR   --> Reranker
        Reranker --> Dedup
        Dedup --> LLM
    end

    %% ─── Knowledge Graph (Jaya) ──────────────────────────────────────────────
    subgraph PersonD_KG ["🕸️  Jaya — Knowledge Graph"]
        KG["NetworkX + SciSpaCy
        ───────────────────
        Entity extraction (en_core_sci_lg)
        Disease · Drug · Gene · Cell · Organism
        Co-occurrence graph
        Low-confidence guard (≤1 edge = excluded)
        Query-time neighbor context"]
    end

    %% ─── Embedding Service ───────────────────────────────────────────────────
    subgraph PersonA_Emb ["🔢  Person A — Embedding Service"]
        EmbSvc["Embedding Service · :8001
        ───────────────────
        POST /embed  → 768-dim BioMedBERT vector
        POST /index  → store in QdrantDB
        POST /query  → top-N similar chunks
        Model: BioMedBERT (frozen encoder)"]
    end

    %% ─── Fine-Tuning Pipeline ────────────────────────────────────────────────
    subgraph PersonA_FT ["🎓  Kunal — Fine-Tuning Pipeline (offline)"]
        FT["Stage 1: LLaMA-3 8B (decoder)
              BioMedBERT (frozen encoder)
        Stage 2: Domain pre-training
              PubMed + arXiv + medRxiv corpus
        Stage 3: LoRA SFT on 28,136 MedAESQA triples
              r=16 · alpha=32 · attention matrices only
              early stopping on 15% held-out split
        Stage 4: Evaluate F1 · ROUGE-L · NDCG@5
              LLM-as-a-Judge hallucination check
              vs. GPT-4 / Claude zero-shot baseline"]
    end

    %% ─── Storage ─────────────────────────────────────────────────────────────
    subgraph PersonB_Store ["🗄️  Person B — Storage"]
        QdrantDB[("QdrantDB · :6333
        Vector store
        768-dim cosine index
        BioMedBERT embeddings")]

        Redis[("Redis · :6379
        Session dedup cache
        TTL: 1 hour")]
    end

    %% ─── Infra ───────────────────────────────────────────────────────────────
    subgraph PersonB_Infra ["⚙️  Person B — Infrastructure"]
        direction LR
        DC["Docker Compose
        Local dev stack"]

        TF["Terraform
        AWS: VPC · EKS · MSK
        S3 · ECR (stretch goal)"]

        K8s["Kubernetes
        Deployment manifests
        ConfigMap · Services"]
    end

    %% ─── Data Collection (Jaya) ──────────────────────────────────────────────
    subgraph PersonD_Data ["📄  Jaya — Data Collection & Annotation"]
        Scraper["arXiv / PubMed / medRxiv Scrapers
        Corpus for fine-tuning"]

        PDFExtract["Docling PDF Extractor
        512-token chunks · 64 overlap
        User-uploaded PDFs"]

        AnnotTool["MedAESQA Annotation
        28,136 triples
        3-tier negative sampling
        → LoRA fine-tuning signal"]
    end

    ExternalAPIs(["🌐 External APIs
    PubMed · arXiv · medRxiv
    DuckDuckGo"])

    %% ═══ QUERY FLOW ══════════════════════════════════════════════════════════
    User          -->|"query / PDF upload"| Streamlit
    Streamlit     -->|"POST /search"| SpringBoot
    SpringBoot    -->|"POST /query"| Supervisor
    RAGAgent      -->|"dense search"| Dense
    RAGAgent      -->|"keyword search"| BM25
    Dense         -->|"POST /query"| EmbSvc
    EmbSvc        -->|"cosine search"| QdrantDB
    Reranker      -->|"check cache"| Redis
    FinalizeNode  -->|"KG context"| KG
    LLM           -->|"stream tokens"| SpringBoot
    SpringBoot    -->|"SSE stream"| Streamlit

    %% ═══ INGESTION FLOW ══════════════════════════════════════════════════════
    KConsumer     -->|"POST /index chunk"| EmbSvc
    EmbSvc        -->|"upsert vectors"| QdrantDB
    KConsumer     -->|"index_document"| KG

    %% ═══ LIVE RETRIEVAL ══════════════════════════════════════════════════════
    MedPapersAgent <-->|"async API calls"| ExternalAPIs
    WebSearchAgent <-->|"DuckDuckGo"| ExternalAPIs

    %% ═══ DATA PIPELINE ═══════════════════════════════════════════════════════
    Scraper       <-->|"API calls"| ExternalAPIs
    PDFExtract    -->|"POST /ingest"| SpringBoot
    AnnotTool     -->|"triples.jsonl"| FT

    %% ═══ STYLES ══════════════════════════════════════════════════════════════
    classDef personA fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a,rx:8
    classDef personB fill:#dcfce7,stroke:#16a34a,color:#14532d,rx:8
    classDef personC fill:#fef9c3,stroke:#ca8a04,color:#713f12,rx:8
    classDef personD fill:#fce7f3,stroke:#db2777,color:#831843,rx:8
    classDef external fill:#fff7ed,stroke:#ea580c,color:#7c2d12,rx:8

    class Dense,BM25,RRF,MMR,Reranker,Dedup,LLM,RAGAgent,Supervisor,EmbSvc,FT,FinalizeNode personA
    class QdrantDB,Redis,DC,TF,K8s personB
    class SpringBoot,Kafka,KConsumer personC
    class Streamlit,Scraper,PDFExtract,AnnotTool,MedPapersAgent,WebSearchAgent,KG personD
    class ExternalAPIs external
```

---

## Fine-Tuning Pipeline — Four Stages

```mermaid
flowchart LR
    S1["Stage 1\nModel Selection\n───────────────\nDecoder: LLaMA-3 8B\nEncoder: BioMedBERT (frozen)\nFallbacks: BioMedLM · Meditron-7B"]
    S2["Stage 2\nDomain Pre-training\n───────────────\nPubMed + arXiv + medRxiv\nOpen Access abstracts\nDomain vocabulary + summarization"]
    S3["Stage 3\nLoRA SFT\n───────────────\n28,136 MedAESQA triples\n3-tier negatives\nr=16 · alpha=32\n3 epochs · early stop\n15% held-out validation"]
    S4["Stage 4\nEvaluation\n───────────────\nF1 · ROUGE-L · EM\nNDCG@5 · MRR\nLLM-as-a-Judge\nvs. GPT-4 zero-shot baseline"]

    S1 --> S2 --> S3 --> S4
    S4 -->|"below threshold\niterate"| S3
```

---

## Query Flow — Step by Step

```mermaid
sequenceDiagram
    actor User
    participant ST  as Streamlit :8501
    participant SB  as Spring Boot :8080
    participant SUP as Supervisor Agent
    participant RA  as RAG Agent
    participant EMB as Embedding Service :8001
    participant DB  as QdrantDB :6333
    participant RE  as Reranker :8003
    participant KG  as Knowledge Graph
    participant FIN as Finalize Node
    participant LLM as LLaMA-3 (local) or OpenAI

    User->>ST: natural language query
    ST->>SB: POST /search {query, sessionId}
    SB->>SUP: POST /query
    SUP->>SUP: classify intent (GPT-4o-mini)
    SUP->>RA: route → rag_agent

    par Dense Search (BioMedBERT)
        RA->>EMB: POST /query {text, n=15}
        EMB->>DB: cosine search (768-dim)
        DB-->>EMB: top-15 passages
        EMB-->>RA: scored passages
    and BM25 Search
        RA->>DB: scroll all docs for BM25
        DB-->>RA: corpus
    end

    RA->>RA: RRF merge → MMR rerank
    RA->>RE: POST /rerank {query, candidates}
    RE-->>RA: cross-encoder scores
    RA->>FIN: passages + scores

    FIN->>FIN: guardrail check (≥2 passages > 0.60)
    FIN->>KG: get_query_context(query)
    KG-->>FIN: entity neighbor context
    FIN->>LLM: prompt = KG context + passages + query
    LLM-->>SB: stream tokens (SSE)
    SB-->>ST: forward SSE stream
    ST-->>User: answer + citations + source provenance
```

---

## AWS Deployment Architecture (Stretch Goal — Weeks 10-11)

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
                    PodEmb["Pod: embedding\n(BioMedBERT)"]
                    PodRAG["Pod: rag"]
                    PodRE["Pod: reranker\n(cross-encoder)"]
                    PodAgents["Pod: agents"]
                    PodConsumer["Pod: kafka_consumer"]
                    PodSB["Pod: spring_boot_api"]
                    PodFE["Pod: frontend"]
                end
                MSK[["MSK Kafka\nkafka.t3.small"]]
                Qdrant[("QdrantDB\n768-dim vectors")]
            end
        end
        ECR[("ECR\nDocker images")]
        S3[("S3\nLoRA adapter weights\nterraform state")]
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
    PodRAG --> PodRE
    PodEmb --> Qdrant
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

## Component Ownership

```mermaid
flowchart LR
    subgraph K ["🔵 Kunal — ML, RAG & Orchestration"]
        K1["Supervisor Agent\nservices/agents/supervisor.py"]
        K2["Fine-Tuning Pipeline\nservices/fine_tuning/train.py"]
        K3["RAG Agent + Pipeline\nservices/rag/ · services/agents/rag_agent.py"]
        K4["Medical Papers Agent\nservices/agents/research_agent.py"]
        K5["Evaluation Framework\nF1 · ROUGE-L · NDCG@5 · LLM-as-a-Judge"]
        K6["Embedding Service\nservices/embedding/ (BioMedBERT)"]
    end

    subgraph B ["🟢 Person B — Infrastructure"]
        B1["Docker Compose\ndocker-compose.yml"]
        B2["Terraform\ninfra/terraform/"]
        B3["Kubernetes\ninfra/kubernetes/"]
    end

    subgraph C ["🟡 Person C — Backend"]
        C1["Spring Boot API\nservices/spring_boot_api/"]
        C2["Kafka Consumer\nservices/kafka_consumer/"]
    end

    subgraph J ["🩷 Jaya — Data, KG & Frontend"]
        J1["Data Collection\ndata/collection/"]
        J2["MedAESQA Annotation\ndata/annotation/"]
        J3["Web Search Agent\nservices/agents/web_search_agent.py"]
        J4["Knowledge Graph\nservices/agents/knowledge_graph.py"]
        J5["Streamlit UI\nservices/frontend/"]
    end
```
