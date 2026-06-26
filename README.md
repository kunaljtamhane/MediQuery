# MediQuery: Biomedical Multi-Agent RAG System

MediQuery is a domain-specific, multi-agent Retrieval-Augmented Generation (RAG) framework designed to assist clinicians and researchers in synthesizing evidence from complex biomedical literature. By orchestrating specialized agents for local PDF analysis, live PubMed literature retrieval, and web context aggregation, MediQuery ensures grounded, citation-traced responses for high-stakes medical inquiries.

## System Architecture

The system utilizes a multi-agent orchestration layer via LangGraph to manage the retrieval, reasoning, and synthesis workflow.

![MediQuery Architecture](mediqueryAWS.drawio.jpg)

## Key Features

* Multi-Agent Retrieval: Orchestrates three specialized retrieval agents: a local vector-database RAG agent powered by PubMedBERT embeddings in QdrantDB, a live PubMed agent querying the NCBI E-utilities API, and a web search agent providing supplementary context.
* LangGraph Supervisor: Synthesizes heterogeneous evidence before invoking the generative model.
* Dynamic Knowledge Graph: Constructs a dynamic biomedical knowledge graph from retrieved evidence using SciSpaCy and NetworkX to represent entities and their relationships.
* Citation Transparency: Explicitly separates evidence sources during synthesis to preserve provenance and maintain transparency across retrieval channels.
* Domain-Adapted Generation: Utilizes LLaMA-3 8B fine-tuned with parameter-efficient adaptation techniques (LoRA) on the PubMedQA corpus for robust local inference, and Amazon Bedrock (Nova Pro Lite) for production serving.

## Technology Stack

| Component | Technology |
| :--- | :--- |
| Frontend UI | Streamlit |
| Workflow Orchestration | LangGraph |
| Vector Database | Qdrant / Amazon OpenSearch |
| Embedding Model | PubMedBERT |
| Generative Models | AWS Bedrock (Amazon Nova Pro Lite), LLaMA-3 (LoRA Fine-tuned) |
| Knowledge Extraction | SciSpaCy, NetworkX |
| Infrastructure & Pipeline | Docker, Apache Kafka, AWS EC2 |

## Performance Evaluation

MediQuery was evaluated on the PubMedQA benchmark, demonstrating that combining heterogeneous retrieval sources significantly improves semantic alignment with expert references.

| Configuration | ROUGE-1 | BERTScore F1 |
| :--- | :--- | :--- |
| Baseline (No Retrieval) | 0.2902 | 0.8748 |
| Full (Multi-Agent) | 0.5789 | 0.9201 |

## Getting Started

### Prerequisites

* Python 3.10+
* Docker Desktop (for persistent local databases)
* AWS Account (with Bedrock model access approved)

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/kunaljtamhane/MediQuery.git](https://github.com/kunaljtamhane/MediQuery.git)
   cd MediQuery
   ```

2. Install dependencies:
   ``` bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Initialize the local Qdrant container:
   ``` bash
   docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

4. Create a .env file and input your AWS configuration keys.


5. Run the application:
   ``` bash
   streamlit run app.py
   ```

6. Citation: 
If you use this work in your research, please cite:
 ```
 Gorla, J. P. Y., & Tamhane, K. J. (2026). MediQuery: A Domain-specific Multi-Agent Retrieval-Augmented Generation System for Biomedical Research Assistance.
 Developed as a Capstone Project at DePaul University, Jarvis College of Computing and Digital Media.
 ```
