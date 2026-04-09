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
git clone https://github.com/your-repo-name.git

# Navigate to project directory
cd your-repo-name

# Build and run using Docker
docker-compose up --build
