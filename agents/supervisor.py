import os
import re
from typing import TypedDict, Literal
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock

# Automatically search parent directories to find the .env file
load_dotenv(find_dotenv())

# Import the worker agents
from rag_agent import RAGAgent
from pubmed_agent import PubMedAgent
from web_scraper_agent import WebScraperAgent

class MediQueryState(TypedDict):
    query: str
    rag_evidence: list
    pubmed_evidence: list
    web_evidence: list
    synthesis_prompt: str
    final_answer: str

class SupervisorAgent:
    def __init__(self):
        print("[Supervisor] Initializing Orchestrator, Worker Nodes, and Expert LLM...")
        self.rag_worker = RAGAgent()
        self.pubmed_worker = PubMedAgent()
        self.web_worker = WebScraperAgent()
        
        print("[Supervisor] Connecting to Amazon Bedrock...")
        # Main Generator LLM
        self.expert_llm = ChatBedrock(
            model_id="amazon.nova-pro-v1:0", 
            model_kwargs={"temperature": 0.1},
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        
        self.graph = self._build_graph()
        print("[Supervisor] LangGraph Orchestrator Ready.")
        
    def _build_graph(self):
        workflow = StateGraph(MediQueryState)
        
        # Define All Nodes
        workflow.add_node("rag_retrieval", self.node_rag)
        workflow.add_node("pubmed_retrieval", self.node_pubmed)
        workflow.add_node("web_retrieval", self.node_web)
        workflow.add_node("synthesize_prompt", self.node_synthesize)
        workflow.add_node("generate_answer", self.node_generate)
        
        # Linear Sequence: Force all agents to collect evidence without short-circuiting
        workflow.set_entry_point("rag_retrieval")
        workflow.add_edge("rag_retrieval", "pubmed_retrieval")
        workflow.add_edge("pubmed_retrieval", "web_retrieval")
        workflow.add_edge("web_retrieval", "synthesize_prompt")
        workflow.add_edge("synthesize_prompt", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
        
    def node_rag(self, state):
        print("\n[Supervisor] [AGENT 1] RAG Retrieval...")
        results = self.rag_worker.retrieve(state["query"])
        return {"rag_evidence": results}
        
    def node_pubmed(self, state):
        print("\n[Supervisor] [AGENT 2] PubMed Retrieval...")
        results = self.pubmed_worker.retrieve_literature(state["query"], top_k=2)
        # FIX: Only update pubmed_evidence. Do not wipe existing rag_evidence!
        return {"pubmed_evidence": results}

    def node_web(self, state):
        print("\n[Supervisor] [AGENT 3] Web Scraper Retrieval...")
        results = self.web_worker.search_definitions(state["query"], top_k=2)
        # FIX: Only update web_evidence. Do not wipe existing pubmed_evidence!
        return {"web_evidence": results}
        
    def node_synthesize(self, state):
        print("\n[Supervisor] Compiling Final Prompt with ALL Evidence...")
        
        prompt = "You are a strict biomedical research assistant. You MUST ONLY answer using the provided contexts below.\n"
        prompt += "PERSPECTIVE RULE: You are an independent AI summarizing third-party research. NEVER use first-person pronouns (I, we, our).\n\n"
        
        prompt += "--- INTERNAL VERIFIED MEMORY ---\n"
        if not state.get("rag_evidence"):
            prompt += "None.\n"
        else:
            for item in state.get("rag_evidence", []):
                prompt += f"[Context] {item.get('context', '')}\n"
            
        prompt += "\n--- PEER-REVIEWED LITERATURE ---\n"
        if not state.get("pubmed_evidence"):
            prompt += "None.\n"
        else:
            for item in state.get("pubmed_evidence", []):
                prompt += f"[PMID: {item.get('pmid', 'N/A')}] {item.get('content', '')}\n"
            
        prompt += "\n--- SUPPLEMENTARY DEFINITIONS ---\n"
        if not state.get("web_evidence"):
            prompt += "None.\n"
        else:
            for item in state.get("web_evidence", []):
                prompt += f"[Warning: Non-Peer-Reviewed] {item.get('content', '')}\n"
            
        prompt += f"\nOUTPUT FORMAT:\n1. Direct Answer Summary\n2. Evidence-Aligned Explanation\n\nQuestion: {state['query']}\nAnswer:"
        
        return {"synthesis_prompt": prompt}

    def node_generate(self, state):
        print("\n[Supervisor] Generating Final Answer via Bedrock...")
        
        # The Circuit Breaker (Only triggers if ALL agents fail)
        if not state.get("rag_evidence") and not state.get("pubmed_evidence") and not state.get("web_evidence"):
            print(" -> Zero sources found across all tiers. Triggering hard fallback.")
            fallback_msg = "**Insufficient Evidence.**\n\nI was unable to find relevant information in the local database, PubMed, or trusted web sources to answer your query."
            return {"final_answer": fallback_msg}
            
        # Generate the raw response from Amazon Bedrock
        response = self.expert_llm.invoke(state["synthesis_prompt"])
        final_text = response.content
        
        # --- FIX: MULTI-SOURCE TAGGING ---
        # Dynamically build the header based on which agents successfully found data
        sources = []
        if state.get("rag_evidence"):
            sources.append("Local Document Database")
        if state.get("pubmed_evidence"):
            sources.append("PubMed Peer-Reviewed Literature")
        if state.get("web_evidence"):
            sources.append("Web Search (Non-Peer-Reviewed)")
            
        source_header = f"**[Sources Used: {', '.join(sources)}]**\n\n"
        
        return {"final_answer": source_header + final_text}
        
    def execute(self, query):
        initial_state = {
            "query": query, 
            "rag_evidence": [], 
            "pubmed_evidence": [], 
            "web_evidence": [], 
            "synthesis_prompt": "",
            "final_answer": ""
        }
        final_state = self.graph.invoke(initial_state)
        return final_state