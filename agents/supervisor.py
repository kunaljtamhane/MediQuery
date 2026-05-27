import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

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
        
        # Initialize the fine-tuned edge model
        self.expert_llm = ChatOllama(model="mediquery_local", temperature=0.1)
        
        self.graph = self._build_graph()
        print("[Supervisor] LangGraph Orchestrator Ready.")
        
    def _build_graph(self):
        workflow = StateGraph(MediQueryState)
        
        workflow.add_node("rag_retrieval", self.node_rag)
        workflow.add_node("pubmed_retrieval", self.node_pubmed)
        workflow.add_node("web_retrieval", self.node_web)
        workflow.add_node("synthesize_prompt", self.node_synthesize)
        workflow.add_node("generate_answer", self.node_generate)
        
        workflow.set_entry_point("rag_retrieval")
        workflow.add_edge("rag_retrieval", "pubmed_retrieval")
        workflow.add_edge("pubmed_retrieval", "web_retrieval")
        workflow.add_edge("web_retrieval", "synthesize_prompt")
        workflow.add_edge("synthesize_prompt", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
        
    def node_rag(self, state):
        print("[Supervisor] Routing to RAG Agent...")
        results = self.rag_worker.retrieve(state["query"])
        return {"rag_evidence": results}
        
    def node_pubmed(self, state):
        print("[Supervisor] Routing to PubMed Agent...")
        results = self.pubmed_worker.retrieve_literature(state["query"], top_k=1)
        return {"pubmed_evidence": results}
        
    def node_web(self, state):
        print("[Supervisor] Routing to Web Scraper Agent...")
        results = self.web_worker.search_definitions(state["query"], top_k=1)
        return {"web_evidence": results}
        
    def node_synthesize(self, state):
        print("[Supervisor] Compiling Evidence and Enforcing Protocol...")
        
        # 1. Extract raw text to find common terms (Term Anchoring)
        all_text = state["query"]
        for item in state.get("rag_evidence", []):
            all_text += " " + item.get('context', '')
            
        # Basic extraction of capitalized multi-word phrases as a proxy for medical terms
        potential_terms = set(re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b', all_text))
        anchored_terms = list(potential_terms)[:3] if potential_terms else ["target condition", "treatment"]
        
        # 2. Construct the STRICT prompt
        prompt = "You are a strict biomedical research assistant. You MUST ONLY answer using the provided contexts below.\n"
        prompt += "PERSPECTIVE RULE: You are an independent AI summarizing third-party research. NEVER use first-person pronouns (I, we, our) as if you wrote the papers. Always use third-person phrasing such as 'The paper states', 'The authors found', or 'The study demonstrates'.\n\n"
        prompt += "CRITICAL: If the provided contexts do not contain the answer to the question, or if all contexts are empty (None), you MUST output exactly: 'I do not have enough retrieved evidence to answer this question.' NEVER invent evidence or rely on outside knowledge.\n\n"
        prompt += f"MANDATORY TERMS TO REUSE: {', '.join(anchored_terms)}\n\n"
        
        prompt += "--- INTERNAL VERIFIED MEMORY ---\n"
        if not state.get("rag_evidence"):
            prompt += "None.\n"
        else:
            for item in state.get("rag_evidence", []):
                prompt += f"[Decision: {item.get('final_decision', 'N/A')}] {item.get('context', '')}\n"
            
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
            
        prompt += f"\nOUTPUT FORMAT:\n1. Direct Answer Summary\n2. Evidence-Aligned Explanation\n3. Confidence Level\n\nQuestion: {state['query']}\nAnswer:"
        
        return {"synthesis_prompt": prompt}

    def node_generate(self, state):
        print("[Supervisor] Checking evidence before LLM handoff...")
        
        # THE CIRCUIT BREAKER: If all evidence arrays are empty, bypass the LLM completely.
        if not state.get("rag_evidence") and not state.get("pubmed_evidence") and not state.get("web_evidence"):
            print("[Supervisor] Zero sources found. Triggering hard fallback to prevent hallucination.")
            fallback_msg = "**Insufficient Evidence.**\n\nI do not have enough retrieved evidence to answer this question. Please upload relevant papers to the local database or refine your search terms."
            return {"final_answer": fallback_msg}
            
        print("[Supervisor] Evidence verified. Handing off to Expert LLM for Generation...")
        # If evidence exists, pass the compiled prompt to your local model as normal
        response = self.expert_llm.invoke(state["synthesis_prompt"])
        return {"final_answer": response.content}
        
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

if __name__ == "__main__":
    supervisor = SupervisorAgent()
    test_query = "What is the sensitivity of rapid prescreening for detecting glandular cell abnormalities?"
    
    print(f"\n[USER QUERY] {test_query}\n")
    result = supervisor.execute(test_query)
    
    print("\n=== FINAL GENERATED RESPONSE ===")
    print(result["final_answer"])