import json
import PyPDF2
import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from supervisor import SupervisorAgent

# --- HELPER FUNCTION: DYNAMIC ENTITY EXTRACTION ---
def extract_graph_entities(text_chunk, user_query):
    """Passes a text chunk to the local LLM and forces a JSON graph output using few-shot prompting."""
    prompt = f"""You are a strict data extraction AI. Extract key medical and technical entities and their relationships from the text below. 
    
    CRITICAL INSTRUCTION: You MUST extract entities and relationships that specifically relate to the user's query: '{user_query}'.
    Do not generalize. Extract specific drugs, proteins, pathways, and diseases mentioned in the text.
    
    You MUST respond with ONLY valid JSON. Do not include markdown formatting or explanations.
    
    CRITICAL NEGATIVE CONSTRAINT: DO NOT extract author names, universities, cities, countries, or academic departments. Ignore all metadata. Focus ONLY on Diseases, Biological Targets, Algorithms, and Frameworks.
    
    CRITICAL INSTRUCTION: The "edges" array MUST contain dictionaries with "source" and "target" keys. These keys MUST exactly match the "id" of the nodes you create.
    
    EXAMPLE FORMAT:
    {{
        "nodes": [
            {{"id": "Ang-2", "label": "Ang-2", "group": "Biological Target"}},
            {{"id": "Nesvacumab", "label": "Nesvacumab", "group": "Treatment"}}
        ],
        "edges": [
            {{"source": "Nesvacumab", "target": "Ang-2", "label": "inhibits"}}
        ]
    }}
    
    Text to extract from:
    {text_chunk}
    """
    try:
        # --- DEBUG LOGGING ---
        print(f"\n[Knowledge Graph] Sending {len(text_chunk)} characters to local LLM for extraction...")
        
        # We use the supervisor's LLM to do the heavy lifting
        response = st.session_state.supervisor.expert_llm.invoke(prompt)
        
        # --- DEBUG LOGGING ---
        print("[Knowledge Graph] LLM successfully generated a response!")
        
        content = response.content
        
        # ROBUST JSON PARSER: Find everything between the first { and last }
        # This ignores any conversational garbage the LLM spits out before or after the JSON.
        match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if match:
            clean_json = match.group(0)
            return json.loads(clean_json)
        else:
            print("\n[Knowledge Graph Error] No JSON bracket structure found in LLM response.")
            print(f"Raw LLM Output:\n{content}\n")
            return None
            
    except json.JSONDecodeError as e:
        print(f"\n[Knowledge Graph Error] JSON Parsing Failed: {e}")
        print(f"Raw LLM Output:\n{content}\n")
        return None
    except Exception as e:
        print(f"\n[Knowledge Graph Error] Extraction Pipeline Failed: {e}")
        return None

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="MediQuery UI")

# --- CUSTOM CSS FOR THEME & CHAT BUBBLES ---
st.markdown("""
<style>
    /* Force main background to crisp white */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Target only the User Chat Bubble and apply custom color */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stChatMessage"]:has(svg[title="user"]) {
        background-color: #F0FAFD !important;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #e1f3f8; /* Optional subtle border to match */
    }
    
    /* Ensure the sidebar matches the light theme closely */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "supervisor" not in st.session_state:
    st.session_state.supervisor = SupervisorAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
# State to hold the evidence from the most recent query
if "current_evidence" not in st.session_state:
    st.session_state.current_evidence = {"rag": [], "pubmed": [], "web": []}
# State to hold dynamic graph data
if "dynamic_graph" not in st.session_state:
    st.session_state.dynamic_graph = None

# --- SIDEBAR: Navigation & Uploads ---
with st.sidebar:
    # Use the exact filename you saved the image as
    st.image("mediquery_logo.png", use_container_width=True)
    st.divider()
    
    st.write("**NAVIGATION**")
    nav = st.radio("", ["Literature search", "Compare papers", "Summarise corpus", "Knowledge graph"])
    st.divider()
    
    st.write("**MY PAPERS**")
    uploaded_file = st.file_uploader("Drop PDFs or click to upload", type="pdf", key="unique_sidebar_uploader")
    
    if st.button("+ Add papers", type="primary", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    full_text = ""
                    for page in pdf_reader.pages:
                        full_text += page.extract_text() + "\n"
                    
                    chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
                    
                    # 1. Standard Vector Ingestion
                    if hasattr(st.session_state.supervisor.rag_worker, 'ingest_chunks'):
                        st.session_state.supervisor.rag_worker.ingest_chunks(chunks, source=uploaded_file.name)
                        st.success(f"Successfully added {len(chunks)} chunks to Qdrant!")
                    else:
                        st.warning("Please add an 'ingest_chunks' method to your RagAgent.")
                        
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
        else:
            st.error("Please upload a PDF first.")

# --- MAIN LAYOUT: Central Chat & Right Citation Panel ---
col1, col2 = st.columns([3, 1])

# --- LEFT COLUMN: Dynamic Main Interface ---
with col1:
    st.header(nav)
    
    # -----------------------------------------
    # ROUTE 1: LITERATURE SEARCH (The Chat)
    # -----------------------------------------
    if nav == "Literature search":
        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle new user input
        if prompt := st.chat_input("Ask about findings, gaps, methods, or request a summary..."):
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Agents are researching..."):
                    result = st.session_state.supervisor.execute(prompt)
                    response = result["final_answer"]
                    
                    # Save the new evidence to session state so the right panel can read it
                    st.session_state.current_evidence = {
                        "rag": result.get("rag_evidence", []),
                        "pubmed": result.get("pubmed_evidence", []),
                        "web": result.get("web_evidence", [])
                    }
                    
                    st.markdown(response)
                    
            # --- NEW KNOWLEDGE GRAPH TRIGGER ---
            # Now that we have retrieved the exact RAG evidence for this prompt, generate the graph.
            with st.spinner("Mapping entities for the Knowledge Graph..."):
                rag_matches = result.get("rag_evidence", [])
                if rag_matches:
                    # --- VRAM SAFETY FIX: Grab only top 2 matches ---
                    top_matches = rag_matches[:2]
                    compiled_rag_context = "\n\n".join([match.get('context', '') for match in top_matches])
                    
                    # Pass the compiled context AND the user's prompt to the LLM
                    graph_data = extract_graph_entities(compiled_rag_context, prompt)
                    
                    if graph_data:
                        st.session_state.dynamic_graph = graph_data
                        st.toast("Knowledge Graph Updated!", icon="🧠")
                else:
                    # If RAG found nothing, clear the graph state to prevent old data from lingering
                    st.session_state.dynamic_graph = None

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun() # Force UI to update the right panel immediately

    # -----------------------------------------
    # ROUTE 2: KNOWLEDGE GRAPH
    # -----------------------------------------
    elif nav == "Knowledge graph":
        st.write("Interactive entity mapping extracted dynamically from your active documents.")
        
        # Check if we have dynamically extracted data
        if not st.session_state.dynamic_graph:
            with st.container(border=True):
                st.info("Ask a question in the Literature Search tab to map entities for the Knowledge Graph!")
        else:
            # 1. Parse dynamic nodes (DEFENSIVE)
            nodes = []
            node_ids = set() # NEW: Keep track of every valid node ID
            
            for n in st.session_state.dynamic_graph.get("nodes", []):
                # Check if the LLM followed instructions and gave us a dictionary
                if isinstance(n, dict):
                    color = "#003DA5" if n.get("group") in ["Framework", "Algorithm"] else "#4CAF50" 
                    # Use .get() safely, fallback to str(n) just in case 'id' is missing
                    node_id = str(n.get("id", "Unknown"))
                    node_label = str(n.get("label", node_id))
                    
                    nodes.append(Node(id=node_id, label=node_label, size=20, color=color))
                    node_ids.add(node_id) # Add to our registry
                
                # Fallback: If the LLM just gave us a list of raw strings
                elif isinstance(n, str):
                    nodes.append(Node(id=n, label=n, size=20, color="#4CAF50"))
                    node_ids.add(n) # Add to our registry

            # 2. Parse dynamic edges (BULLETPROOF)
            edges = []
            for e in st.session_state.dynamic_graph.get("edges", []):
                if isinstance(e, dict):
                    # Forgive the LLM if it capitalizes keys unexpectedly
                    source = str(e.get("source") or e.get("Source"))
                    target = str(e.get("target") or e.get("Target"))
                    label = str(e.get("label") or e.get("Label", ""))
                    
                    # NEW: Only draw the edge if BOTH the source and target actually exist!
                    if source in node_ids and target in node_ids:
                        edges.append(Edge(source=source, target=target, label=label))
                    else:
                        print(f"[Knowledge Graph Warning] Skipped invalid edge: {source} -> {target}")
            
            # 3. Configure and Render
            if not nodes:
                 st.warning("The AI extracted data, but no valid nodes were found. Please try another query.")
            else:
                config = Config(
                    width="100%", height=500, directed=True, physics=True, hierarchical=False,
                    nodeHighlightBehavior=True, highlightColor="#F7A7A6"
                )
                
                with st.container(border=True):
                    agraph(nodes=nodes, edges=edges, config=config)

    # -----------------------------------------
    # ROUTE 3 & 4: OTHER TABS (Placeholders)
    # -----------------------------------------
    elif nav == "Compare papers":
        st.write("Paper comparison matrix will render here.")
    elif nav == "Summarise corpus":
        st.write("Global corpus summary will render here.")

# --- RIGHT COLUMN: Dynamic Citation Panel ---
with col2:
    st.subheader("Retrieved sources")
    
    # Calculate total sources found
    total_sources = len(st.session_state.current_evidence["rag"]) + \
                    len(st.session_state.current_evidence["pubmed"]) + \
                    len(st.session_state.current_evidence["web"])
    
    st.write(f"**{total_sources} sources** compiled by agents")
    
    # Create tabs mapping to your 3 evidence streams
    tab_rag, tab_pubmed, tab_web = st.tabs(["Local DB", "PubMed", "Web"])
    
    with tab_rag:
        if not st.session_state.current_evidence["rag"]:
            st.info("No local database matches found.")
        else:
            for i, ev in enumerate(st.session_state.current_evidence["rag"]):
                with st.container(border=True): # Creates the "Card" look
                    st.markdown("**Internal Vector Match**")
                    # Display a truncated snippet of the context
                    snippet = ev.get('context', '')[:150] + "..." 
                    st.write(snippet)
                    # Simulated badge
                    st.caption("🟢 RAG Agent")

    with tab_pubmed:
        if not st.session_state.current_evidence["pubmed"]:
            st.info("No PubMed literature retrieved.")
        else:
            for ev in st.session_state.current_evidence["pubmed"]:
                with st.container(border=True):
                    st.markdown(f"**PMID: {ev.get('pmid', 'N/A')}**")
                    snippet = ev.get('content', '')[:150] + "..."
                    st.write(snippet)
                    st.caption("🔵 PubMed Agent")
                    
    with tab_web:
        if not st.session_state.current_evidence["web"]:
            st.info("No web definitions retrieved.")
        else:
            for i, ev in enumerate(st.session_state.current_evidence["web"]):
                with st.container(border=True):
                    st.markdown("**Web Snippet**")
                    snippet = ev.get('content', '')[:150] + "..."
                    st.write(snippet)
                    st.caption("🟠 Web Scraper Agent")
    
    st.divider()
    st.button("Copy refs", type="primary", use_container_width=True)
    st.button("Export report", type="primary", use_container_width=True)