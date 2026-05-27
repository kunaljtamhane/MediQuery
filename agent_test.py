import networkx as nx
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# 1. Initialize the MediQuery Knowledge Graph
G = nx.DiGraph()
G.add_node("Patient_45M", type="Patient", age=45, gender="Male")
G.add_node("Severe Asthma", type="Condition")
G.add_node("Non-selective Beta-blockers", type="MedicationClass")
G.add_edge("Patient_45M", "Severe Asthma", relation="HAS_CONDITION")
G.add_edge("Severe Asthma", "Non-selective Beta-blockers", relation="CONTRAINDICATES", reason="High risk of acute bronchospasm")

# 2. Define the Graph Retrieval Tool
@tool
def check_medical_graph(patient_id: str) -> str:
    """Queries the medical knowledge graph to find patient conditions and medication contraindications. Always use this to check patient history."""
    print(f"\n[System Log] Tool activated! Scanning graph for: {patient_id}...\n")
    
    if patient_id not in G:
        return f"Error: Patient {patient_id} not found in the database."
        
    conditions = [v for u, v, d in G.edges(data=True) if u == patient_id and d.get("relation") == "HAS_CONDITION"]
    
    if not conditions:
        return f"No active conditions found for {patient_id}."

    report = f"Medical Graph Report for {patient_id}:\n"
    for condition in conditions:
        report += f"- Active Condition: {condition}\n"
        contraindicated_meds = [v for u, v, d in G.edges(data=True) if u == condition and d.get("relation") == "CONTRAINDICATES"]
        for med in contraindicated_meds:
            edge_data = G.get_edge_data(condition, med)
            report += f"  [!] CONTRAINDICATION WARNING: {med}\n"
            report += f"  [!] REASON: {edge_data['reason']}\n"
            
    return report

# 3. Boot up the Local LLM and Bind the Tools
print("Booting up LangGraph Router Agent...")

# Using the native tool-calling model as the Manager
llm = ChatOllama(model="llama3.2", temperature=0.1)
tools = [check_medical_graph]

# 4. Create the LangGraph Agent Executor
agent_executor = create_react_agent(llm, tools)

# 5. Run the Clinical Test
query = "I am reviewing the chart for Patient_45M. Look them up in the database. Are there any medications I should completely avoid prescribing to them?"

print(f"\nUser Query: {query}")
print("-" * 50)

# Stream the agent's thought process
events = agent_executor.stream(
    {"messages": [("user", query)]},
    stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()