import networkx as nx
from langchain_core.tools import tool

# 1. Initialize Graph (Same as before)
G = nx.DiGraph()
G.add_node("Patient_45M", type="Patient", age=45, gender="Male")
G.add_node("Severe Asthma", type="Condition")
G.add_node("Non-selective Beta-blockers", type="MedicationClass")
G.add_edge("Patient_45M", "Severe Asthma", relation="HAS_CONDITION")
G.add_edge("Severe Asthma", "Non-selective Beta-blockers", relation="CONTRAINDICATES", reason="High risk of acute bronchospasm")

# 2. Define the LangChain Tool
@tool
def check_medical_graph(patient_id: str) -> str:
    """Queries the medical knowledge graph to find patient conditions and medication contraindications."""
    
    # Check if patient exists
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

# 3. Test the Tool directly
if __name__ == "__main__":
    print(check_medical_graph.invoke({"patient_id": "Patient_45M"}))