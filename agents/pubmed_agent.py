import requests
import xml.etree.ElementTree as ET

class PubMedAgent:
    def __init__(self):
        # NCBI E-utilities base URL for querying PubMed
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        print("[PubMed Agent] Initialized external scientific authority link.")

    def retrieve_literature(self, query, top_k=3):
        """
        Searches PubMed for the query and retrieves the latest peer-reviewed abstracts.
        """
        # Step 1: Search for relevant PubMed IDs (PMIDs)
        search_url = f"{self.base_url}esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax={top_k}"
        try:
            response = requests.get(search_url, timeout=10).json()
            id_list = response.get("esearchresult", {}).get("idlist", [])
        except requests.exceptions.RequestException as e:
            print(f"[PubMed Agent] Network error during search: {e}")
            return []

        if not id_list:
            return []

        # Step 2: Fetch the full XML records for the retrieved PMIDs
        ids = ",".join(id_list)
        fetch_url = f"{self.base_url}efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
        
        try:
            fetch_response = requests.get(fetch_url, timeout=10)
            root = ET.fromstring(fetch_response.content)
        except (requests.exceptions.RequestException, ET.ParseError) as e:
            print(f"[PubMed Agent] Error parsing literature data: {e}")
            return []

        results = []

        # Step 3: Parse the XML to extract the structured abstract
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle")
            
            # PubMed abstracts are often divided into structured sections
            abstract_texts = article.findall(".//AbstractText")
            abstract_sections = []
            for elem in abstract_texts:
                label = elem.get("Label", "")
                text = elem.text if elem.text else ""
                if label:
                    abstract_sections.append(f"{label}: {text}")
                else:
                    abstract_sections.append(text)
            
            full_abstract = " ".join(abstract_sections)

            if full_abstract:
                results.append({
                    "source": "PubMed (Peer-Reviewed)",
                    "pmid": pmid,
                    "title": title,
                    "content": full_abstract
                })

        return results

# Execution block for local testing
if __name__ == "__main__":
    agent = PubMedAgent()
    test_query = "efficacy of rapid prescreening glandular cell abnormalities"
    print(f"\nQuerying live PubMed database for: '{test_query}'...\n")
    
    literature = agent.retrieve_literature(test_query, top_k=2)
    
    for i, doc in enumerate(literature):
        print(f"--- Document {i+1} (PMID: {doc['pmid']}) ---")
        print(f"Title: {doc['title']}")
        print(f"Abstract Snippet: {doc['content'][:250]}...\n")