import os
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# CRITICAL: Load environment variables BEFORE importing or initializing LangChain/Bedrock
load_dotenv()

from langchain_aws import ChatBedrock

class PubMedAgent:
    def __init__(self):
        # NCBI E-utilities base URL for querying PubMed
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Initialize the model using the credentials loaded above
        self.llm = ChatBedrock(
            model_id="amazon.nova-pro-v1:0",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        print("[PubMed Agent] Initialized external scientific authority link.")

    def translate_query(self, raw_query):
        """
        Uses the LLM to strip conversational words and generate a strict PubMed Boolean search.
        """
        system_prompt = (
            "You are a medical research librarian. Convert the user's question into a strict, "
            "concise Boolean search query for PubMed. Extract only the core medical terms, drugs, "
            "and conditions. Use AND/OR. Do NOT include conversational filler like 'what is' or 'efficacy of'. "
            "Do NOT include unrelated appended words like 'Mayo Clinic' or 'NIH'. "
            "Return ONLY the raw search string, nothing else."
        )
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_query}
            ])
            translated = response.content.strip()
            print(f"[PubMed Agent] Translated '{raw_query}' -> '{translated}'")
            return translated
        except Exception as e:
            print(f"[PubMed Agent] LLM Translation failed: {e}. Falling back to raw query.")
            return raw_query

    def retrieve_literature(self, query, top_k=3):
        """
        Searches PubMed for the query and retrieves the latest peer-reviewed abstracts.
        """
        clean_query = self.translate_query(query)
        
        search_url = f"{self.base_url}esearch.fcgi?db=pubmed&term={clean_query}&retmode=json&retmax={top_k}"
        try:
            response = requests.get(search_url, timeout=10).json()
            id_list = response.get("esearchresult", {}).get("idlist", [])
        except requests.exceptions.RequestException as e:
            print(f"[PubMed Agent] Network error during search: {e}")
            return []

        if not id_list:
            print(f"[PubMed Agent] 0 PMIDs found for clean query: {clean_query}")
            return []

        ids = ",".join(id_list)
        fetch_url = f"{self.base_url}efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
        
        try:
            fetch_response = requests.get(fetch_url, timeout=10)
            root = ET.fromstring(fetch_response.content)
        except (requests.exceptions.RequestException, ET.ParseError) as e:
            print(f"[PubMed Agent] Error parsing literature data: {e}")
            return []

        results = []

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle")
            
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

if __name__ == "__main__":
    agent = PubMedAgent()
    test_query = "Can you tell me what the efficacy of Pembrolizumab is in treating non-small cell lung cancer? Mayo clinic"
    print(f"\nProcessing query...\n")
    
    literature = agent.retrieve_literature(test_query, top_k=2)
    
    for i, doc in enumerate(literature):
        print(f"--- Document {i+1} (PMID: {doc['pmid']}) ---")
        print(f"Title: {doc['title']}")
        print(f"Abstract Snippet: {doc['content'][:250]}...\n")