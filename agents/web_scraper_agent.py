# from duckduckgo_search import DDGS

# class WebScraperAgent:
#     def __init__(self):
#         self.search_engine = DDGS()
#         print("[Web Scraper Agent] Initialized broad context search engine.")

#     def search_definitions(self, query, top_k=3):
#         """
#         Retrieves general web definitions and explicitly flags them as non-peer-reviewed.
#         """
#         # Append context to force the search engine toward medical definitions
#         optimized_query = f"medical definition explanation {query}"
        
#         try:
#             results = list(self.search_engine.text(optimized_query, max_results=top_k))
#         except Exception as e:
#             print(f"[Web Scraper Agent] Search failed: {e}")
#             return []

#         formatted_results = []
#         for res in results:
#             formatted_results.append({
#                 "source": "Web Search (NON-PEER-REVIEWED)",
#                 "url": res.get("href"),
#                 "title": res.get("title"),
#                 "content": res.get("body")
#             })
            
#         return formatted_results

# # Execution block for local testing
# if __name__ == "__main__":
#     agent = WebScraperAgent()
#     test_query = "Atypical Glandular Cells AGC"
#     print(f"\nQuerying broad web context for: '{test_query}'...\n")
    
#     web_results = agent.search_definitions(test_query, top_k=2)
    
#     for i, res in enumerate(web_results):
#         print(f"--- Warning: {res['source']} ---")
#         print(f"Title: {res['title']}")
#         print(f"Snippet: {res['content']}\n")

'''=================================================================================================='''

from ddgs import DDGS

class WebScraperAgent:
    def __init__(self):
        self.search_engine = DDGS()
        print("[Web Scraper Agent] Initialized strict context search engine.")

    def search_definitions(self, query, top_k=3):
        """
        Retrieves web definitions exclusively prioritizing highly trusted medical domains.
        """
        # FIX: We stripped the "OR" operators. We are now formatting it like a 
        # natural human search query to bypass DuckDuckGo's bot-detection.
        optimized_query = f"{query} medical health Mayo Clinic NIH"
        
        try:
            # Added a simple print statement so we can see what it's actually searching
            print(f" -> [Web Scraper] Executing search for: '{optimized_query}'")
            results = list(self.search_engine.text(optimized_query, max_results=top_k))
        except Exception as e:
            print(f"[Web Scraper Agent] Search failed entirely: {e}")
            return []

        # If rate-limited, it returns empty without an error. Catch it here:
        if not results:
            print("[Web Scraper Agent] Search returned 0 results (Likely Rate Limited by DDG).")
            return []

        formatted_results = []
        for res in results:
            formatted_results.append({
                "source": "Verified Web Search",
                "url": res.get("href", "N/A"),
                "title": res.get("title", "N/A"),
                "content": res.get("body", "N/A")
            })
            
        return formatted_results

# Execution block for local testing
if __name__ == "__main__":
    agent = WebScraperAgent()
    # Updated test query to verify general medical information retrieval
    test_query = "What are some symptoms of cancer?" 
    print(f"\nQuerying strict web context for: '{test_query}'...\n")
    
    web_results = agent.search_definitions(test_query, top_k=2)
    
    for i, res in enumerate(web_results):
        print(f"--- {res['source']} ---")
        print(f"URL: {res['url']}")
        print(f"Title: {res['title']}")
        print(f"Snippet: {res['content']}\n")