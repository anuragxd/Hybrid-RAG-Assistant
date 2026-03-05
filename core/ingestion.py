from tavily import TavilyClient
import os
from typing import List, Dict

class WebIngestor:
    def __init__(self):
        self.client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def search_topic(self, topic: str, max_results: int = 20) -> List[Dict[str, str]]:
        response = self.client.search(query=topic, max_results=max_results, search_depth="advanced")
        return [{"url": res["url"], "title": res.get("title", "")} for res in response.get("results", [])]