from core.llm import GroqEngine
from typing import List

class QueryRewriter:
    def __init__(self, llm: GroqEngine):
        self.llm = llm

    def rewrite(self, query: str) -> List[str]:
        prompt = f"Rewrite the following query into 3 distinct search queries to maximize document retrieval recall. Output strictly one query per line, no numbering or markdown.\n\nQuery: {query}"
        response = self.llm.generate(prompt, system_prompt="You are a search query expansion expert.")
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        return queries if queries else [query]