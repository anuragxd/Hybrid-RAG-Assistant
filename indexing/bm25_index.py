from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import pickle
import os

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunk_ids = []

    def build(self, chunks: list):
        self.chunk_ids = [c.id for c in chunks]
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 30) -> List[str]:
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        # Get top_k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunk_ids[i] for i in top_indices if scores[i] > 0]

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunk_ids': self.chunk_ids}, f)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunk_ids = data['chunk_ids']