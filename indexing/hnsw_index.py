import faiss
import numpy as np
from typing import List
import os

class HNSWIndex:
    def __init__(self, dimension: int, m: int = 32, ef_c: int = 200, ef_s: int = 100):
        # M: links per node, efConstruction: build depth
        self.index = faiss.IndexHNSWFlat(dimension, m)
        self.index.hnsw.efConstruction = ef_c
        self.index.hnsw.efSearch = ef_s
        self.id_map = {} # Maps FAISS int ID to Chunk string ID
        self.current_count = 0

    def add(self, embeddings: np.ndarray, chunk_ids: List[str]):
        # FAISS expects float32
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        for cid in chunk_ids:
            self.id_map[self.current_count] = cid
            self.current_count += 1

    def search(self, query_embedding: np.ndarray, top_k: int = 30) -> List[str]:
        if self.current_count == 0:
            return []
        
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        
        # Filter out -1 (FAISS returns -1 if not enough results)
        return [self.id_map[idx] for idx in indices[0] if idx != -1]

    def save(self, filepath: str):
        faiss.write_index(self.index, filepath)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            self.index = faiss.read_index(filepath)