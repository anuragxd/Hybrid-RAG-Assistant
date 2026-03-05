from sentence_transformers import CrossEncoder
from utils.schemas import Chunk, RetrievedChunk
from typing import List

class NeuralReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[RetrievedChunk]:
        if not chunks:
            return []
            
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        retrieved = []
        for i, chunk in enumerate(chunks):
            rc = RetrievedChunk(**chunk.model_dump())
            rc.rerank_score = float(scores[i])
            retrieved.append(rc)
            
        return sorted(retrieved, key=lambda x: x.rerank_score, reverse=True)[:top_k]