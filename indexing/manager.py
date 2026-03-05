import os
import pickle
from typing import List, Dict
from utils.schemas import Chunk
from indexing.bm25_index import BM25Index
from indexing.hnsw_index import HNSWIndex
from core.embeddings import EmbeddingEngine

class IndexManager:
    """Central store for chunks and indexes."""
    def __init__(self, embedder: EmbeddingEngine, hnsw_config: dict):
        self.chunk_store: Dict[str, Chunk] = {}
        self.bm25 = BM25Index()
        self.hnsw = HNSWIndex(
            dimension=embedder.dimension,
            m=hnsw_config.get("hnsw_m", 32),
            ef_c=hnsw_config.get("hnsw_ef_construction", 200),
            ef_s=hnsw_config.get("hnsw_ef_search", 100)
        )
        self.embedder = embedder

    def index_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
        
        # 1. Update Document Store
        for c in chunks:
            self.chunk_store[c.id] = c
            
        # 2. Build BM25
        self.bm25.build(list(self.chunk_store.values()))
        
        # 3. Build HNSW
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)
        self.hnsw.add(embeddings, [c.id for c in chunks])

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self.chunk_store.get(chunk_id)

    def save(self, directory: str = "saved_index"):
        """Persists the entire knowledge base to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # 1. Save Document Store
        with open(os.path.join(directory, "chunk_store.pkl"), "wb") as f:
            pickle.dump(self.chunk_store, f)
            
        # 2. Save HNSW metadata (id_map and count)
        with open(os.path.join(directory, "hnsw_meta.pkl"), "wb") as f:
            pickle.dump({'id_map': self.hnsw.id_map, 'current_count': self.hnsw.current_count}, f)
            
        # 3. Save the actual indexes
        self.hnsw.save(os.path.join(directory, "faiss.index"))
        self.bm25.save(os.path.join(directory, "bm25.pkl"))

    def load(self, directory: str = "saved_index") -> bool:
        """Loads the knowledge base from disk. Returns True if successful."""
        if not os.path.exists(directory):
            return False
            
        try:
            # 1. Load Document Store
            with open(os.path.join(directory, "chunk_store.pkl"), "rb") as f:
                self.chunk_store = pickle.load(f)
                
            # 2. Load HNSW metadata
            with open(os.path.join(directory, "hnsw_meta.pkl"), "rb") as f:
                meta = pickle.load(f)
                self.hnsw.id_map = meta['id_map']
                self.hnsw.current_count = meta['current_count']
                
            # 3. Load the actual indexes
            self.hnsw.load(os.path.join(directory, "faiss.index"))
            self.bm25.load(os.path.join(directory, "bm25.pkl"))
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False