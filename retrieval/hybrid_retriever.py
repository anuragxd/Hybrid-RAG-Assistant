from indexing.manager import IndexManager
from retrieval.query_rewriter import QueryRewriter
from retrieval.reranker import NeuralReranker
from retrieval.rrf_fusion import compute_rrf
from utils.schemas import RetrievedChunk
from typing import List

class HybridRetriever:
    """Orchestrates the multi-stage retrieval pipeline."""
    def __init__(
        self, 
        index_manager: IndexManager, 
        rewriter: QueryRewriter, 
        reranker: NeuralReranker,
        config: dict
    ):
        self.index_manager = index_manager
        self.rewriter = rewriter
        self.reranker = reranker
        self.top_k_retrieval = config.get("top_k_retrieval", 30)
        self.top_k_rerank = config.get("top_k_rerank", 5)
        self.rrf_k = config.get("rrf_k", 60)

    def retrieve(self, original_query: str) -> List[RetrievedChunk]:
        # 1. Expand query
        queries = self.rewriter.rewrite(original_query)
        if original_query not in queries:
            queries.append(original_query)
            
        all_bm25_ids = []
        all_hnsw_ids = []
        
        # 2. Parallel Search (Simulated via loop for safety, can use ThreadPoolExecutor)
        for q in queries:
            all_bm25_ids.extend(self.index_manager.bm25.search(q, top_k=self.top_k_retrieval))
            query_emb = self.index_manager.embedder.encode([q])[0]
            all_hnsw_ids.extend(self.index_manager.hnsw.search(query_emb, top_k=self.top_k_retrieval))
            
        # 3. Fuse Results (RRF)
        # We treat the aggregated BM25 and HNSW lists as our two ranked lists
        fused_ids = compute_rrf([all_bm25_ids, all_hnsw_ids], k=self.rrf_k)
        
        # 4. Map IDs back to actual Chunk objects
        # This solves the `chunk_lookup is not defined` error
        candidate_chunks = [
            self.index_manager.get_chunk(cid) 
            for cid in fused_ids[:self.top_k_retrieval] 
            if self.index_manager.get_chunk(cid) is not None
        ]
        
        # 5. Cross-Encoder Reranking
        final_chunks = self.reranker.rerank(original_query, candidate_chunks, top_k=self.top_k_rerank)
        
        return final_chunks