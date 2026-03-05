from typing import List, Dict

def compute_rrf(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    """Combines multiple ranked lists of chunk IDs using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
            
    # Sort by descending RRF score
    sorted_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in sorted_ids]