from pydantic import BaseModel, Field
from typing import Dict, Any, List

class ChunkMetadata(BaseModel):
    url: str
    title: str = ""
    chunk_index: int

class Chunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata

class RetrievedChunk(Chunk):
    score: float = 0.0
    rerank_score: float = 0.0