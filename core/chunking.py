from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.schemas import Chunk, ChunkMetadata
import hashlib

class SemanticChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def chunk_document(self, text: str, url: str, title: str) -> list[Chunk]:
        texts = self.splitter.split_text(text)
        chunks = []
        for i, t in enumerate(texts):
            chunk_id = hashlib.md5(f"{url}_{i}".encode()).hexdigest()
            chunks.append(Chunk(
                id=chunk_id,
                text=t,
                metadata=ChunkMetadata(url=url, title=title, chunk_index=i)
            ))
        return chunks