from pydantic import BaseModel
from typing import List, Optional


class ChunkRecord(BaseModel):
    id: str
    doc_id: str
    page_num: int
    chunk_index: int
    text: str
    embedding: List[float]
    source_uri: str


class QueryResult(BaseModel):
    answer: str
    sources: List[str]
    matched_chunks: Optional[List[ChunkRecord]] = None


