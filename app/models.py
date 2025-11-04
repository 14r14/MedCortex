"""Data models for RAG pipeline.

This module defines Pydantic models for chunk records and query results.
"""

from pydantic import BaseModel


class ChunkRecord(BaseModel):
    """Record representing a text chunk with metadata.

    Attributes:
        id: Unique chunk identifier.
        doc_id: Document identifier this chunk belongs to.
        page_num: Page number in the source document.
        chunk_index: Index of chunk within the document.
        text: Chunk text content.
        embedding: Embedding vector for the chunk.
        source_uri: URI of the source document.
    """

    id: str
    doc_id: str
    page_num: int
    chunk_index: int
    text: str
    embedding: list[float]
    source_uri: str


class QueryResult(BaseModel):
    """Query result with answer and source information.

    Attributes:
        answer: Generated answer text.
        sources: List of source URIs.
        matched_chunks: Optional list of matched chunk records.
    """

    answer: str
    sources: list[str]
    matched_chunks: list[ChunkRecord] | None = None
