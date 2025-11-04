"""Text chunking utilities.

This module provides functions for splitting text into chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split pages of text into chunks.

    Args:
        pages: List of page texts to chunk.
        chunk_size: Target size for each chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    full_text = "\n\n".join(pages)
    return splitter.split_text(full_text)
