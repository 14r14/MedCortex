from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    full_text = "\n\n".join(pages)
    return splitter.split_text(full_text)


