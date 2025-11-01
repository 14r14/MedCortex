from typing import List, BinaryIO

from pypdf import PdfReader


def extract_text_per_page(fileobj: BinaryIO) -> List[str]:
    reader = PdfReader(fileobj)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        normalized = " ".join(text.split())
        pages.append(normalized)
    return pages


