from typing import List, BinaryIO, Dict, Optional

from pypdf import PdfReader


def extract_text_per_page(fileobj: BinaryIO) -> List[str]:
    reader = PdfReader(fileobj)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        normalized = " ".join(text.split())
        pages.append(normalized)
    return pages


def extract_metadata(fileobj: BinaryIO) -> Dict[str, Optional[str]]:
    """
    Extract PDF metadata (title, author) from PDF file.
    
    Returns:
        Dict with 'title' and 'author' keys. Values are None if not found.
    """
    try:
        fileobj.seek(0)  # Reset to beginning
        reader = PdfReader(fileobj)
        metadata = reader.metadata
        
        title = None
        author = None
        
        if metadata:
            # Try different metadata fields for title
            title = (
                metadata.get("/Title") or 
                metadata.get("Title") or 
                metadata.get("/Subject") or
                metadata.get("Subject")
            )
            # Remove leading slash if present
            if title and title.startswith("/"):
                title = title[1:]
            
            # Try different metadata fields for author
            author = (
                metadata.get("/Author") or 
                metadata.get("Author") or
                metadata.get("/Creator") or
                metadata.get("Creator")
            )
            # Remove leading slash if present
            if author and author.startswith("/"):
                author = author[1:]
        
        # If no metadata, try extracting from first page text
        if not title or not author:
            try:
                first_page = reader.pages[0]
                first_page_text = first_page.extract_text() or ""
                lines = first_page_text.split("\n")
                
                # Look for title (usually first few lines, all caps or bold)
                if not title and lines:
                    for line in lines[:10]:  # Check first 10 lines
                        line_clean = line.strip()
                        if len(line_clean) > 10 and len(line_clean) < 200:
                            # Likely a title if it's a reasonable length
                            if not title:
                                title = line_clean
                                break
                
                # Look for author (often near title, may contain "by" or "Author:")
                if not author and lines:
                    for i, line in enumerate(lines[:15]):  # Check first 15 lines
                        line_clean = line.strip().lower()
                        if "author:" in line_clean or ("by" in line_clean and i < 5):
                            # Extract author name
                            author_line = lines[i].strip()
                            author = author_line.replace("Author:", "").replace("author:", "").replace("By:", "").replace("by:", "").strip()
                            if author:
                                break
            except:
                pass
        
        return {
            "title": title.strip() if title else None,
            "author": author.strip() if author else None
        }
    except Exception as e:
        return {
            "title": None,
            "author": None
        }


