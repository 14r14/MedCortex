from typing import List, BinaryIO, Dict, Optional
import re
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Try to import pdfplumber for better font/position analysis
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available, falling back to pypdf-based extraction")


def extract_text_per_page(fileobj: BinaryIO) -> List[str]:
    reader = PdfReader(fileobj)
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        normalized = " ".join(text.split())
        pages.append(normalized)
    return pages


def _is_non_title_pattern(text: str) -> bool:
    """Check if text matches common non-title patterns."""
    text_lower = text.lower().strip()
    
    # Common prefixes that indicate non-title content
    non_title_prefixes = [
        "abstract", "keywords", "introduction", "background", 
        "methodology", "results", "discussion", "conclusion",
        "references", "acknowledgment", "acknowledgments",
        "1.", "2.", "3.", "i.", "ii.", "iii.",
        "figure", "table", "page", "doi:", "http", "www.",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    
    # Check if starts with non-title prefix
    for prefix in non_title_prefixes:
        if text_lower.startswith(prefix):
            return True
    
    # Check if it's just a date (e.g., "2024", "January 2024")
    if re.match(r'^\d{4}$|^[A-Za-z]+\s+\d{4}$', text_lower):
        return True
    
    # Check if it's a URL or email
    if re.search(r'http://|https://|www\.|@', text):
        return True
    
    # Check if it's too short (likely not a title)
    if len(text.strip()) < 15:
        return True
    
    # Check if it's too long (likely abstract or body text)
    if len(text.strip()) > 250:
        return True
    
    # Check if it has too many special characters (likely not a title)
    special_char_count = len(re.findall(r'[^\w\s]', text))
    if special_char_count > len(text) * 0.3:  # More than 30% special chars
        return True
    
    return False


def _is_title_case(text: str) -> bool:
    """Check if text follows title case pattern (most words capitalized)."""
    words = text.split()
    if not words:
        return False
    
    # Filter out common small words
    small_words = {'a', 'an', 'and', 'as', 'at', 'by', 'for', 'from',
                   'in', 'into', 'of', 'on', 'or', 'the', 'to', 'with'}
    
    capitalized_words = 0
    total_significant_words = 0
    
    for word in words:
        # Remove punctuation for checking
        word_clean = re.sub(r'[^\w]', '', word)
        if not word_clean:
            continue
        
        if word_clean.lower() not in small_words:
            total_significant_words += 1
            if word_clean[0].isupper():
                capitalized_words += 1
    
    # Title case: at least 60% of significant words should be capitalized
    if total_significant_words > 0:
        return (capitalized_words / total_significant_words) >= 0.6
    
    return False


def _is_all_caps(text: str) -> bool:
    """Check if text is all uppercase (common for titles)."""
    # Remove punctuation and check
    text_clean = re.sub(r'[^\w\s]', '', text)
    if not text_clean:
        return False
    
    # Check if all letters are uppercase
    letters = [c for c in text_clean if c.isalpha()]
    if not letters:
        return False
    
    return all(c.isupper() for c in letters)


def _extract_by_font_analysis(fileobj: BinaryIO) -> Optional[str]:
    """Extract title using font size/weight analysis with pdfplumber."""
    if not PDFPLUMBER_AVAILABLE:
        return None
    
    try:
        fileobj.seek(0)
        with pdfplumber.open(fileobj) as pdf:
            if len(pdf.pages) == 0:
                return None
            
            first_page = pdf.pages[0]
            chars = first_page.chars
            
            if not chars:
                return None
            
            # Group characters by font size and position
            # Title is usually in top 1/3 of page with largest font
            page_height = first_page.height
            top_y_threshold = page_height * 0.7  # Top 30% of page
            
            # Find font sizes
            font_sizes = {}
            for char in chars:
                if char['y0'] > top_y_threshold:  # Only look at top portion
                    font_size = char.get('size', 0)
                    if font_size > 0:
                        if font_size not in font_sizes:
                            font_sizes[font_size] = []
                        font_sizes[font_size].append(char)
            
            if not font_sizes:
                return None
            
            # Get largest font size (likely title)
            max_font_size = max(font_sizes.keys())
            title_chars = font_sizes[max_font_size]
            
            # Sort by position (top to bottom, left to right)
            title_chars.sort(key=lambda c: (-c['y0'], c['x0']))
            
            # Extract text from these characters
            title_lines = []
            current_line = []
            current_y = None
            
            for char in title_chars:
                char_y = char['y0']
                
                # Group characters on same line (within 5 pixels)
                if current_y is None or abs(char_y - current_y) < 5:
                    current_line.append(char)
                    current_y = char_y
                else:
                    # New line
                    if current_line:
                        line_text = ''.join(c['text'] for c in sorted(current_line, key=lambda c: c['x0']))
                        if line_text.strip():
                            title_lines.append(line_text.strip())
                    current_line = [char]
                    current_y = char_y
            
            # Add last line
            if current_line:
                line_text = ''.join(c['text'] for c in sorted(current_line, key=lambda c: c['x0']))
                if line_text.strip():
                    title_lines.append(line_text.strip())
            
            # Combine lines to form title
            if title_lines:
                # Take first 3 lines max (titles rarely span more)
                title_candidate = ' '.join(title_lines[:3])
                
                # Validate it's not a non-title pattern
                if not _is_non_title_pattern(title_candidate):
                    # Check if it's reasonable length
                    if 15 <= len(title_candidate) <= 250:
                        return title_candidate.strip()
            
            return None
    except Exception as e:
        logger.warning(f"Font-based extraction failed: {e}")
        return None


def _extract_by_position_pypdf(fileobj: BinaryIO) -> Optional[str]:
    """Extract title using position-based heuristics with pypdf (fallback)."""
    try:
        fileobj.seek(0)
        reader = PdfReader(fileobj)
        if len(reader.pages) == 0:
            return None
        
        first_page = reader.pages[0]
        first_page_text = first_page.extract_text() or ""
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
        
        if not lines:
            return None
        
        # Check first 15 lines (titles are usually in top portion)
        candidates = []
        for i, line in enumerate(lines[:15]):
            line_clean = line.strip()
            
            # Skip if it's a non-title pattern
            if _is_non_title_pattern(line_clean):
                continue
            
            # Validate length
            if not (15 <= len(line_clean) <= 250):
                continue
            
            # Score based on position (earlier = better)
            score = 100 - (i * 5)
            
            # Bonus for title case or all caps
            if _is_title_case(line_clean):
                score += 20
            if _is_all_caps(line_clean):
                score += 10
            
            candidates.append((line_clean, score, i))
        
        if candidates:
            # Sort by score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Check if we can combine consecutive lines (multi-line title)
            best_candidate = candidates[0][0]
            best_idx = candidates[0][2]
            
            # Try to combine with next line if it's similar
            combined_title = best_candidate
            if best_idx + 1 < len(lines):
                next_line = lines[best_idx + 1].strip()
                if not _is_non_title_pattern(next_line):
                    # Check if next line looks like continuation
                    if 10 <= len(next_line) <= 150:
                        # Check if it's not likely author name (contains "by" or "author")
                        if "by " not in next_line.lower() and "author:" not in next_line.lower():
                            combined_title = f"{best_candidate} {next_line}"
                            if len(combined_title) <= 250:
                                best_candidate = combined_title
            
            return best_candidate
        
        return None
    except Exception as e:
        logger.warning(f"Position-based extraction failed: {e}")
        return None


def _is_non_author_pattern(text: str) -> bool:
    """Check if text matches common non-author patterns."""
    text_lower = text.lower().strip()
    
    # Common software names that might be in Creator field
    software_names = [
        "adobe", "microsoft", "latex", "pdf", "word", "excel", "powerpoint",
        "acrobat", "reader", "creator", "author", "unknown", "n/a", "na",
        "libreoffice", "openoffice", "google", "apple", "pages", "keynote"
    ]
    
    # Check if it's a software name
    if any(software in text_lower for software in software_names):
        return True
    
    # Common prefixes that indicate non-author content
    non_author_prefixes = [
        "abstract", "keywords", "introduction", "background", "methodology",
        "results", "discussion", "conclusion", "references", "acknowledgment",
        "1.", "2.", "3.", "i.", "ii.", "iii.", "figure", "table", "page",
        "doi:", "http", "www.", "january", "february", "march", "april",
        "may", "june", "july", "august", "september", "october", "november", "december"
    ]
    
    # Check if starts with non-author prefix
    for prefix in non_author_prefixes:
        if text_lower.startswith(prefix):
            return True
    
    # Check if it's just a date (e.g., "2024", "January 2024")
    if re.match(r'^\d{4}$|^[A-Za-z]+\s+\d{4}$', text_lower):
        return True
    
    # Check if it's a URL or email
    if re.search(r'http://|https://|www\.|@', text):
        return True
    
    # Check if it's too short (likely not a name)
    if len(text.strip()) < 3:
        return True
    
    # Check if it's too long (likely not a name)
    if len(text.strip()) > 200:
        return True
    
    # Check if it has too many special characters (likely not a name)
    special_char_count = len(re.findall(r'[^\w\s]', text))
    if special_char_count > len(text) * 0.3:  # More than 30% special chars
        return True
    
    return False


def _is_author_name(text: str) -> bool:
    """Check if text looks like an author name."""
    if _is_non_author_pattern(text):
        return False
    
    # Check length
    if not (5 <= len(text.strip()) <= 150):
        return False
    
    # Check if it contains letters
    if not re.search(r'[a-zA-Z]', text):
        return False
    
    # Check if it has reasonable word count (typically 2-10 words for author names)
    words = text.split()
    if not (2 <= len(words) <= 15):
        return False
    
    # Check if it has capital letters (names typically have capitals)
    if not re.search(r'[A-Z]', text):
        return False
    
    # Check if it doesn't contain only numbers or symbols
    if re.match(r'^[\d\s\W]+$', text):
        return False
    
    return True


def _extract_multiple_authors(text: str) -> str:
    """Extract and format multiple authors from text."""
    # Remove parentheticals (affiliations, emails, etc.)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Handle "et al." patterns
    if 'et al' in text.lower():
        text = text.split('et al')[0].strip()
    
    # Split by common separators
    authors = []
    
    # Try comma separation first
    if ',' in text:
        parts = [p.strip() for p in text.split(',')]
        # Filter out parts that don't look like names
        for part in parts:
            if _is_author_name(part):
                authors.append(part)
    # Try "and" separation
    elif ' and ' in text.lower():
        parts = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if _is_author_name(part):
                authors.append(part)
    # Try semicolon separation
    elif ';' in text:
        parts = [p.strip() for p in text.split(';')]
        for part in parts:
            if _is_author_name(part):
                authors.append(part)
    else:
        # Single author
        if _is_author_name(text):
            authors.append(text)
    
    # Limit to max 5 authors, then "et al."
    if len(authors) > 5:
        authors = authors[:5]
        return ', '.join(authors) + ', et al.'
    
    return ', '.join(authors) if authors else text


def _extract_author_by_font(fileobj: BinaryIO, title_y: Optional[float] = None) -> Optional[str]:
    """Extract author using font size/weight analysis with pdfplumber."""
    if not PDFPLUMBER_AVAILABLE:
        return None
    
    try:
        fileobj.seek(0)
        with pdfplumber.open(fileobj) as pdf:
            if len(pdf.pages) == 0:
                return None
            
            first_page = pdf.pages[0]
            chars = first_page.chars
            
            if not chars:
                return None
            
            # Group characters by font size and position
            # Authors are usually in top portion, after title, with medium font size
            page_height = first_page.height
            top_y_threshold = page_height * 0.7  # Top 30% of page
            
            # Find font sizes
            font_sizes = {}
            for char in chars:
                if char['y0'] > top_y_threshold:  # Only look at top portion
                    # If we know title position, look for text below title
                    if title_y and char['y0'] >= title_y:
                        continue  # Skip if above or at title position
                    
                    font_size = char.get('size', 0)
                    if font_size > 0:
                        if font_size not in font_sizes:
                            font_sizes[font_size] = []
                        font_sizes[font_size].append(char)
            
            if not font_sizes:
                return None
            
            # Get font sizes (excluding the largest, which is likely the title)
            sorted_font_sizes = sorted(font_sizes.keys(), reverse=True)
            
            # Authors are typically in the second or third largest font
            # (after title, but larger than body text)
            candidate_font_sizes = sorted_font_sizes[1:3] if len(sorted_font_sizes) > 1 else sorted_font_sizes[:1]
            
            author_candidates = []
            for font_size in candidate_font_sizes:
                author_chars = font_sizes[font_size]
                
                # Sort by position (top to bottom, left to right)
                author_chars.sort(key=lambda c: (-c['y0'], c['x0']))
                
                # Extract text from these characters
                author_lines = []
                current_line = []
                current_y = None
                
                for char in author_chars:
                    char_y = char['y0']
                    
                    # Group characters on same line (within 5 pixels)
                    if current_y is None or abs(char_y - current_y) < 5:
                        current_line.append(char)
                        current_y = char_y
                    else:
                        # New line
                        if current_line:
                            line_text = ''.join(c['text'] for c in sorted(current_line, key=lambda c: c['x0']))
                            if line_text.strip():
                                author_lines.append(line_text.strip())
                        current_line = [char]
                        current_y = char_y
                
                # Add last line
                if current_line:
                    line_text = ''.join(c['text'] for c in sorted(current_line, key=lambda c: c['x0']))
                    if line_text.strip():
                        author_lines.append(line_text.strip())
                
                # Combine lines to form author name(s)
                if author_lines:
                    # Take first 3 lines max (authors rarely span more)
                    author_candidate = ' '.join(author_lines[:3])
                    
                    # Validate it's an author name
                    if _is_author_name(author_candidate):
                        author_candidates.append(author_candidate)
            
            # Select best candidate
            if author_candidates:
                # Prefer shorter candidates (single author) or formatted multi-author
                best_candidate = min(author_candidates, key=lambda x: len(x.split(',')))
                return _extract_multiple_authors(best_candidate)
            
            return None
    except Exception as e:
        logger.warning(f"Font-based author extraction failed: {e}")
        return None


def _extract_author_by_position(fileobj: BinaryIO, title_line_idx: Optional[int] = None) -> Optional[str]:
    """Extract author using position-based heuristics (after title, before abstract)."""
    try:
        fileobj.seek(0)
        reader = PdfReader(fileobj)
        if len(reader.pages) == 0:
            return None
        
        first_page = reader.pages[0]
        first_page_text = first_page.extract_text() or ""
        lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
        
        if not lines:
            return None
        
        # Find title position if not provided
        if title_line_idx is None:
            # Look for title (usually first few lines, all caps or title case)
            for i, line in enumerate(lines[:10]):
                if _is_title_case(line) or _is_all_caps(line):
                    if 15 <= len(line) <= 250:
                        title_line_idx = i
                        break
        
        # Look for authors after title (2-8 lines after title)
        start_idx = (title_line_idx + 2) if title_line_idx is not None else 0
        end_idx = min(start_idx + 8, len(lines))
        
        candidates = []
        for i in range(start_idx, end_idx):
            line = lines[i].strip()
            
            # Skip if it's a non-author pattern
            if _is_non_author_pattern(line):
                continue
            
            # Skip if it looks like abstract/keywords
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in ["abstract", "keywords", "introduction", "1.", "2."]):
                break
            
            # Check if line contains author patterns
            has_author_pattern = False
            author_text = line
            
            # Check for explicit author patterns
            for pattern in ["Author:", "author:", "Authors:", "authors:", "By:", "by:", "Written by:", "written by:"]:
                if pattern in author_text:
                    author_text = author_text.split(pattern, 1)[1].strip()
                    has_author_pattern = True
                    break
            
            # Validate if it looks like an author name
            if has_author_pattern or _is_author_name(author_text):
                # Clean up the text
                author_text = re.sub(r'\([^)]*\)', '', author_text)  # Remove parentheticals
                author_text = re.sub(r'\[[^\]]*\]', '', author_text)  # Remove brackets
                author_text = author_text.strip()
                
                if _is_author_name(author_text):
                    # Score based on position (earlier = better)
                    score = 100 - ((i - start_idx) * 10)
                    if has_author_pattern:
                        score += 20  # Bonus for explicit pattern
                    candidates.append((author_text, score, i))
        
        if candidates:
            # Sort by score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Extract multiple authors from best candidate
            best_candidate = candidates[0][0]
            return _extract_multiple_authors(best_candidate)
        
        return None
    except Exception as e:
        logger.warning(f"Position-based author extraction failed: {e}")
        return None


def _validate_metadata_author(author: str) -> bool:
    """Validate if metadata author looks like a real author name."""
    if not author or not author.strip():
        return False
    
    author = author.strip()
    
    # Check length
    if not (5 <= len(author) <= 150):
        return False
    
    # Check if it's a software name
    software_names = [
        "adobe", "microsoft", "latex", "pdf", "word", "excel", "powerpoint",
        "acrobat", "reader", "creator", "author", "unknown", "n/a", "na",
        "libreoffice", "openoffice", "google", "apple", "pages", "keynote"
    ]
    
    author_lower = author.lower()
    if any(software in author_lower for software in software_names):
        return False
    
    # Check if it's too generic
    generic_names = ["author", "unknown", "n/a", "na", "anonymous", "user"]
    if author_lower in generic_names:
        return False
    
    # Check if it looks like a name
    if not _is_author_name(author):
        return False
    
    return True


def _extract_author_improved(fileobj: BinaryIO, title_position: Optional[float] = None, title_line_idx: Optional[int] = None) -> Optional[str]:
    """Extract author using improved multi-strategy approach."""
    try:
        fileobj.seek(0)
        
        # Strategy 1: Try metadata (validated, but often wrong)
        metadata_author = None
        try:
            reader = PdfReader(fileobj)
            metadata = reader.metadata
            if metadata:
                author = (
                    metadata.get("/Author") or 
                    metadata.get("Author") or
                    metadata.get("/Creator") or
                    metadata.get("Creator")
                )
                if author and author.startswith("/"):
                    author = author[1:]
                if author and author.strip():
                    author = author.strip()
                    # Validate metadata author quality
                    if _validate_metadata_author(author):
                        metadata_author = author
        except Exception as e:
            logger.debug(f"Metadata author extraction failed: {e}")
        
        # Strategy 2: Font-based extraction (most reliable, requires pdfplumber)
        font_author = _extract_author_by_font(fileobj, title_position)
        
        # Strategy 3: Position-based extraction (fallback)
        fileobj.seek(0)
        position_author = _extract_author_by_position(fileobj, title_line_idx)
        
        # Score and select best author
        candidates = []
        
        # Font-based gets highest priority (most reliable)
        if font_author:
            score = 40  # Base score
            # Bonus if it looks like a proper name
            if _is_author_name(font_author):
                score += 20
            candidates.append((font_author, score, "font"))
        
        # Position-based gets medium priority
        if position_author:
            score = 30  # Base score
            if _is_author_name(position_author):
                score += 15
            candidates.append((position_author, score, "position"))
        
        # Metadata gets lower priority (often wrong)
        if metadata_author:
            score = 20  # Base score
            if _is_author_name(metadata_author):
                score += 10
            candidates.append((metadata_author, score, "metadata"))
        
        # Select best candidate
        author = None
        if candidates:
            # Sort by score
            candidates.sort(key=lambda x: x[1], reverse=True)
            author = candidates[0][0]
            logger.info(f"Selected author using {candidates[0][2]} method: {author[:50]}...")
        
        return author
    except Exception as e:
        logger.error(f"Author extraction failed: {e}")
        return None


def extract_metadata(fileobj: BinaryIO) -> Dict[str, Optional[str]]:
    """
    Extract PDF metadata (title, author) using improved multi-strategy approach.
    
    Strategies:
    1. Font-based extraction (pdfplumber) - analyzes font size/weight
    2. Position-based extraction (pypdf) - analyzes position and patterns
    3. Metadata extraction - uses PDF metadata fields
    4. Pattern validation - filters out non-title content
    
    Returns:
        Dict with 'title' and 'author' keys. Values are None if not found.
    """
    try:
        fileobj.seek(0)
        
        # Strategy 1: Try metadata (quick check, but often wrong)
        metadata_title = None
        try:
            reader = PdfReader(fileobj)
            metadata = reader.metadata
            if metadata:
                metadata_title = (
                    metadata.get("/Title") or 
                    metadata.get("Title") or 
                    metadata.get("/Subject") or
                    metadata.get("Subject")
                )
                if metadata_title and metadata_title.startswith("/"):
                    metadata_title = metadata_title[1:]
                if metadata_title:
                    metadata_title = metadata_title.strip()
                    # Validate metadata title quality
                    if _is_non_title_pattern(metadata_title) or len(metadata_title) < 10:
                        metadata_title = None
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
        
        # Strategy 2: Font-based extraction (most reliable, requires pdfplumber)
        font_title = _extract_by_font_analysis(fileobj)
        
        # Strategy 3: Position-based extraction (fallback)
        fileobj.seek(0)
        position_title = _extract_by_position_pypdf(fileobj)
        
        # Score and select best title
        candidates = []
        
        # Font-based gets highest priority (most reliable)
        if font_title:
            score = 40  # Base score
            if _is_title_case(font_title) or _is_all_caps(font_title):
                score += 20
            candidates.append((font_title, score, "font"))
        
        # Position-based gets medium priority
        if position_title:
            score = 30  # Base score
            if _is_title_case(position_title) or _is_all_caps(position_title):
                score += 15
            candidates.append((position_title, score, "position"))
        
        # Metadata gets lower priority (often wrong)
        if metadata_title:
            score = 20  # Base score
            if _is_title_case(metadata_title) or _is_all_caps(metadata_title):
                score += 10
            candidates.append((metadata_title, score, "metadata"))
        
        # Select best candidate
        title = None
        if candidates:
            # Sort by score
            candidates.sort(key=lambda x: x[1], reverse=True)
            title = candidates[0][0]
            logger.info(f"Selected title using {candidates[0][2]} method: {title[:50]}...")
        
        # Extract author (pass title position info if available)
        fileobj.seek(0)
        # Get title position for better author extraction
        title_position = None
        title_line_idx = None
        
        if title:
            # Try to get title position from font analysis
            try:
                if PDFPLUMBER_AVAILABLE:
                    fileobj.seek(0)
                    with pdfplumber.open(fileobj) as pdf:
                        if len(pdf.pages) > 0:
                            first_page = pdf.pages[0]
                            chars = first_page.chars
                            if chars:
                                # Find title position
                                for char in chars:
                                    if char.get('y0', 0) > first_page.height * 0.7:
                                        # Check if this character is part of title
                                        # (simplified - just use first large font)
                                        font_size = char.get('size', 0)
                                        if font_size > 0:
                                            title_position = char.get('y0')
                                            break
            except Exception:
                pass
            
            # Get title line index from position-based extraction
            try:
                fileobj.seek(0)
                reader = PdfReader(fileobj)
                if len(reader.pages) > 0:
                    first_page = reader.pages[0]
                    first_page_text = first_page.extract_text() or ""
                    lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
                    
                    # Find title line
                    for i, line in enumerate(lines[:10]):
                        if title.lower() in line.lower() or line.lower() in title.lower():
                            title_line_idx = i
                            break
            except Exception:
                pass
        
        author = _extract_author_improved(fileobj, title_position, title_line_idx)
        
        return {
            "title": title,
            "author": author
        }
    except Exception as e:
        logger.error(f"Title extraction failed: {e}")
        return {
            "title": None,
            "author": None
        }


