"""Table extraction from PDFs for TableRAG."""
import io
import json
import logging
from typing import List, Dict, Any, Optional

try:
    import pdfplumber
    import pandas as pd
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None
    pd = None

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)


def extract_tables_from_pdf(file_stream: io.BytesIO, doc_id: str) -> List[Dict[str, Any]]:
    """
    Extract tables from PDF using pdfplumber.
    
    Args:
        file_stream: PDF file stream (BytesIO)
        doc_id: Document ID for tracking
        
    Returns:
        List of table dictionaries with structured data
    """
    if not PDFPLUMBER_AVAILABLE:
        logger.warning("pdfplumber not available, skipping table extraction")
        return []
    
    tables = []
    
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract tables from current page
                page_tables = page.extract_tables()
                
                for table_idx, table in enumerate(page_tables):
                    if not table or len(table) == 0:
                        continue
                    
                    # Convert table to DataFrame for structured access
                    try:
                        # Use first row as header if it looks like headers
                        df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                        
                        # Clean DataFrame (remove empty rows/cols)
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        # Convert to dict for JSON serialization
                        table_dict = {
                            "doc_id": doc_id,
                            "page_num": page_num,
                            "table_index": table_idx,
                            "table_id": f"{doc_id}_page{page_num}_table{table_idx}",
                            "data": df.to_dict(orient='records'),  # List of dicts (rows)
                            "columns": list(df.columns) if len(df.columns) > 0 else [f"Col{i+1}" for i in range(len(table[0]) if table else 0)],
                            "row_count": len(df),
                            "col_count": len(df.columns) if len(df.columns) > 0 else 0,
                            "text_repr": df.to_string(),  # String representation for search
                        }
                        
                        tables.append(table_dict)
                        
                    except Exception as e:
                        logger.warning(f"Error processing table on page {page_num}, table {table_idx}: {e}")
                        continue
                        
    except Exception as e:
        logger.error(f"Error extracting tables from PDF: {e}")
        return []
    
    logger.info(f"Extracted {len(tables)} tables from document {doc_id}")
    return tables


def store_tables_in_session(tables: List[Dict[str, Any]], session_key: str = "tables") -> None:
    """Store extracted tables in Streamlit session state."""
    if st is None:
        return
    
    if session_key not in st.session_state:
        st.session_state[session_key] = []
    
    st.session_state[session_key].extend(tables)


def get_tables_for_docs(doc_ids: List[str], session_key: str = "tables") -> List[Dict[str, Any]]:
    """Get all tables for given document IDs."""
    if st is None:
        return []
    
    if session_key not in st.session_state:
        return []
    
    all_tables = st.session_state[session_key]
    return [t for t in all_tables if t.get("doc_id") in doc_ids]


def get_table_by_id(table_id: str, session_key: str = "tables") -> Optional[Dict[str, Any]]:
    """Get a specific table by its ID."""
    if st is None:
        return None
    
    if session_key not in st.session_state:
        return None
    
    all_tables = st.session_state[session_key]
    for table in all_tables:
        if table.get("table_id") == table_id:
            return table
    
    return None

