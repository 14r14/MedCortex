"""Table extraction from PDFs for TableRAG using camelot-py."""

import io
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Import pandas separately from camelot to ensure pandas is available even if camelot fails
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import camelot

    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    camelot = None

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)


def extract_tables_camelot(
    file_stream: io.BytesIO, doc_id: str
) -> list["pd.DataFrame"]:
    """
    Extract tables from PDF using camelot-py.

    Returns a list of pandas DataFrames.

    Args:
        file_stream: PDF file stream (BytesIO)
        doc_id: Document ID for tracking

    Returns:
        List of pandas DataFrames (one per table)
    """
    if not CAMELOT_AVAILABLE:
        logger.warning("camelot-py not available, skipping table extraction")
        return []

    dataframes = []

    try:
        # Save BytesIO to temporary file path for camelot
        # camelot requires a file path, not a stream
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            file_stream.seek(0)
            tmp_file.write(file_stream.read())
            tmp_path = tmp_file.name

        try:
            # Extract tables using camelot
            tables = camelot.read_pdf(tmp_path, pages="all", flavor="lattice")

            for table_idx, table in enumerate(tables):
                if table.df is not None and len(table.df) > 0:
                    # Clean DataFrame (remove empty rows/cols)
                    df = table.df.copy()
                    df = df.dropna(how="all").dropna(axis=1, how="all")

                    # Reset index
                    df = df.reset_index(drop=True)

                    if len(df) > 0:
                        dataframes.append(df)
                        logger.info(
                            f"Extracted table {table_idx + 1} with shape {df.shape} from doc {doc_id}"
                        )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"Error extracting tables from PDF using camelot: {e}")
        return []

    logger.info(f"Extracted {len(dataframes)} tables from document {doc_id}")
    return dataframes


def store_tables_in_session(
    dataframes: list["pd.DataFrame"], doc_id: str, session_key: str = "table_store"
) -> None:
    """
    Store extracted DataFrames in Streamlit session state.

    Args:
        dataframes: List of pandas DataFrames
        doc_id: Document ID
        session_key: Session state key (default: "table_store")
    """
    if st is None:
        return

    if not CAMELOT_AVAILABLE:
        return

    if session_key not in st.session_state:
        st.session_state[session_key] = {}

    # Store DataFrames mapped by doc_id
    st.session_state[session_key][doc_id] = dataframes
    logger.info(f"Stored {len(dataframes)} tables in session for doc {doc_id}")


def get_tables_for_docs(
    doc_ids: list[str], session_key: str = "table_store"
) -> list[dict]:
    """Get all tables (as DataFrames) for given document IDs."""
    if st is None or not CAMELOT_AVAILABLE:
        return []

    if session_key not in st.session_state:
        return []

    table_store = st.session_state[session_key]
    tables = []

    for doc_id in doc_ids:
        if doc_id in table_store:
            dataframes = table_store[doc_id]
            for idx, df in enumerate(dataframes):
                tables.append({"doc_id": doc_id, "table_index": idx, "dataframe": df})

    return tables
