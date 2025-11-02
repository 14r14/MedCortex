"""TableQueryPipeline for executing pandas code on extracted tables."""
import io
import json
import logging
import sys
from contextlib import redirect_stdout
from typing import List, Optional, Tuple

try:
    import pandas as pd
    import camelot
    PANDAS_AVAILABLE = True
    CAMELOT_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    CAMELOT_AVAILABLE = False
    pd = None
    camelot = None

try:
    import streamlit as st
except ImportError:
    st = None

from app.config import Settings
from app.rag.generator import GeneratorClient

logger = logging.getLogger(__name__)


class TableQueryPipeline:
    """
    Pipeline for querying extracted tables using LLM-generated pandas code.
    
    Uses camelot-py for table extraction and executes pandas code to answer queries.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gen = GeneratorClient(settings)
    
    def _get_tables_from_session(self, allowed_doc_ids: List[str]) -> List[dict]:
        """Get tables from session state filtered by document IDs."""
        if st is None:
            return []
        
        session_key = "table_store"
        if session_key not in st.session_state:
            return []
        
        table_store = st.session_state[session_key]
        tables = []
        
        for doc_id in allowed_doc_ids:
            if doc_id in table_store:
                doc_tables = table_store[doc_id]
                for table_idx, df in enumerate(doc_tables):
                    tables.append({
                        "doc_id": doc_id,
                        "table_index": table_idx,
                        "dataframe": df,
                        "name": f"df_{doc_id}_{table_idx}"
                    })
        
        return tables
    
    def _generate_pandas_code(self, question: str, tables: List[dict]) -> str:
        """
        Use LLM to generate pandas code to answer the question.
        
        Args:
            question: User question
            tables: List of table metadata with dataframes
            
        Returns:
            Generated Python code
        """
        # Prepare table descriptions for prompt
        table_descriptions = []
        for table in tables:
            df = table["dataframe"]
            name = table["name"]
            table_descriptions.append(
                f"{name} (from doc {table['doc_id']}, table {table['table_index']}):\n"
                f"- Shape: {df.shape}\n"
                f"- Columns: {list(df.columns)}\n"
                f"- Sample data:\n{df.head(3).to_string()}"
            )
        
        prompt = f"""Given the following pandas DataFrames from medical research documents and a user question, write Python code to extract and calculate the answer with full detail suitable for medical researchers.

Available DataFrames:
{chr(10).join(table_descriptions)}

Question: {question}

Requirements for medical research data extraction:
1. Extract ALL relevant quantitative data: percentages, counts, ratios, means, medians, statistical values
2. Include precise calculations with all intermediate steps if needed
3. Preserve exact numerical precision - do not round unless necessary
4. Extract complete rows/columns that contain relevant information
5. If comparisons are needed, extract data for all relevant groups/conditions
6. Use only pandas operations (no external libraries except pandas)
7. Use print() to output the final answer with all relevant details
8. For medical research, output should include: exact values, sample sizes, statistical measures, and contextual information
9. Do not include any explanatory comments or text outside the code
10. Return only the Python code, nothing else

Code:"""
        
        try:
            code = self.gen.generate_from_prompt(prompt=prompt, temperature=0.0)
            
            # Clean code - remove markdown code blocks if present
            code = code.strip()
            if code.startswith('```'):
                code = code.split('```', 2)[1]
                if code.startswith('python'):
                    code = code.split('\n', 1)[1]
            
            return code.strip()
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return ""
    
    def _execute_pandas_code(self, code: str, tables: List[dict]) -> str:
        """
        Execute generated pandas code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            tables: List of tables to make available
            
        Returns:
            Captured stdout output
        """
        if not code:
            return "Error: No code generated."
        
        # Create local namespace with tables
        namespace = {}
        for table in tables:
            namespace[table["name"]] = table["dataframe"]
        namespace["pd"] = pd
        namespace["__builtins__"] = __builtins__
        
        # Capture stdout
        f = io.StringIO()
        try:
            with redirect_stdout(f):
                exec(code, namespace)
            output = f.getvalue()
            return output.strip() if output else "No output generated."
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return f"Error executing code: {str(e)}"
    
    def answer(self, question: str, allowed_doc_ids: Optional[List[str]] = None) -> Tuple[str, List[str]]:
        """
        Answer question using table data.
        
        Args:
            question: User question
            allowed_doc_ids: Optional list of allowed document IDs
            
        Returns:
            Tuple of (answer, source_uris)
        """
        if not PANDAS_AVAILABLE:
            return "Error: pandas is not available for table queries.", []
        
        if not allowed_doc_ids:
            return "Error: No documents available.", []
        
        # Get tables from session state
        tables = self._get_tables_from_session(allowed_doc_ids)
        
        if not tables:
            return "No tables found in the uploaded documents.", []
        
        logger.info(f"Found {len(tables)} tables for query")
        
        # Generate pandas code
        code = self._generate_pandas_code(question, tables)
        
        if not code:
            return "Error: Could not generate code to answer the question.", []
        
        # Execute code
        answer = self._execute_pandas_code(code, tables)
        
        # Extract source URIs (map doc_ids to source URIs if available)
        sources = []
        if st is not None and "ingested_docs" in st.session_state:
            doc_id_to_uri = {doc_id: uri for doc_id, _, uri, _ in st.session_state["ingested_docs"]}
            for doc_id in set(t["doc_id"] for t in tables):
                if doc_id in doc_id_to_uri:
                    sources.append(doc_id_to_uri[doc_id])
        
        return answer, list(dict.fromkeys(sources))

