"""Table reasoning for TableRAG - uses LLM to query structured table data."""
import json
import logging
from typing import List, Dict, Any, Optional

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from app.config import Settings
from app.rag.generator import GeneratorClient

logger = logging.getLogger(__name__)


class TableReasoner:
    """Reason over structured table data using LLM to generate queries and execute them."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gen = GeneratorClient(settings)
    
    def is_table_query(self, query: str, tables: List[Dict[str, Any]]) -> bool:
        """
        Determine if a query is likely about tabular data.
        
        Checks for keywords like "rate", "percentage", "statistics", "data", 
        "table", "results", "outcomes", etc.
        """
        table_keywords = [
            "rate", "percentage", "percent", "statistic", "statistics", "data",
            "table", "tables", "results", "outcomes", "comparison", "compare",
            "average", "mean", "median", "sum", "total", "count", "number of",
            "demographic", "clinical", "trial", "survival", "efficacy", "efficiency"
        ]
        
        query_lower = query.lower()
        has_keywords = any(keyword in query_lower for keyword in table_keywords)
        
        # Also check if query mentions specific numerical operations
        numerical_ops = ["how many", "what is the", "calculate", "compute", "what was"]
        has_numerical = any(query_lower.startswith(op) or op in query_lower for op in numerical_ops)
        
        return has_keywords or has_numerical or len(tables) > 0
    
    def reason_over_tables(self, query: str, tables: List[Dict[str, Any]], max_tables: int = 3) -> Optional[str]:
        """
        Use LLM to reason over tables and extract relevant information.
        
        Args:
            query: User query
            tables: List of table dictionaries
            max_tables: Maximum number of tables to include in reasoning
            
        Returns:
            Extracted information from tables, or None if no relevant tables
        """
        if not tables:
            return None
        
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available for table reasoning")
            return None
        
        # Limit number of tables to avoid token limits
        relevant_tables = tables[:max_tables]
        
        # Format tables for LLM
        table_contexts = []
        for table in relevant_tables:
            table_text = f"""
Table {table.get('table_id', 'unknown')} (Page {table.get('page_num', '?')}):
Columns: {', '.join(table.get('columns', []))}
Rows: {table.get('row_count', 0)}
Data (first 10 rows):
{table.get('text_repr', '')[:2000]}  # Limit text length
"""
            table_contexts.append(table_text)
        
        # Build prompt for LLM to query tables
        prompt = f"""You are analyzing structured data from research papers. Answer the question based on the tables provided.

Question: {query}

Tables:
{chr(10).join(table_contexts)}

Instructions:
1. Identify which table(s) contain relevant information
2. Extract the specific data points or statistics needed to answer the question
3. If the answer requires calculation, show your reasoning
4. Provide a clear, accurate answer based on the table data
5. Do NOT include placeholder citations like [Source 1], [Table Data], etc. in your answer
6. Just provide the answer text itself

Answer:"""
        
        try:
            # Use generate_from_prompt for custom table reasoning prompt
            answer = self.gen.generate_from_prompt(
                prompt=prompt,
                temperature=0.0  # Low temperature for factual extraction
            )
            return answer
        except Exception as e:
            logger.error(f"Error reasoning over tables: {e}")
            return None
    
    def execute_table_query(self, query: str, table: Dict[str, Any]) -> Optional[str]:
        """
        Execute a specific query on a single table using pandas.
        
        This is a simpler approach for well-structured queries that can be
        directly executed on the DataFrame.
        """
        if not PANDAS_AVAILABLE:
            return None
        
        try:
            # Reconstruct DataFrame from stored data
            data = table.get("data", [])
            if not data:
                return None
            
            df = pd.DataFrame(data)
            
            # Use LLM to generate pandas code or extract answer directly
            # For now, use direct text search and extraction
            query_lower = query.lower()
            
            # Simple keyword matching for common queries
            if "how many" in query_lower or "count" in query_lower:
                # Count rows matching some criteria
                # This is simplified - in production, use LLM to generate pandas code
                return f"The table contains {len(df)} rows."
            
            # Try to extract relevant rows based on query
            # In production, use LLM to generate filtering logic
            return f"Found relevant data in table: {table.get('table_id', 'unknown')}"
            
        except Exception as e:
            logger.error(f"Error executing table query: {e}")
            return None

