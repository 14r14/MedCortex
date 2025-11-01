"""Iterative agentic framework (i-MedRAG) for multi-hop reasoning."""
import logging
from typing import List, Dict, Optional, Tuple

from app.config import Settings
from app.rag.generator import GeneratorClient
from app.rag.faiss_store import FaissStore
from app.rag.bm25_store import BM25Store
from app.rag.embeddings import EmbeddingClient
from app.rag.reranker import Reranker
from app.rag.table_reasoner import TableReasoner

logger = logging.getLogger(__name__)


class IterativeAgent:
    """
    Iterative agentic framework for complex multi-hop queries.
    
    Based on i-MedRAG approach:
    1. Deconstruct complex query into sub-questions
    2. Perform retrieval pass for first sub-question
    3. Analyze results and generate follow-up query
    4. Perform second retrieval pass
    5. Synthesize all evidence into complete answer
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embed = EmbeddingClient(settings)
        self.vs = FaissStore(settings, session_key="faiss_store")
        self.bm25 = BM25Store(session_key="faiss_store")
        self.reranker = Reranker(settings)
        self.gen = GeneratorClient(settings)
        self.table_reasoner = TableReasoner(settings)
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into sub-questions using LLM.
        
        Args:
            query: Original complex query
            
        Returns:
            List of sub-questions
        """
        prompt = f"""Analyze this research question and break it down into logical sub-questions that can be answered sequentially.

Original question: {query}

Instructions:
1. Identify if this question requires multiple pieces of information
2. If yes, list the sub-questions in logical order
3. If no, return the original question as a single item

Output format (one sub-question per line):
- Sub-question 1
- Sub-question 2
- ...

Sub-questions:"""
        
        try:
            # Use generate_from_prompt for custom prompt formatting
            response = self.gen.generate_from_prompt(
                prompt=prompt,
                temperature=0.3  # Slightly creative for decomposition
            )
            
            # Parse response into sub-questions
            sub_questions = []
            for line in response.split('\n'):
                line = line.strip()
                # Remove bullet points and dashes
                line = line.lstrip('- ').lstrip('• ').lstrip('* ')
                if line and len(line) > 10:  # Filter out very short lines
                    sub_questions.append(line)
            
            # If decomposition failed, use original query
            if not sub_questions:
                sub_questions = [query]
                
            return sub_questions[:5]  # Limit to 5 sub-questions max
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, using original query")
            return [query]
    
    def retrieve_for_query(self, sub_query: str, allowed_doc_ids: Optional[List[str]] = None) -> List[Dict]:
        """Retrieve relevant chunks for a sub-query using hybrid search."""
        # Hybrid search
        q_emb = self.embed.embed_query(sub_query)
        semantic_hits = self.vs.search(q_emb, top_k=15, allowed_doc_ids=allowed_doc_ids)
        keyword_hits = self.bm25.search(sub_query, top_k=15, allowed_doc_ids=allowed_doc_ids)
        
        # Combine using RRF
        fused_hits = self._reciprocal_rank_fusion(semantic_hits, keyword_hits, k=30)
        
        # Re-rank
        try:
            reranked_hits = self.reranker.rerank(sub_query, fused_hits[:20], top_k=6)
            if not reranked_hits:
                reranked_hits = fused_hits[:6]
        except Exception:
            reranked_hits = fused_hits[:6]
        
        return reranked_hits
    
    def generate_followup_query(self, original_query: str, sub_queries: List[str], 
                                previous_results: List[Dict]) -> Optional[str]:
        """
        Generate a follow-up query based on previous results and remaining sub-questions.
        
        Args:
            original_query: Original complex query
            sub_queries: List of sub-questions (some may already be answered)
            previous_results: Results from previous retrieval passes
            
        Returns:
            Follow-up query, or None if all sub-questions are answered
        """
        if not previous_results:
            return None
        
        # Extract context from previous results
        context_texts = [r.get("text", "") for r in previous_results[:3]]
        
        prompt = f"""Based on the original question and information found so far, generate a focused follow-up question to find the remaining information needed.

Original question: {original_query}

Sub-questions identified:
{chr(10).join([f"- {sq}" for sq in sub_queries])}

Information found so far:
{chr(10).join([f"- {ctx[:200]}" for ctx in context_texts])}

Generate a specific, focused follow-up question that will help complete the answer. If all information is found, respond with "DONE".

Follow-up question:"""
        
        try:
            # Use generate_from_prompt for custom prompt formatting
            followup = self.gen.generate_from_prompt(
                prompt=prompt,
                temperature=0.2
            )
            
            if "DONE" in followup.upper() or len(followup.strip()) < 10:
                return None
            
            return followup.strip()
            
        except Exception as e:
            logger.warning(f"Follow-up query generation failed: {e}")
            return None
    
    def synthesize_answer(self, original_query: str, all_results: List[List[Dict]], 
                          table_answer: Optional[str] = None) -> str:
        """
        Synthesize all evidence from multiple retrieval passes into a complete answer.
        
        Args:
            original_query: Original complex query
            all_results: List of result lists from each retrieval pass
            table_answer: Optional answer from table reasoning
            
        Returns:
            Synthesized complete answer
        """
        # Flatten all results
        all_contexts = []
        seen_texts = set()
        for result_list in all_results:
            for result in result_list:
                text = result.get("text", "")
                if text and text not in seen_texts:
                    all_contexts.append(text)
                    seen_texts.add(text)
        
        # Limit total contexts
        all_contexts = all_contexts[:15]
        
        # Build synthesis prompt
        synthesis_prompt = f"""You are a research assistant analyzing multiple sources of information to answer a complex question.

Original question: {original_query}

Information gathered from multiple sources:
{chr(10).join([f"[Source {i+1}] {ctx[:300]}" for i, ctx in enumerate(all_contexts)])}
{f"[Table Data] {table_answer}" if table_answer else ""}

Instructions:
1. Integrate information from all sources coherently
2. Address all aspects of the original question
3. If there are contradictions, note them
4. Provide a comprehensive, well-structured answer
5. Do NOT include placeholder citations like [Source 1], [Source 2], [Table Data], etc. in your answer
6. Sources will be listed separately - just provide the answer text itself

Comprehensive answer:"""
        
        try:
            # Use generate_from_prompt for custom synthesis prompt
            answer = self.gen.generate_from_prompt(
                prompt=synthesis_prompt,
                temperature=self.settings.temperature
            )
            return answer
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return "I apologize, but I encountered an error synthesizing the answer from multiple sources."
    
    def answer_iteratively(self, query: str, allowed_doc_ids: Optional[List[str]] = None, 
                          max_iterations: int = 3) -> Tuple[str, List[str]]:
        """
        Main iterative agentic reasoning method.
        
        Args:
            query: Original complex query
            allowed_doc_ids: Optional list of allowed document IDs for filtering
            max_iterations: Maximum number of retrieval iterations
            
        Returns:
            Tuple of (final_answer, source_uris)
        """
        # Step 1: Decompose query into sub-questions
        sub_queries = self.decompose_query(query)
        logger.info(f"Decomposed query into {len(sub_queries)} sub-questions")
        
        # Step 2: Check if query is about tables
        from app.rag.table_extractor import get_tables_for_docs
        tables = get_tables_for_docs(allowed_doc_ids or [])
        table_answer = None
        
        if tables and self.table_reasoner.is_table_query(query, tables):
            logger.info("Query appears to be about tabular data, performing table reasoning")
            table_answer = self.table_reasoner.reason_over_tables(query, tables)
        
        # Step 3: Iterative retrieval
        all_results = []
        current_query = query
        iteration = 0
        
        while iteration < max_iterations and current_query:
            logger.info(f"Iteration {iteration + 1}: Retrieving for '{current_query}'")
            
            # Retrieve for current query
            results = self.retrieve_for_query(current_query, allowed_doc_ids)
            if results:
                all_results.append(results)
            
            # Generate follow-up query for next iteration
            if iteration < max_iterations - 1:
                current_query = self.generate_followup_query(query, sub_queries, results)
                if not current_query:
                    break
            
            iteration += 1
        
        # Step 4: Synthesize all evidence
        final_answer = self.synthesize_answer(query, all_results, table_answer)
        
        # Extract unique sources
        all_sources = []
        for result_list in all_results:
            for result in result_list:
                source = result.get("source_uri", "")
                if source and source not in all_sources:
                    all_sources.append(source)
        
        return final_answer, all_sources
    
    def _reciprocal_rank_fusion(self, semantic_hits: List[Dict], keyword_hits: List[Dict], 
                                k: int = 60) -> List[Dict]:
        """Combine semantic and keyword search results using RRF."""
        hit_map = {}
        rrf_scores = {}
        
        for rank, hit in enumerate(semantic_hits, start=1):
            hit_id = hit.get("id") or hit.get("text", "")[:100]
            hit_map[hit_id] = hit
            rrf_scores[hit_id] = rrf_scores.get(hit_id, 0.0) + (1.0 / (k + rank))
        
        for rank, hit in enumerate(keyword_hits, start=1):
            hit_id = hit.get("id") or hit.get("text", "")[:100]
            if hit_id not in hit_map:
                hit_map[hit_id] = hit
            rrf_scores[hit_id] = rrf_scores.get(hit_id, 0.0) + (1.0 / (k + rank))
        
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [hit_map[hit_id] for hit_id in sorted_ids[:20]]

