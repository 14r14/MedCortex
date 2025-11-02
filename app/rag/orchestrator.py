"""Orchestrator layer for agentic iterative framework."""
import json
import logging
from typing import List, Optional, Tuple

try:
    import streamlit as st
except ImportError:
    st = None

from app.config import Settings
from app.rag.generator import GeneratorClient

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator layer that manages multi-step reasoning using QueryPipeline.
    
    Based on i-MedRAG approach:
    1. Query decomposition into sub-questions
    2. Iterative retrieval for each sub-question
    3. Final synthesis of all evidence
    """
    
    def __init__(self, settings: Settings, query_pipeline):
        """
        Initialize orchestrator.
        
        Args:
            settings: Application settings
            query_pipeline: QueryPipeline instance to use for answering sub-questions
        """
        self.settings = settings
        self.query_pipeline = query_pipeline
        self.gen = GeneratorClient(settings)
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into sub-questions.
        
        Args:
            query: Original complex query
            
        Returns:
            List of sub-questions
        """
        prompt = """You are an expert medical research assistant. Deconstruct the following complex user query into a series of simple, sequential, and answerable sub-questions that will enable a comprehensive answer for medical researchers.

Each sub-question should focus on extracting specific, detailed information such as:
- Quantitative data (statistics, percentages, p-values, sample sizes)
- Methodologies (study designs, protocols, procedures)
- Clinical findings (outcomes, efficacy, safety data)
- Comparative information (treatments, approaches, populations)

Return only a numbered list of these sub-questions.

Query: {query}

Sub-questions:""".format(query=query)
        
        try:
            response = self.gen.generate_from_prompt(prompt=prompt, temperature=0.3)
            
            # Parse response into sub-questions
            sub_questions = []
            for line in response.split('\n'):
                line = line.strip()
                # Remove numbering (1., 2., etc.) and bullets
                line = line.lstrip('1234567890.').lstrip(')').lstrip(' ').lstrip('- ').lstrip('• ').lstrip('* ')
                # Remove common prefixes
                if line.lower().startswith('sub-question'):
                    line = line.split(':', 1)[-1].strip()
                if line and len(line) > 10:  # Filter out very short lines
                    sub_questions.append(line)
            
            # If decomposition failed, use original query
            if not sub_questions:
                sub_questions = [query]
            
            return sub_questions[:5]  # Limit to 5 sub-questions max
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, using original query")
            return [query]
    
    def route_query(self, query: str) -> List[dict]:
        """
        Route query: decompose and classify each sub-question as TEXT or TABLE.
        
        Args:
            query: Original query
            
        Returns:
            List of dicts with 'question' and 'type' ('TEXT' or 'TABLE')
        """
        prompt = """You are a research query router. Deconstruct the user query into sub-questions. For each sub-question, classify it as either TEXT (for conceptual, procedural, or discussion-based info) or TABLE (for quantitative data, statistics, p-values, or comparisons).

Query: {query}

Return a JSON list of objects, like this: [{{"question": "sub-question 1", "type": "TEXT"}}, {{"question": "sub-question 2", "type": "TABLE"}}]

Only return the JSON list, nothing else:""".format(query=query)
        
        try:
            response = self.gen.generate_from_prompt(prompt=prompt, temperature=0.2)
            
            # Try to extract JSON from response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if '```' in response:
                # Find code block
                parts = response.split('```')
                for i, part in enumerate(parts):
                    if part.strip().startswith('json') or (i > 0 and parts[i-1].strip().startswith('json')):
                        # Take the part after json marker
                        if part.strip().startswith('json'):
                            json_part = part.split('json', 1)[1].strip()
                        else:
                            json_part = part.strip()
                        response = json_part
                        break
                    elif '[' in part and ']' in part:
                        # Found JSON-like content
                        response = part.strip()
                        break
            
            # Try to find JSON array - look for the first complete JSON array
            start_idx = response.find('[')
            if start_idx >= 0:
                # Find matching closing bracket
                bracket_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(response)):
                    if response[i] == '[':
                        bracket_count += 1
                    elif response[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i
                            break
                
                if end_idx > start_idx:
                    json_str = response[start_idx:end_idx+1]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # Validate structure
                        valid = []
                        for item in parsed:
                            if isinstance(item, dict) and "question" in item and "type" in item:
                                valid.append({
                                    "question": str(item["question"]),
                                    "type": str(item["type"]).upper()
                                })
                        if valid:
                            return valid
                        elif len(parsed) > 0:
                            # Try to convert if structure is close
                            return parsed
            
            # Fallback: try to parse as single JSON object or other formats
            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except:
                pass
            
            # Fallback: treat all as TEXT questions
            sub_questions = self.decompose_query(query)
            return [{"question": sq, "type": "TEXT"} for sq in sub_questions]
            
        except json.JSONDecodeError as e:
            logger.warning(f"Query routing JSON parse failed: {e}, using TEXT classification")
            # Fallback to simple decomposition
            sub_questions = self.decompose_query(query)
            return [{"question": sq, "type": "TEXT"} for sq in sub_questions]
        except Exception as e:
            logger.warning(f"Query routing failed: {e}, using TEXT classification")
            # Fallback to simple decomposition
            sub_questions = self.decompose_query(query)
            return [{"question": sq, "type": "TEXT"} for sq in sub_questions]
    
    def answer_iteratively(self, query: str, allowed_doc_ids: Optional[List[str]] = None) -> Tuple[str, List[str]]:
        """
        Answer query using iterative decomposition and retrieval.
        
        Args:
            query: Original query
            allowed_doc_ids: Optional list of allowed document IDs
            
        Returns:
            Tuple of (final_answer, source_uris)
        """
        # Step 1: Decompose query into sub-questions (with routing)
        routed_questions = self.route_query(query)
        
        # Step 2: Iterative retrieval - loop through sub-questions
        intermediate_answers = []
        all_sources = []
        all_source_chunks = []  # Collect chunks for verification
        
        for i, routed_q in enumerate(routed_questions, 1):
            sub_question = routed_q.get("question", "")
            q_type = routed_q.get("type", "TEXT")
            
            logger.info(f"Step {i}/{len(routed_questions)}: Answering {q_type} question: {sub_question}")
            
            if q_type == "TABLE":
                # Handle table query
                from app.rag.table_query import TableQueryPipeline
                table_pipeline = TableQueryPipeline(self.settings)
                answer, sources = table_pipeline.answer(sub_question, allowed_doc_ids=allowed_doc_ids)
                # For table queries, use answer as chunk (tables are structured data)
                chunks = [answer] if answer else []
            else:
                # Handle text query using QueryPipeline
                # Disable orchestrator recursion for sub-questions (use direct RAG)
                answer, sources = self.query_pipeline.answer(
                    sub_question, 
                    allowed_doc_ids=allowed_doc_ids,
                    use_orchestrator=False  # Prevent recursion
                )
                # Get source chunks from retrieval (need to access them from query pipeline)
                # We'll collect chunks from the retrieval step
                chunks = self._get_source_chunks_for_query(sub_question, allowed_doc_ids)
            
            intermediate_answers.append({
                "step": i,
                "question": sub_question,
                "type": q_type,
                "answer": answer,
                "sources": sources,
                "chunks": chunks
            })
            all_sources.extend(sources)
            all_source_chunks.extend(chunks)
        
        # Step 3: Final synthesis
        # Format evidence without numbered labels to prevent citations
        evidence_text = "\n\n".join([
            f"Information from {a['type']} analysis:\nQuestion addressed: {a['question']}\nFindings: {a['answer']}"
            for a in intermediate_answers
        ])
        
        synthesis_prompt = """You are an expert medical research assistant. Using the following collected information from multiple analyses, synthesize a single, highly detailed, comprehensive answer tailored for medical researchers.

Original Query: {query}

Collected Information:
{evidence}

Instructions for synthesis:
- Provide a thorough, detailed answer that addresses all aspects of the original query
- Include ALL specific quantitative data: exact figures, percentages, p-values, confidence intervals, sample sizes, statistical significance
- Detail methodologies: study designs, protocols, procedures, inclusion/exclusion criteria
- Present clinical findings comprehensively: patient outcomes, efficacy rates, side effects, safety profiles
- Include comparative analyses where relevant
- Address limitations, contextual nuances, and generalizability concerns
- Use precise medical and scientific terminology
- Do not oversimplify - medical researchers need complete information with all details
- Integrate information from all sources coherently and naturally - do NOT reference the analysis parts
- Do NOT cite or reference "Evidence 1", "Evidence 2", "Part 1", "Part 2", "Information from TEXT analysis", etc.
- Do NOT include placeholder citations like [Source 1], [Source 2], [Table Data], (Evidence 1), (Evidence 2), etc.
- Do NOT include meta-commentary or descriptions of what the answer includes (e.g., "This answer integrates...", "This synthesized answer...", "The answer includes...")
- Do NOT summarize or describe your methodology - just provide the actual answer directly
- Present the information as a unified, coherent answer without referencing the intermediate steps

Synthesized Answer:""".format(
            query=query,
            evidence=evidence_text
        )
        
        # Collect all source chunks for verification
        all_source_chunks = []
        for answer_data in intermediate_answers:
            # Get chunks from the query pipeline for this sub-question
            # Note: We can't easily get chunks here, so we'll use the answers as proxy
            # In practice, we'd need to pass chunks through the orchestrator
            pass
        
        try:
            final_answer = self.gen.generate_from_prompt(
                prompt=synthesis_prompt,
                temperature=self.settings.temperature
            )
            
            # Verify the final synthesized answer against source chunks
            try:
                from app.rag.verifier import AnswerVerifier
                verifier = AnswerVerifier(self.settings)
                # Use all collected source chunks for verification
                verification_chunks = []
                for a in intermediate_answers:
                    verification_chunks.extend(a.get("chunks", []))
                
                if verification_chunks:
                    verification_results = verifier.verify_answer(final_answer, verification_chunks)
                    
                    if st is not None:
                        if "verification_results" not in st.session_state:
                            st.session_state["verification_results"] = []
                        st.session_state["verification_results"].append({
                            "answer": final_answer,
                            "verification": verification_results,
                            "sources": list(dict.fromkeys(all_sources))
                        })
            except Exception as e:
                logger.warning(f"Verification failed for synthesized answer: {e}")
            
            return final_answer, list(dict.fromkeys(all_sources))
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback: return concatenated answers
            final_answer = "\n\n".join([a["answer"] for a in intermediate_answers])
            return final_answer, list(dict.fromkeys(all_sources))
    
    def _get_source_chunks_for_query(self, question: str, allowed_doc_ids: Optional[List[str]]) -> List[str]:
        """
        Get source chunks for a query (for verification purposes).
        
        This performs a quick retrieval to get the chunks without generating an answer.
        """
        try:
            # Use the query pipeline's internal retrieval methods
            q_emb = self.query_pipeline.embed.embed_query(question)
            semantic_hits = self.query_pipeline.vs.search(q_emb, top_k=6, allowed_doc_ids=allowed_doc_ids)
            keyword_hits = self.query_pipeline.bm25.search(question, top_k=6, allowed_doc_ids=allowed_doc_ids)
            
            # Combine using RRF
            fused_hits = self.query_pipeline._reciprocal_rank_fusion(semantic_hits, keyword_hits)
            
            # Get top chunks
            chunks = [h.get("text", "") for h in fused_hits[:6] if h.get("text")]
            return chunks
        except Exception as e:
            logger.warning(f"Failed to get source chunks for verification: {e}")
            return []

