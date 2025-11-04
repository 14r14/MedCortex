"""RAG pipeline for document ingestion and query processing.

This module provides the IngestionPipeline and QueryPipeline classes
for document processing and query answering.
"""

import io
import logging
import uuid

from ibm_watsonx_ai.wml_client_error import ApiRequestFailure

from app.config import Settings
from app.rag.bm25_store import BM25Store
from app.rag.chunker import chunk_pages
from app.rag.cos_client import COSClient
from app.rag.embeddings import EmbeddingClient
from app.rag.faiss_store import FaissStore
from app.rag.generator import GeneratorClient
from app.rag.orchestrator import Orchestrator
from app.rag.pdf_extractor import extract_metadata, extract_text_per_page
from app.rag.reranker import Reranker
from app.rag.table_extractor import extract_tables_camelot, store_tables_in_session
from app.rag.verifier import AnswerVerifier

try:
    import streamlit as st
except ImportError:
    # For testing outside Streamlit
    st = None

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting and indexing documents.

    Handles document upload, text extraction, chunking, embedding,
    and storage in vector and keyword search indices.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize ingestion pipeline.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.cos = COSClient(settings)
        self.embed = EmbeddingClient(settings)
        self.vs = FaissStore(settings, session_key="faiss_store")
        # Initialize BM25 store - shares session state with FAISS
        self.bm25 = BM25Store(session_key="faiss_store")

    def upload_to_cos(self, doc_id: str, filename: str, file_obj) -> str:
        """Upload a file to Cloud Object Storage.

        Args:
            doc_id: Document identifier.
            filename: Name of the file.
            file_obj: File object to upload.

        Returns:
            S3 URI of the uploaded file.
        """
        key = f"docs/{doc_id}/{filename}"
        return self.cos.upload_fileobj(key, file_obj)

    def _split_oversized_chunk(self, chunk: str, max_chars: int = 500) -> list[str]:
        """Split a chunk that's too long into smaller chunks at word boundaries.

        Args:
            chunk: Chunk text to split.
            max_chars: Maximum characters per chunk.

        Returns:
            List of split chunks.
        """
        if len(chunk) <= max_chars:
            return [chunk]

        result = []
        words = chunk.split()
        current = []
        current_len = 0

        for word in words:
            word_with_space = (" " if current else "") + word
            if current_len + len(word_with_space) <= max_chars:
                current.append(word)
                current_len += len(word_with_space)
            else:
                if current:
                    result.append(" ".join(current))
                # Handle single words longer than max_chars (very rare, but possible)
                if len(word) > max_chars:
                    # Split long word by characters (last resort)
                    for i in range(0, len(word), max_chars):
                        result.append(word[i : i + max_chars])
                    current = []
                    current_len = 0
                else:
                    current = [word]
                    current_len = len(word)

        if current:
            result.append(" ".join(current))

        return result

    def _ensure_chunks_are_safe(
        self, chunks: list[str], max_chars: int = 500
    ) -> list[str]:
        """Ensure all chunks are within the safe character limit.

        Args:
            chunks: List of chunks to validate.
            max_chars: Maximum characters per chunk.

        Returns:
            List of safe chunks (split if necessary).
        """
        safe_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                safe_chunks.append(chunk)
            else:
                # Split oversized chunks
                split_chunks = self._split_oversized_chunk(chunk, max_chars)
                safe_chunks.extend(split_chunks)
        return safe_chunks

    def _embed_with_retry(
        self, chunks: list[str], max_retries: int = 2
    ) -> tuple[list[list[float]], list[str]]:
        """Embed chunks with automatic retry and re-chunking on token limit errors.

        Args:
            chunks: List of text chunks to embed.
            max_retries: Maximum number of retry attempts.

        Returns:
            Tuple of (embeddings, safe_chunks) - the chunks that were
            successfully embedded.
        """
        # Conservative estimate: ~2.5 chars/token, so 256 tokens = ~640 chars max
        # Use 500 chars to be extra safe and account for special tokens
        max_embed_chars = 500
        current_chunks = chunks

        for attempt in range(max_retries + 1):
            try:
                # Ensure chunks are safe before embedding
                safe_chunks = self._ensure_chunks_are_safe(
                    current_chunks, max_embed_chars
                )

                if not safe_chunks:
                    return [], []

                embeddings = self.embed.embed_texts(safe_chunks)
                return embeddings, safe_chunks

            except ApiRequestFailure as e:
                error_msg = str(e)
                # Check if it's a token sequence length error
                if (
                    "Token sequence length" in error_msg
                    or "exceeds the maximum sequence length" in error_msg
                ):
                    if attempt < max_retries:
                        # Extract problematic chunk index if mentioned
                        # Error format: "... for text at index: 5"
                        logger.warning(
                            f"Token limit error on attempt {attempt + 1}: {error_msg}"
                        )

                        # Reduce max_chars and re-chunk all chunks
                        max_embed_chars = int(max_embed_chars * 0.8)  # Reduce by 20%
                        logger.info(
                            f"Reducing max_embed_chars to {max_embed_chars} and re-chunking"
                        )

                        # Re-chunk all chunks more aggressively
                        current_chunks = self._ensure_chunks_are_safe(
                            current_chunks, max_embed_chars
                        )
                        continue
                    # Last attempt failed, try with very conservative limit
                    logger.warning(
                        "Final retry with very conservative chunk size (400 chars)"
                    )
                    max_embed_chars = 400
                    safe_chunks = self._ensure_chunks_are_safe(
                        current_chunks, max_embed_chars
                    )
                    if safe_chunks:
                        embeddings = self.embed.embed_texts(safe_chunks)
                        return embeddings, safe_chunks
                    raise
                # Different error, re-raise
                raise
            except Exception as e:
                # Other errors, re-raise
                logger.error(f"Embedding error: {e}")
                raise

        return [], []

    def ingest_pdf(
        self, doc_id: str, filename: str, source_uri: str
    ) -> tuple[int, dict[str, str | None]]:
        """Ingest a PDF document into the knowledge base.

        Args:
            doc_id: Document identifier.
            filename: Name of the PDF file.
            source_uri: S3 URI of the source document.

        Returns:
            Tuple of (upserted_count, metadata_dict) where metadata_dict
            contains title and author if available.
        """
        file_stream = self._fetch_cos_stream(source_uri)

        # Extract metadata (title, author) - need to read file first
        file_stream.seek(0)
        metadata = extract_metadata(file_stream)

        # Reset stream for table extraction
        file_stream.seek(0)
        # Extract tables for TableRAG using camelot
        dataframes = extract_tables_camelot(file_stream, doc_id)
        if dataframes:
            store_tables_in_session(dataframes, doc_id)
            logger.info(f"Extracted {len(dataframes)} tables from {filename}")

        # Reset stream for text extraction
        file_stream.seek(0)
        pages = extract_text_per_page(file_stream)
        non_empty_pages = [p for p in pages if p.strip()]
        chunks = chunk_pages(
            non_empty_pages, self.settings.chunk_size, self.settings.chunk_overlap
        )

        if not chunks:
            return 0

        # Embed with automatic retry and re-chunking
        embeddings, safe_chunks = self._embed_with_retry(chunks)

        if not embeddings or not safe_chunks:
            logger.warning("No embeddings generated after retries")
            return 0

        if len(embeddings) != len(safe_chunks):
            # This shouldn't happen, but log if it does
            logger.warning(
                f"Embedding count ({len(embeddings)}) doesn't match chunk count ({len(safe_chunks)})"
            )
            # Take the minimum to avoid index errors
            min_len = min(len(embeddings), len(safe_chunks))
            embeddings = embeddings[:min_len]
            safe_chunks = safe_chunks[:min_len]
        records: list[tuple[str, str, int, int, str, list[float], str]] = []
        metadata_list = []
        for idx, (text, emb) in enumerate(zip(safe_chunks, embeddings)):
            rec_id = str(uuid.uuid4())[:32]
            records.append((rec_id, doc_id, 0, idx, text, emb, source_uri))
            # Prepare metadata for BM25
            metadata_list.append(
                {
                    "id": rec_id,
                    "doc_id": doc_id,
                    "page_num": 0,
                    "chunk_index": idx,
                    "text": text,
                    "source_uri": source_uri,
                }
            )
        # Upsert to FAISS (stores in session state)
        upserted = self.vs.upsert_chunks(records)
        # Also index in BM25 (rebuilds from session state metadata)
        self.bm25.add_chunks(metadata_list)
        return upserted, metadata

    def _fetch_cos_stream(self, s3_url: str) -> io.BytesIO:
        """Fetch a file from Cloud Object Storage as a stream.

        Args:
            s3_url: S3 URI of the file (format: s3://bucket/key).

        Returns:
            BytesIO stream containing the file content.
        """
        # s3://bucket/key
        assert s3_url.startswith("s3://")
        _, rest = s3_url.split("s3://", 1)
        bucket, key = rest.split("/", 1)
        obj = self.cos.client.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        return io.BytesIO(data)


class QueryPipeline:
    """Pipeline for processing queries and generating answers.

    Handles query embedding, retrieval (semantic and keyword),
    reranking, context compression, generation, and verification.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize query pipeline.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.embed = EmbeddingClient(settings)
        self.vs = FaissStore(settings, session_key="faiss_store")
        self.bm25 = BM25Store(session_key="faiss_store")
        self.reranker = Reranker(settings)
        self.gen = GeneratorClient(settings)
        self.verifier = AnswerVerifier(settings)
        # Orchestrator will be initialized lazily to avoid circular dependency
        self._orchestrator = None

    def _reciprocal_rank_fusion(
        self,
        semantic_hits: list[dict],
        keyword_hits: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Combine semantic and keyword search results using Reciprocal Rank Fusion (RRF).

        Args:
            semantic_hits: Results from semantic (vector) search.
            keyword_hits: Results from keyword (BM25) search.
            k: RRF parameter (default: 60).

        Returns:
            Combined and ranked results.
        """
        # Map chunk IDs to their hits and RRF scores
        hit_map = {}  # chunk_id -> hit
        rrf_scores = {}  # chunk_id -> RRF score

        # Score semantic hits
        for rank, hit in enumerate(semantic_hits, start=1):
            hit_id = hit.get("id")
            if not hit_id:
                # Use text as fallback ID
                hit_id = hit.get("text", "")[:100]
            hit_map[hit_id] = hit
            rrf_scores[hit_id] = rrf_scores.get(hit_id, 0.0) + (1.0 / (k + rank))

        # Score keyword hits
        for rank, hit in enumerate(keyword_hits, start=1):
            hit_id = hit.get("id")
            if not hit_id:
                # Use text as fallback ID
                hit_id = hit.get("text", "")[:100]
            if hit_id not in hit_map:
                hit_map[hit_id] = hit
            rrf_scores[hit_id] = rrf_scores.get(hit_id, 0.0) + (1.0 / (k + rank))

        # Sort by RRF score and return top 25 hits for re-ranking
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )
        fused_hits = [hit_map[hit_id] for hit_id in sorted_ids[:25]]
        return fused_hits

    def _is_complex_query(self, question: str) -> bool:
        """Detect if a query is complex enough to require iterative reasoning.

        Complex queries typically:
        - Ask multiple related questions (e.g., "What is X and what are its Y?")
        - Reference multiple documents/contexts
        - Require multi-hop reasoning (e.g., "What is the standard treatment
          for condition A as described in paper B?")

        Args:
            question: User question to analyze.

        Returns:
            True if query is complex, False otherwise.
        """
        # Keywords that suggest complexity
        complexity_indicators = [
            " and ",
            " also ",
            " furthermore ",
            " additionally ",
            " what ",
            " how ",
            " why ",
            " compare ",
            " difference ",
            " as described ",
            " according to ",
            " in paper ",
            " in document ",
            " standard ",
            " treatment ",
            " side effects ",
            " outcomes ",
        ]

        question_lower = question.lower()

        # Count question words (multiple questions suggest complexity)
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        question_count = sum(1 for word in question_words if word in question_lower)

        # Count complexity indicators
        indicator_count = sum(
            1 for indicator in complexity_indicators if indicator in question_lower
        )

        # Consider complex if:
        # - Multiple question words
        # - Multiple complexity indicators
        # - Question is long (> 50 chars) with indicators
        is_complex = (
            question_count >= 2
            or indicator_count >= 2
            or (len(question) > 50 and indicator_count >= 1)
        )

        return is_complex

    def answer(
        self,
        question: str,
        allowed_doc_ids: list[str] | None = None,
        use_orchestrator: bool | None = None,
    ) -> tuple[str, list[str]]:
        """Answer question with optional filtering by document IDs (session-based).

        Uses orchestrator for complex queries, standard RAG for simple queries.

        Args:
            question: User question.
            allowed_doc_ids: Optional list of allowed document IDs.
            use_orchestrator: Optional override flag. If None, auto-detect.
                If False, skip orchestrator.

        Returns:
            Tuple of (answer, sources).
        """
        # Detect if query is complex (unless explicitly disabled)
        if use_orchestrator is None:
            use_iterative = self._is_complex_query(question)
        else:
            use_iterative = use_orchestrator

        if use_iterative:
            logger.info("Using orchestrator for complex query")
            try:
                # Initialize orchestrator lazily
                if self._orchestrator is None:
                    self._orchestrator = Orchestrator(self.settings, self)
                # Check if Streamlit is available for trajectory visualization
                show_trajectory = st is not None
                # Get status callback from session state if available
                status_callback = (
                    st.session_state.get("_status_callback") if st is not None else None
                )
                result = self._orchestrator.answer_iteratively(
                    question,
                    allowed_doc_ids=allowed_doc_ids,
                    show_trajectory=show_trajectory,
                    status_callback=status_callback,
                )
                answer, sources, trajectory = (
                    result if len(result) == 3 else (result[0], result[1], None)
                )
                # Store trajectory in session state for UI display
                if show_trajectory and trajectory:
                    if "agent_trajectory" not in st.session_state:
                        st.session_state["agent_trajectory"] = []
                    st.session_state["agent_trajectory"].append(
                        {"query": question, "trajectory": trajectory, "answer": answer}
                    )
                return answer, list(dict.fromkeys(sources))
            except Exception as e:
                logger.warning(
                    f"Orchestrator failed: {e}, falling back to standard RAG"
                )
                # Fall through to standard RAG

        # Standard RAG pipeline for simple queries
        # Step 1: Hybrid Search - Run both semantic (FAISS) and keyword (BM25) search
        q_emb = self.embed.embed_query(question)
        semantic_hits = self.vs.search(q_emb, top_k=25, allowed_doc_ids=allowed_doc_ids)
        keyword_hits = self.bm25.search(
            question, top_k=25, allowed_doc_ids=allowed_doc_ids
        )

        # Step 2: Combine results using Reciprocal Rank Fusion (RRF)
        fused_hits = self._reciprocal_rank_fusion(semantic_hits, keyword_hits)

        # Step 3: Re-rank top 25 using cross-encoder (fallback to fused_hits if reranking fails)
        try:
            reranked_hits = self.reranker.rerank(
                question, fused_hits, top_k=self.settings.top_k
            )
            if not reranked_hits:
                # Fallback to top K from fused results if reranking returns empty
                reranked_hits = fused_hits[: self.settings.top_k]
        except Exception:
            # Fallback to fused results if reranking fails entirely
            reranked_hits = fused_hits[: self.settings.top_k]

        # Step 4: Extract contexts and sources from re-ranked results
        contexts = [h["text"] for h in reranked_hits]
        sources = []
        for h in reranked_hits:
            sources.append(h.get("source_uri", ""))

        # Step 5: Table queries are now handled by orchestrator/router
        # For simple queries, we just use text-based retrieval

        # Step 6: Two-step generation - compress then generate
        try:
            summary = self.gen.compress_context(question, contexts, temperature=0.0)
            effective_contexts = [summary] if summary else contexts
        except Exception:
            effective_contexts = contexts

        answer = self.gen.generate(
            question, effective_contexts, temperature=self.settings.temperature
        )

        # Step 7: Verify answer against source chunks (for simple queries)
        # Note: Complex queries handled by orchestrator with trajectory
        try:
            verification_results = self.verifier.verify_answer(answer, contexts)
            # Store verification results in session state for UI display
            if st is not None:
                if "verification_results" not in st.session_state:
                    st.session_state["verification_results"] = []
                st.session_state["verification_results"].append(
                    {
                        "answer": answer,
                        "verification": verification_results,
                        "sources": list(dict.fromkeys(sources)),
                    }
                )
        except Exception as e:
            logger.warning(f"Answer verification failed: {e}")
            # Continue without verification

        return answer, list(dict.fromkeys(sources))
