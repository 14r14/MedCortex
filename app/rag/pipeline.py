import io
import uuid
import logging
from typing import List, Tuple

from ibm_watsonx_ai.wml_client_error import ApiRequestFailure

from app.config import Settings
from app.rag.cos_client import COSClient
from app.rag.pdf_extractor import extract_text_per_page
from app.rag.chunker import chunk_pages
from app.rag.embeddings import EmbeddingClient
from app.rag.faiss_store import FaissStore
from app.rag.bm25_store import BM25Store
from app.rag.reranker import Reranker
from app.rag.generator import GeneratorClient

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cos = COSClient(settings)
        self.embed = EmbeddingClient(settings)
        self.vs = FaissStore(settings)
        # Initialize BM25 store with same metadata path as FAISS
        self.bm25 = BM25Store(settings.faiss_meta_path)

    def upload_to_cos(self, doc_id: str, filename: str, file_obj) -> str:
        key = f"docs/{doc_id}/{filename}"
        return self.cos.upload_fileobj(key, file_obj)

    def _split_oversized_chunk(self, chunk: str, max_chars: int = 500) -> List[str]:
        """Split a chunk that's too long into smaller chunks at word boundaries."""
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
                        result.append(word[i:i + max_chars])
                    current = []
                    current_len = 0
                else:
                    current = [word]
                    current_len = len(word)
        
        if current:
            result.append(" ".join(current))
        
        return result
    
    def _ensure_chunks_are_safe(self, chunks: List[str], max_chars: int = 500) -> List[str]:
        """Ensure all chunks are within the safe character limit."""
        safe_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                safe_chunks.append(chunk)
            else:
                # Split oversized chunks
                split_chunks = self._split_oversized_chunk(chunk, max_chars)
                safe_chunks.extend(split_chunks)
        return safe_chunks
    
    def _embed_with_retry(self, chunks: List[str], max_retries: int = 2) -> Tuple[List[List[float]], List[str]]:
        """Embed chunks with automatic retry and re-chunking on token limit errors.
        
        Returns:
            Tuple of (embeddings, safe_chunks) - the chunks that were successfully embedded
        """
        # Conservative estimate: ~2.5 chars/token, so 256 tokens = ~640 chars max
        # Use 500 chars to be extra safe and account for special tokens
        max_embed_chars = 500
        current_chunks = chunks
        
        for attempt in range(max_retries + 1):
            try:
                # Ensure chunks are safe before embedding
                safe_chunks = self._ensure_chunks_are_safe(current_chunks, max_embed_chars)
                
                if not safe_chunks:
                    return [], []
                
                embeddings = self.embed.embed_texts(safe_chunks)
                return embeddings, safe_chunks
                
            except ApiRequestFailure as e:
                error_msg = str(e)
                # Check if it's a token sequence length error
                if "Token sequence length" in error_msg or "exceeds the maximum sequence length" in error_msg:
                    if attempt < max_retries:
                        # Extract problematic chunk index if mentioned
                        # Error format: "... for text at index: 5"
                        logger.warning(f"Token limit error on attempt {attempt + 1}: {error_msg}")
                        
                        # Reduce max_chars and re-chunk all chunks
                        max_embed_chars = int(max_embed_chars * 0.8)  # Reduce by 20%
                        logger.info(f"Reducing max_embed_chars to {max_embed_chars} and re-chunking")
                        
                        # Re-chunk all chunks more aggressively
                        current_chunks = self._ensure_chunks_are_safe(current_chunks, max_embed_chars)
                        continue
                    else:
                        # Last attempt failed, try with very conservative limit
                        logger.warning("Final retry with very conservative chunk size (400 chars)")
                        max_embed_chars = 400
                        safe_chunks = self._ensure_chunks_are_safe(current_chunks, max_embed_chars)
                        if safe_chunks:
                            embeddings = self.embed.embed_texts(safe_chunks)
                            return embeddings, safe_chunks
                        else:
                            raise
                else:
                    # Different error, re-raise
                    raise
            except Exception as e:
                # Other errors, re-raise
                logger.error(f"Embedding error: {e}")
                raise
        
        return [], []
    
    def ingest_pdf(self, doc_id: str, filename: str, source_uri: str) -> int:
        file_stream = self._fetch_cos_stream(source_uri)
        pages = extract_text_per_page(file_stream)
        non_empty_pages = [p for p in pages if p.strip()]
        chunks = chunk_pages(non_empty_pages, self.settings.chunk_size, self.settings.chunk_overlap)
        
        if not chunks:
            return 0
        
        # Embed with automatic retry and re-chunking
        embeddings, safe_chunks = self._embed_with_retry(chunks)
        
        if not embeddings or not safe_chunks:
            logger.warning("No embeddings generated after retries")
            return 0
        
        if len(embeddings) != len(safe_chunks):
            # This shouldn't happen, but log if it does
            logger.warning(f"Embedding count ({len(embeddings)}) doesn't match chunk count ({len(safe_chunks)})")
            # Take the minimum to avoid index errors
            min_len = min(len(embeddings), len(safe_chunks))
            embeddings = embeddings[:min_len]
            safe_chunks = safe_chunks[:min_len]
        records: List[Tuple[str, str, int, int, str, List[float], str]] = []
        metadata_list = []
        for idx, (text, emb) in enumerate(zip(safe_chunks, embeddings)):
            rec_id = str(uuid.uuid4())[:32]
            records.append((rec_id, doc_id, 0, idx, text, emb, source_uri))
            # Prepare metadata for BM25
            metadata_list.append({
                "id": rec_id,
                "doc_id": doc_id,
                "page_num": 0,
                "chunk_index": idx,
                "text": text,
                "source_uri": source_uri,
            })
        # Upsert to FAISS
        upserted = self.vs.upsert_chunks(records)
        # Also index in BM25
        self.bm25.add_chunks(metadata_list)
        return upserted

    def _fetch_cos_stream(self, s3_url: str) -> io.BytesIO:
        # s3://bucket/key
        assert s3_url.startswith("s3://")
        _, rest = s3_url.split("s3://", 1)
        bucket, key = rest.split("/", 1)
        obj = self.cos.client.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        return io.BytesIO(data)


class QueryPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embed = EmbeddingClient(settings)
        self.vs = FaissStore(settings)
        self.bm25 = BM25Store(settings.faiss_meta_path)
        self.reranker = Reranker(settings)
        self.gen = GeneratorClient(settings)

    def _reciprocal_rank_fusion(self, semantic_hits: List[dict], keyword_hits: List[dict], k: int = 60) -> List[dict]:
        """Combine semantic and keyword search results using Reciprocal Rank Fusion (RRF)."""
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
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        fused_hits = [hit_map[hit_id] for hit_id in sorted_ids[:25]]
        return fused_hits

    def answer(self, question: str) -> tuple[str, List[str]]:
        # Step 1: Hybrid Search - Run both semantic (FAISS) and keyword (BM25) search
        q_emb = self.embed.embed_query(question)
        semantic_hits = self.vs.search(q_emb, top_k=25)  # Get more for fusion
        keyword_hits = self.bm25.search(question, top_k=25)
        
        # Step 2: Combine results using Reciprocal Rank Fusion (RRF)
        fused_hits = self._reciprocal_rank_fusion(semantic_hits, keyword_hits)
        
        # Step 3: Re-rank top 25 using cross-encoder (fallback to fused_hits if reranking fails)
        try:
            reranked_hits = self.reranker.rerank(question, fused_hits, top_k=self.settings.top_k)
            if not reranked_hits:
                # Fallback to top K from fused results if reranking returns empty
                reranked_hits = fused_hits[:self.settings.top_k]
        except Exception:
            # Fallback to fused results if reranking fails entirely
            reranked_hits = fused_hits[:self.settings.top_k]
        
        # Step 4: Extract contexts and sources from re-ranked results
        contexts = [h["text"] for h in reranked_hits]
        sources = []
        for h in reranked_hits:
            sources.append(h.get("source_uri", ""))
        
        # Step 5: Two-step generation - compress then generate
        try:
            summary = self.gen.compress_context(question, contexts, temperature=0.0)
            effective_contexts = [summary] if summary else contexts
        except Exception:
            effective_contexts = contexts
        
        answer = self.gen.generate(question, effective_contexts, temperature=self.settings.temperature)
        return answer, list(dict.fromkeys(sources))


