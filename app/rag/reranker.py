from app.config import Settings


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Note: Cross-encoder models don't support text generation API in watsonx.ai
        # Using keyword-based reranking as a hybrid approach
        # This combines keyword overlap with semantic signals from hybrid search

    def rerank(self, query: str, candidates: list[dict], top_k: int = 6) -> list[dict]:
        """
        Re-rank candidates using keyword-based scoring combined with hybrid search signals.
        Returns top_k re-ranked candidates.
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            # If we have fewer candidates than top_k, just return them
            return candidates

        # Score each candidate using combined signals
        scored_candidates = []
        for candidate in candidates:
            chunk_text = candidate.get("text", "")
            score = self._score_pair(query, chunk_text, candidate)
            candidate_with_score = candidate.copy()
            candidate_with_score["rerank_score"] = score
            scored_candidates.append(candidate_with_score)

        # Sort by rerank score (descending) and return top_k
        scored_candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return scored_candidates[:top_k]

    def _score_pair(self, query: str, document: str, candidate: dict) -> float:
        """Score a query-document pair using keyword overlap and hybrid search signals."""
        # Combine multiple signals for reranking

        # Signal 1: Keyword overlap (Jaccard similarity)
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        intersection = len(query_terms & doc_terms)
        union = len(query_terms | doc_terms)
        jaccard_score = (intersection / union) if union > 0 else 0.0

        # Signal 2: Exact phrase match boost
        query_lower = query.lower()
        doc_lower = document.lower()
        phrase_boost = 0.3 if query_lower in doc_lower else 0.0

        # Signal 3: Use existing scores from hybrid search (FAISS similarity, BM25, RRF)
        semantic_score = candidate.get("score", 0.0)  # From FAISS
        bm25_score = candidate.get("bm25_score", 0.0)  # From BM25

        # Normalize scores to 0-1 range and combine
        # Assuming FAISS uses inner product (cosine similarity), scores are typically 0-1
        # BM25 scores can vary widely, so normalize them
        normalized_semantic = (
            max(0, min(1, semantic_score)) if semantic_score > 0 else 0
        )
        # Normalize BM25 score (typical BM25 scores are positive, normalize by max reasonable value)
        normalized_bm25 = min(1.0, bm25_score / 10.0) if bm25_score > 0 else 0

        # Weighted combination: 40% semantic, 30% keyword (Jaccard), 20% BM25, 10% phrase boost
        final_score = (
            0.4 * normalized_semantic
            + 0.3 * jaccard_score
            + 0.2 * normalized_bm25
            + 0.1 * phrase_boost
        )

        return min(final_score, 1.0)
