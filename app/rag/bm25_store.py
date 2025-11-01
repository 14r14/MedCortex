import os
import json
from typing import List, Dict, Tuple, Optional
import streamlit as st

from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self, session_key: str = "faiss_store"):
        """Initialize BM25 store from session state (shares metadata with FaissStore)."""
        self.session_key = session_key
        self.bm25 = None
        self.chunk_map: Dict[int, Dict] = {}  # Maps BM25 index to chunk metadata
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def _load(self) -> None:
        """Load BM25 index from session state metadata (shared with FaissStore)."""
        if self.session_key in st.session_state:
            session_data = st.session_state[self.session_key]
            metadata = session_data.get("metadata", [])
            if metadata and len(metadata) > 0:
                # Rebuild BM25 index from metadata
                corpus = [self._tokenize(m.get("text", "")) for m in metadata]
                # Filter out empty texts
                valid_metadata = [m for m, c in zip(metadata, corpus) if c]
                valid_corpus = [c for c in corpus if c]
                if valid_corpus:
                    self.bm25 = BM25Okapi(valid_corpus)
                    self.chunk_map = {i: m for i, m in enumerate(valid_metadata)}
                else:
                    self.bm25 = None
                    self.chunk_map = {}
            else:
                self.bm25 = None
                self.chunk_map = {}
        else:
            self.bm25 = None
            self.chunk_map = {}

    def add_chunks(self, metadata: List[Dict]) -> None:
        """Add chunks to BM25 index. Rebuilds entire index from session state (simple but works)."""
        # Reload from session state to get all chunks (FAISS already saved them)
        # This ensures BM25 stays in sync with FAISS
        if self.session_key in st.session_state:
            session_data = st.session_state[self.session_key]
            all_metadata = session_data.get("metadata", [])
            if all_metadata:
                corpus = [self._tokenize(m.get("text", "")) for m in all_metadata]
                # Filter out empty texts
                valid_metadata = [m for m, c in zip(all_metadata, corpus) if c]
                valid_corpus = [c for c in corpus if c]
                if valid_corpus:
                    self.bm25 = BM25Okapi(valid_corpus)
                    self.chunk_map = {i: m for i, m in enumerate(valid_metadata)}

    def search(self, query: str, top_k: int = 25, allowed_doc_ids: Optional[List[str]] = None) -> List[Dict]:
        """Search using BM25 keyword matching with optional filtering by document IDs."""
        if self.bm25 is None or len(self.chunk_map) == 0:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K results (search more if filtering)
        search_k = top_k * 3 if allowed_doc_ids else top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:search_k]
        
        hits = []
        for idx in top_indices:
            if idx in self.chunk_map:
                chunk = self.chunk_map[idx].copy()
                # Filter by allowed document IDs if provided
                if allowed_doc_ids and chunk.get("doc_id") not in allowed_doc_ids:
                    continue
                chunk["bm25_score"] = float(scores[idx])
                hits.append(chunk)
                if len(hits) >= top_k:
                    break
        
        return hits

