import os
import json
from typing import List, Tuple, Dict, Any, Optional
import io

import faiss
import numpy as np
import streamlit as st

from app.config import Settings


class FaissStore:
    def __init__(self, settings: Settings, session_key: str = "faiss_store"):
        self.settings = settings
        self.dim = settings.embedding_dim
        self.session_key = session_key
        # Initialize session-based storage
        if session_key not in st.session_state:
            st.session_state[session_key] = {
                "embeddings": [],  # Store raw embeddings
                "metadata": [],
                "dim": settings.embedding_dim,
            }
        self._init_from_session()

    def _init_from_session(self) -> None:
        """Initialize from session state and rebuild index if needed."""
        session_data = st.session_state[self.session_key]
        self.dim = session_data.get("dim", self.settings.embedding_dim)
        self.metadata: List[Dict[str, Any]] = session_data.get("metadata", [])
        embeddings = session_data.get("embeddings", [])
        
        # Rebuild FAISS index from stored embeddings if they exist and index is empty
        if embeddings and len(embeddings) > 0:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            if embeddings_array.shape[1] != self.dim:
                self.dim = embeddings_array.shape[1]
            # Always rebuild from stored embeddings to ensure consistency
            # This ensures the index matches the stored embeddings exactly
            self._create_index(self.dim)
            embeddings_normalized = self._normalize(embeddings_array)
            self.index.add(embeddings_normalized)
        else:
            self.index = None

    def _save_to_session(self) -> None:
        """Save embeddings and metadata to session state."""
        session_data = st.session_state[self.session_key]
        # Embeddings are already stored in session state during upsert_chunks
        # Metadata is stored here
        session_data["metadata"] = self.metadata
        session_data["dim"] = self.dim

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def _create_index(self, dim: int) -> None:
        # Inner product search on normalized vectors = cosine similarity
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def upsert_chunks(self, records: List[Tuple[str, str, int, int, str, List[float], str]]) -> int:
        if not records:
            return 0
        
        # Store raw embeddings and metadata in session first
        session_data = st.session_state[self.session_key]
        embeddings_raw = [r[5] for r in records]
        session_data["embeddings"].extend(embeddings_raw)
        
        for r in records:
            self.metadata.append({
                "id": r[0],
                "doc_id": r[1],
                "page_num": r[2],
                "chunk_index": r[3],
                "text": r[4],
                "source_uri": r[6],
            })
        
        # Rebuild index from all stored embeddings to ensure consistency
        all_embeddings = session_data.get("embeddings", [])
        if all_embeddings:
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            vec_dim = embeddings_array.shape[1]
            
            # Rebuild entire index from all stored embeddings
            # This ensures consistency and avoids duplicates
            self._create_index(vec_dim)
            embeddings_normalized = self._normalize(embeddings_array)
            self.index.add(embeddings_normalized)
        
        self._save_to_session()
        return len(records)

    def search(self, query_embedding: List[float], top_k: int = 6, allowed_doc_ids: Optional[List[str]] = None) -> List[dict]:
        """Search with optional filtering by document IDs."""
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype=np.float32)
        q = self._normalize(q)
        # Search more results initially if we need to filter
        search_k = top_k * 3 if allowed_doc_ids else top_k
        scores, idxs = self.index.search(q, min(search_k, self.index.ntotal))
        hits: List[dict] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            # Filter by allowed document IDs if provided
            if allowed_doc_ids and m.get("doc_id") not in allowed_doc_ids:
                continue
            hits.append({
                **m,
                "score": float(score),
            })
            if len(hits) >= top_k:
                break
        return hits


