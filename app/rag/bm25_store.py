import os
import json
from typing import List, Dict, Tuple
import pickle

from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        self.bm25 = None
        self.chunk_map: Dict[int, Dict] = {}  # Maps BM25 index to chunk metadata
        self._load()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def _load(self) -> None:
        """Load BM25 index from FAISS metadata (they share the same file)."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
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

    def _save(self) -> None:
        # Metadata is saved via FAISS store, so we don't duplicate here
        pass

    def add_chunks(self, metadata: List[Dict]) -> None:
        """Add chunks to BM25 index. Rebuilds entire index (simple but works)."""
        # Reload from metadata file to get all chunks (FAISS already saved them)
        # This ensures BM25 stays in sync with FAISS
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                all_metadata = json.load(f)
            if all_metadata:
                corpus = [self._tokenize(m.get("text", "")) for m in all_metadata]
                # Filter out empty texts
                valid_metadata = [m for m, c in zip(all_metadata, corpus) if c]
                valid_corpus = [c for c in corpus if c]
                if valid_corpus:
                    self.bm25 = BM25Okapi(valid_corpus)
                    self.chunk_map = {i: m for i, m in enumerate(valid_metadata)}

    def search(self, query: str, top_k: int = 25) -> List[Dict]:
        """Search using BM25 keyword matching."""
        if self.bm25 is None or len(self.chunk_map) == 0:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        hits = []
        for idx in top_indices:
            if idx in self.chunk_map:
                chunk = self.chunk_map[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                hits.append(chunk)
        
        return hits

