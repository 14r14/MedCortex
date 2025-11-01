import os
import json
from typing import List, Tuple, Dict, Any

import faiss
import numpy as np

from app.config import Settings


class FaissStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dim = settings.embedding_dim
        self.index_path = settings.faiss_index_path
        self.meta_path = settings.faiss_meta_path
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._load()

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def _create_index(self, dim: int) -> None:
        # Inner product search on normalized vectors = cosine similarity
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            self.dim = self.index.d
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            # Defer index creation until first upsert so we can infer dim
            self.index = None
            self.metadata = []

    def _save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def upsert_chunks(self, records: List[Tuple[str, str, int, int, str, List[float], str]]) -> int:
        if not records:
            return 0
        # Append to index and metadata; idempotency is not enforced in this simple MVP
        embeddings = np.array([r[5] for r in records], dtype=np.float32)
        embeddings = self._normalize(embeddings)
        # Ensure index exists and matches dimension
        vec_dim = embeddings.shape[1]
        if self.index is None:
            self._create_index(vec_dim)
        elif self.index.d != vec_dim:
            if self.index.ntotal == 0:
                # Safe to recreate with new dimension
                self._create_index(vec_dim)
            else:
                raise RuntimeError(
                    f"FAISS index dimension mismatch: index.d={self.index.d} vs embeddings.d={vec_dim}. "
                    f"Either set EMBEDDING_DIM={self.index.d} and use the same embedding model, or delete the existing "
                    f"FAISS files ({self.index_path}, {self.meta_path}) to rebuild with the new dimension."
                )
        self.index.add(embeddings)
        for r in records:
            self.metadata.append({
                "id": r[0],
                "doc_id": r[1],
                "page_num": r[2],
                "chunk_index": r[3],
                "text": r[4],
                "source_uri": r[6],
            })
        self._save()
        return len(records)

    def search(self, query_embedding: List[float], top_k: int = 6) -> List[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.array([query_embedding], dtype=np.float32)
        q = self._normalize(q)
        scores, idxs = self.index.search(q, top_k)
        hits: List[dict] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            hits.append({
                **m,
                "score": float(score),
            })
        return hits


