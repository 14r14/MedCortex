import uuid
from typing import List, Tuple

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from app.config import Settings


class MilvusStore:
    def __init__(self, settings: Settings, collection_name: str = "rag_chunks"):
        self.settings = settings
        self.collection_name = collection_name
        self._connect()
        self._ensure_collection()

    def _connect(self) -> None:
        alias = "default"
        if connections.has_connection(alias):
            return
        connections.connect(
            alias=alias,
            host=self.settings.milvus_host,
            port=str(self.settings.milvus_port),
            secure=self.settings.milvus_tls,
        )

    def _ensure_collection(self) -> None:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.settings.embedding_dim),
            FieldSchema(name="source_uri", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields=fields, description="RAG Chunks")

        if not utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name, schema=schema)
            self.collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "params": {"nlist": 1024},
                },
            )
        else:
            self.collection = Collection(self.collection_name)

        self.collection.load()

    def upsert_chunks(self, records: List[Tuple[str, str, int, int, str, List[float], str]]) -> int:
        if not records:
            return 0
        ids, doc_ids, page_nums, chunk_idxs, texts, embeddings, sources = zip(*records)
        mr = self.collection.insert(
            [
                list(ids),
                list(doc_ids),
                list(page_nums),
                list(chunk_idxs),
                list(texts),
                list(embeddings),
                list(sources),
            ]
        )
        self.collection.flush()
        return len(records)

    def search(self, query_embedding: List[float], top_k: int = 6) -> List[dict]:
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["id", "doc_id", "page_num", "chunk_index", "text", "source_uri"],
        )
        hits = []
        for hit in results[0]:
            rec = {f: hit.entity.get(f) for f in ["id", "doc_id", "page_num", "chunk_index", "text", "source_uri"]}
            rec["score"] = hit.distance
            hits.append(rec)
        return hits


