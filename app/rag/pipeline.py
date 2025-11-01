import io
import uuid
from typing import List, Tuple

from app.config import Settings
from app.rag.cos_client import COSClient
from app.rag.pdf_extractor import extract_text_per_page
from app.rag.chunker import chunk_pages
from app.rag.embeddings import EmbeddingClient
from app.rag.faiss_store import FaissStore
from app.rag.generator import GeneratorClient


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cos = COSClient(settings)
        self.embed = EmbeddingClient(settings)
        self.vs = FaissStore(settings)

    def upload_to_cos(self, doc_id: str, filename: str, file_obj) -> str:
        key = f"docs/{doc_id}/{filename}"
        return self.cos.upload_fileobj(key, file_obj)

    def ingest_pdf(self, doc_id: str, filename: str, source_uri: str) -> int:
        file_stream = self._fetch_cos_stream(source_uri)
        pages = extract_text_per_page(file_stream)
        non_empty_pages = [p for p in pages if p.strip()]
        chunks = chunk_pages(non_empty_pages, self.settings.chunk_size, self.settings.chunk_overlap)
        # Safety: ensure chunks fit embedding model max tokens (~256).
        # Conservative estimate: ~2.5 chars/token, so 256 tokens = ~640 chars max, use 600 to be safe
        max_embed_chars = min(self.settings.chunk_size, 600)
        safe_chunks: list[str] = []
        for c in chunks:
            if len(c) <= max_embed_chars:
                safe_chunks.append(c)
            else:
                # Word-boundary split to avoid breaking words mid-token
                words = c.split()
                current = []
                current_len = 0
                for word in words:
                    word_with_space = (" " if current else "") + word
                    if current_len + len(word_with_space) <= max_embed_chars:
                        current.append(word)
                        current_len += len(word_with_space)
                    else:
                        if current:
                            safe_chunks.append(" ".join(current))
                        current = [word]
                        current_len = len(word)
                if current:
                    safe_chunks.append(" ".join(current))
        chunks = safe_chunks
        if not chunks:
            return 0
        embeddings = self.embed.embed_texts(chunks)
        records: List[Tuple[str, str, int, int, str, List[float], str]] = []
        for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
            rec_id = str(uuid.uuid4())[:32]
            records.append((rec_id, doc_id, 0, idx, text, emb, source_uri))
        upserted = self.vs.upsert_chunks(records)
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
        self.gen = GeneratorClient(settings)

    def answer(self, question: str) -> tuple[str, List[str]]:
        q_emb = self.embed.embed_query(question)
        hits = self.vs.search(q_emb, top_k=self.settings.top_k)
        contexts = [h["text"] for h in hits]
        sources = []
        for h in hits:
            sources.append(h["source_uri"])  # could convert to signed URLs if public access not set
        # Two-step: compress retrieved contexts, then generate final answer from the summary
        try:
            summary = self.gen.compress_context(question, contexts, temperature=0.0)
            effective_contexts = [summary] if summary else contexts
        except Exception:
            effective_contexts = contexts
        answer = self.gen.generate(question, effective_contexts, temperature=self.settings.temperature)
        return answer, list(dict.fromkeys(sources))


