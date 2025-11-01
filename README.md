# IBM Cloud RAG MVP

Streamlit single-page app with RAG pipeline using watsonx.ai (Granite-13B for answers, `ibm/granite-embedding-30m-english` for embeddings), FAISS vector search, and IBM Cloud Object Storage.

## Prerequisites
- Python 3.11+
- IBM Cloud account with watsonx.ai, COS, watsonx.data (Milvus), Code Engine
- `uv` package manager

## Environment
Set required variables via `.env` or environment. See `app/config.py` and the project plan for details.

## Run locally
```bash
uv run streamlit run app/main.py --server.port 8080 --server.address 0.0.0.0
```

## Container
```bash
docker build -t ibm-rag-mvp:latest .
```

## Deploy (IBM Cloud Code Engine)
- Push image to IBM Cloud Container Registry (`us.icr.io/<namespace>/ibm-rag-mvp:latest`).
- Create a Code Engine project and app, set env vars and bind secrets.
- Expose port 8080.

## Notes
- FAISS index and metadata are persisted to local files (see `FAISS_INDEX_PATH`, `FAISS_META_PATH`). For persistence across deploys, back them up to COS.
- COS can use IAM (no HMAC). Presigned URLs require HMAC; IAM mode uses internal access.
- Keep services in the same region for best latency.


