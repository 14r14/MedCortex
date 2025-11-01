# Setup and Running Instructions

## Prerequisites

- Python 3.11+
- IBM Cloud account with watsonx.ai, COS, Code Engine
- `uv` package manager

## Environment Setup

Set required variables via `.env` or environment. See `app/config.py` and `.env.example` for details.

## Run Locally

```bash
uv run streamlit run app/main.py --server.port 8080 --server.address 0.0.0.0
```

## Container Build

```bash
docker build -t ibm-rag-mvp:latest .
```

## Deploy (IBM Cloud Code Engine)

1. Push image to IBM Cloud Container Registry (`us.icr.io/<namespace>/ibm-rag-mvp:latest`).
2. Create a Code Engine project and app, set env vars and bind secrets.
3. Expose port 8080.

## Notes

- FAISS index and metadata are stored in session state (session-based isolation).
- COS uses IAM mode (no HMAC keys required). Presigned URLs require HMAC; IAM mode uses internal access.
- Keep services in the same region for best latency.

