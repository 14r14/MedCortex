# MedCortex Setup and Running Instructions

This guide will help you set up and run MedCortex, the AI research analyst for medical professionals, scientists, and academics.

## Prerequisites

- Python 3.11+
- IBM Cloud account with watsonx.ai, IBM Cloud Object Storage (COS), and Code Engine
- `uv` package manager (recommended) or `pip`/`poetry`

## Environment Setup

Set required environment variables via `.env` file or system environment. Required variables include:

- **IBM Cloud API Key**: `IBM_CLOUD_API_KEY`
- **watsonx.ai Configuration**: `WATSONX_REGION`, `WATSONX_PROJECT_ID`
- **Model Configuration**: `WATSONX_EMBED_MODEL`, `WATSONX_GEN_MODEL`
- **Cloud Object Storage**: `COS_ENDPOINT`, `COS_BUCKET`, `COS_INSTANCE_CRN`, `COS_AUTH_ENDPOINT`

See `app/config.py` and `.env.example` for complete configuration details.

## Run Locally

```bash
uv run streamlit run app/main.py --server.port 8080 --server.address 0.0.0.0
```

## Container Build

```bash
docker build -t medcortex:latest .
```

## Deploy (IBM Cloud Code Engine)

1. Push image to IBM Cloud Container Registry (`us.icr.io/<namespace>/medcortex:latest`).
2. Create a Code Engine project and app, set environment variables and bind secrets.
3. Expose port 8080.
4. Ensure all required services (watsonx.ai, COS) are in the same region for optimal latency.

## Notes

- **Session-Based Storage**: All data (FAISS index, BM25 index, tables, documents) is stored in Streamlit session state, ensuring complete isolation between user sessions.
- **Cloud Object Storage**: COS uses IAM-based authentication (recommended). HMAC keys are optional and only needed for presigned URLs.
- **Performance**: Keep all IBM Cloud services (watsonx.ai, COS) in the same region for optimal latency.
- **Model Configuration**: The default embedding model is `ibm/granite-embedding-30m-english` (1024 dimensions). The generation model is `ibm/granite-3-8b-instruct`.

