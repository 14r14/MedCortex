# VeriCite: The AI Medical Research Assistant

VeriCite is a production-ready, enterprise-grade AI assistant designed to help medical and clinical researchers synthesize vast amounts of scientific literature.   

Researchers are often overwhelmed by the volume of new papers, making it difficult to find and synthesize relevant information. This Retrieval-Augmented Generation (RAG) agent  solves this by ingesting a library of PDF documents and providing accurate, synthesized answers grounded in verifiable, citable sources.   

This application is built as a cloud-native, containerized application ready for deployment on IBM Cloud Code Engine, leveraging a modern Python stack and powerful services from IBM Cloud.

## ✨ Core Features

### Core RAG Pipeline

- **Advanced Document Ingestion**: Features multi-PDF parallel processing with automatic text extraction (PyPDF).

- **Intelligent Text Chunking**: Uses RecursiveCharacterTextSplitter with semantic overlap to preserve the context and meaning of complex medical text.   

- **High-Performance Vector Search**: Employs a local FAISS index for efficient, low-latency similarity search on high-dimensional embeddings.   

- **Semantic Retrieval System**: A configurable top-K retriever with automatic deduplication of sources to ensure a clean context.

### Intelligent Answer Generation (with watsonx.ai)

- **Two-Stage Generation**: A sophisticated pipeline that first compresses context from retrieved chunks to distill relevant facts before passing them to the generator for the final answer.

- **Streaming Generation**: Streams tokens directly from the IBM Granite-3-8B-Instruct model for a real-time, responsive chat experience.   

- **Strict Source Attribution**: All generated answers are fully grounded in the provided documents. The system uses a "sources list" in the prompt to prevent the LLM from referencing hallucinated sources.

### Cloud Infrastructure Integration (IBM Cloud)

- **IBM Cloud Object Storage (COS)**: Securely uploads and manages source PDFs using IAM-based authentication, integrated with a scalable S3-compatible API.

- **watsonx.ai Foundation Models**: Uses the latest ModelInference API to connect to IBM's powerful Granite models for generation and embedding.

- **Production-Ready Containerization**: Includes a Dockerfile with multi-stage build optimization, a non-root user, and pre-configured for deployment on IBM Cloud Code Engine.

### User Interface & Architecture

- **Streamlit Single-Page Application**: A modern, real-time chat interface with full message history.

- **Interactive Document Management**: Features a multi-file uploader and a sidebar that tracks ingested documents and their chunk counts.

- **Modern Python Stack**: Built with uv for package management, Pydantic for data validation, and flexible configuration via environment variables (no hardcoded credentials).

For setup and running instructions, see [SETUP.md](SETUP.md).
