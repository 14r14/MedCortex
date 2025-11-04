# MedCortex
## Research Analyst • Verifiable Synthesis • Powered by watsonx.ai

MedCortex is an advanced AI research analyst designed for medical professionals, scientists, and academics. It's built to solve the "information synthesis headache"—the overwhelming and manual process of finding, analyzing, and citing information from dense scientific literature.

This is not just another "chat with your docs" app. MedCortex is a complete, agentic workspace that moves beyond simple search to provide deep analysis, trustworthy verification, and a final, exportable deliverable.

## Value Propositions

The MedCortex platform is built on three core principles that directly address the pain points of medical researchers.

### 1. Advanced Agentic Reasoning (The "Analysis Breakdown")

When you ask a complex question, a simple RAG tool will fail. MedCortex excels here. As shown in the UI, it features an "Analysis Breakdown" (inspired by "Chain of Thought" reasoning). It performs Query Analysis and Query Decomposition before answering, executing a multi-step plan to retrieve and analyze information from both unstructured text and structured tables.

### 2. Built-in Claim Verification (The "Trust Layer")

MedCortex targets the number one anxiety of using AI for research: hallucinations. Every generated claim is automatically fact-checked against its source documents using a Natural Language Inference (NLI) model. Findings in the chat are clearly and instantly tagged as "VERIFIED" or "REFUTED" (like the "EXTRAPOLATED" label), giving you full transparency and the confidence to use its output.

### 3. A Complete Workspace (The "Synthesis Studio")

Your workflow doesn't end with an answer. As you find verified insights in the chat, you can add them to the "Synthesis Studio". This is your curation workspace, where you build your final "Deliverable"—a concept central to any professional research plan. When ready, export your curated, verified findings and citations as a formatted .docx or Markdown file. This feature turns hours of manual report-writing into minutes of curation.

## What It Does

- **Chat Analyst**: An interactive chat UI (the "MedCortex Analyst") where you ask complex, natural language questions.
- **Multi-Modal Ingestion**: Upload your PDF documents. The engine processes heterogeneous data, extracting both unstructured text (for semantic search) and structured tables (for queryable, SQL-like analysis).
- **Domain-Specific Hybrid Search**: Uses a combination of modern vector search (FAISS) and keyword search (BM25) to find both broad concepts and specific medical terminology (like gene names or drug abbreviations).
- **Agentic Routing (Text vs. Table)**: The agent's "Query Analysis" step intelligently routes your question to the correct tool—it performs vector search for text-based questions and structured data analysis for table-based questions.
- **In-Line Verification**: Every generated claim is tagged with VERIFIED or UNSUPPORTED for immediate quality control.
- **Curated Report Exporting**: The "Synthesis Studio" lets you curate all your verified findings into a single document and export it to .docx or Markdown.

## Technology

This project is built to be an enterprise-grade, cloud-native application.

- **Frontend**: Streamlit
- **AI Orchestration & LLM**: IBM watsonx.ai
- **Foundation Model**: IBM Granite-3-8B-Instruct
- **Embedding Model**: MEDTE (a domain-specific model for biomedical text) or `ibm/granite-embedding-30m-english`
- **Retrieval**: Hybrid Search (FAISS + BM25) with custom signal-based reranking
- **Verification**: NLI-based claim verification
- **Deployment**: Containerized for IBM Cloud Code Engine

For setup and running instructions, see [SETUP.md](SETUP.md).

---

## 🔬 Technical Architecture & Implementation Details

### System Architecture Overview

MedCortex implements a sophisticated multi-stage RAG (Retrieval-Augmented Generation) pipeline with an agentic iterative framework, hybrid search capabilities, and post-generation verification. The system processes queries through two primary pathways:

1. **Simple Query Path**: Direct RAG pipeline with hybrid search, reranking, and answer generation
2. **Complex Query Path**: Iterative agentic framework (i-MedRAG inspired) with query decomposition, multi-hop retrieval, and synthesis

### Document Ingestion Pipeline

The ingestion process transforms raw PDF documents into searchable, semantically-encoded chunks stored in a session-based vector store.

#### Step 1: Document Upload & Storage
- **Method**: PDFs are uploaded via Streamlit UI and stored in **IBM Cloud Object Storage (COS)** using the `ibm-cos-sdk` library
- **Authentication**: IAM-based authentication (using `COS_API_KEY`, `COS_INSTANCE_CRN`, `COS_AUTH_ENDPOINT`)
- **Storage Format**: Files are stored with path structure `docs/{doc_id}/{filename}`
- **Library**: `ibm-cos-sdk` (S3-compatible API)

#### Step 2: Text Extraction
- **Method**: **PyPDF** library extracts text content page-by-page
- **Process**: Each page is extracted as a separate text block, preserving document structure
- **Library**: `pypdf>=4.2`

#### Step 3: Table Extraction (TableRAG)
- **Method**: **Camelot-py** with `flavor='lattice'` for structured table extraction
- **Output**: Extracted tables are stored as **pandas DataFrames** in Streamlit session state
- **Storage**: `st.session_state["table_store"][doc_id] = [df1, df2, ...]`
- **Library**: `camelot-py[cv]>=0.11.0` with OpenCV dependency
- **Purpose**: Enables quantitative data queries via LLM-generated pandas code execution

#### Step 4: Text Chunking
- **Algorithm**: **RecursiveCharacterTextSplitter** (LangChain)
- **Configuration**:
  - `chunk_size`: 1200 characters (default, configurable via `CHUNK_SIZE`)
  - `chunk_overlap`: 150 characters (default, configurable via `CHUNK_OVERLAP`)
  - `separators`: `["\n\n", "\n", " ", ""]` (hierarchical splitting)
- **Process**: Splits text at semantic boundaries (paragraphs, sentences, words) to preserve context
- **Library**: `langchain-text-splitters>=0.2`
- **Token Limit Handling**: Automatic re-chunking for oversized chunks (>500 chars) to comply with embedding model token limits (256 tokens for `sentence-transformers/all-minilm-l6-v2`)

#### Step 5: Vector Embedding
- **Model**: **watsonx.ai Embeddings API** using `ibm-watsonx-ai` SDK
- **Default Model**: `sentence-transformers/all-minilm-l6-v2` (384 dimensions) or `ibm/granite-embedding-30m-english` (1024 dimensions)
- **API**: `WXEmbeddings.embed_documents()` for batch embedding
- **Process**: 
  - Chunks are embedded in batches using the watsonx.ai API
  - Embeddings are normalized (L2 normalization) for cosine similarity calculation
  - Response parsing handles multiple SDK output formats: `{"results": [{"embedding"|"vector"|"values": [...]}]}`, `{"embeddings": [[...]]}`, or direct list format
- **Error Handling**: Automatic retry with re-chunking on token limit errors (`ApiRequestFailure: Token sequence length exceeds maximum`)

#### Step 6: Vector Index Storage
- **Algorithm**: **FAISS (Facebook AI Similarity Search)**
- **Index Type**: `IndexFlatIP` (Inner Product on normalized vectors = Cosine Similarity)
- **Storage**: **Session-based** (Streamlit session state) - no persistent disk storage
- **Data Structure**: 
  ```python
  st.session_state["faiss_store"] = {
      "embeddings": [[float, ...], ...],  # Raw embedding vectors
      "metadata": [{id, doc_id, page_num, chunk_index, text, source_uri}, ...],
      "dim": 384  # Embedding dimension
  }
  ```
- **Process**: 
  - Embeddings stored as raw float arrays in session state
  - FAISS index rebuilt from stored embeddings on each session initialization
  - Index supports filtering by `doc_id` for session-based document isolation
- **Library**: `faiss-cpu>=1.7.4`
- **Metric**: Cosine Similarity (via inner product on L2-normalized vectors)

#### Step 7: Keyword Index Storage (BM25)
- **Algorithm**: **BM25 (Best Matching 25)** - Probabilistic ranking function for keyword search
- **Library**: `rank-bm25>=0.2.2` (BM25Okapi implementation)
- **Storage**: **Session-based** - shares metadata with FAISS store
- **Process**:
  - Tokenizes text using simple whitespace splitting (`text.lower().split()`)
  - Builds BM25 index from chunk text corpus
  - Stores mapping: `chunk_map[index] = metadata` for result retrieval
  - Rebuilds index from session state on initialization
- **Purpose**: Provides keyword-based search to complement semantic search for medical terminology, drug names, gene markers

### Query Processing Pipeline

#### Simple Query Path (Standard RAG)

For straightforward queries, the system uses a direct RAG pipeline:

##### Step 1: Query Embedding
- **Method**: Embed user query using same embedding model as ingestion
- **API**: `WXEmbeddings.embed_query()` or `embed_documents([query])`
- **Output**: Normalized query vector (384 or 1024 dimensions)

##### Step 2: Hybrid Search

The system performs **dual retrieval** combining semantic and keyword search:

**2a. Semantic Search (FAISS)**
- **Algorithm**: **FAISS IndexFlatIP** - Exact nearest neighbor search via inner product
- **Process**:
  - Query vector normalized (L2 normalization)
  - `index.search(query_vector, top_k=25)` returns top 25 most similar chunks
  - Similarity scores are cosine similarity values (0-1 range)
- **Filtering**: Optional `allowed_doc_ids` parameter filters results to session-specific documents

**2b. Keyword Search (BM25)**
- **Algorithm**: **BM25** - Term frequency-inverse document frequency (TF-IDF) based ranking
- **Process**:
  - Query tokenized: `query.lower().split()`
  - `bm25.get_scores(tokenized_query)` calculates BM25 scores for all chunks
  - Top 25 chunks by BM25 score retrieved
  - Scores can vary widely (typically 0-20+ range, normalized later)
- **Filtering**: Optional `allowed_doc_ids` parameter for session-based filtering

##### Step 3: Rank Fusion
- **Algorithm**: **Reciprocal Rank Fusion (RRF)**
- **Formula**: `RRF_score(doc) = Σ(1 / (k + rank(doc, list_i)))` for all lists i
  - `k`: Constant (typically 60) to prevent divide-by-zero and smooth ranking differences
  - `rank(doc, list_i)`: Position of document in list i (1-indexed)
- **Process**:
  - Combines ranked lists from FAISS and BM25 search
  - Calculates RRF score for each unique chunk
  - Sorts by RRF score (descending) to get fused ranking
- **Output**: Top-ranked chunks from hybrid search (typically top 25-30)

##### Step 4: Re-ranking
- **Algorithm**: **Hybrid Signal-Based Reranking**
- **Signals Combined**:
  1. **Semantic Score (40% weight)**: Normalized FAISS cosine similarity (0-1 range)
  2. **Jaccard Similarity (30% weight)**: Keyword overlap: `|query_terms ∩ doc_terms| / |query_terms ∪ doc_terms|`
  3. **BM25 Score (20% weight)**: Normalized BM25 score (normalized by dividing by 10)
  4. **Phrase Match Boost (10% weight)**: Binary indicator (0.3 if query phrase found in document, else 0.0)
- **Formula**: 
  ```
  rerank_score = 0.4 * normalized_semantic + 
                 0.3 * jaccard_score + 
                 0.2 * normalized_bm25 + 
                 0.1 * phrase_boost
  ```
- **Output**: Top K chunks (default K=6, configurable via `TOP_K`) reordered by rerank score

##### Step 5: Context Compression
- **Method**: **LLM-based Context Compression** (two-stage generation)
- **Model**: Same generation model (Granite-3-8B-Instruct) with `temperature=0.0`
- **Process**:
  - Takes top K re-ranked chunks as context
  - Generates compressed summary retaining all critical details (quantitative data, methodologies, findings)
  - Prompt instructs model to preserve specificity for medical researchers
- **Purpose**: Reduces context length while preserving essential information for final answer generation

##### Step 6: Answer Generation
- **Model**: **IBM Granite-3-8B-Instruct** via watsonx.ai ModelInference API
- **Configuration**:
  - `temperature`: 0.2 (default, configurable via `TEMPERATURE`)
  - `max_new_tokens`: 4096 (for comprehensive medical research answers)
  - `return_options`: Includes input text for prompt tracking
- **Prompt Structure**:
  - System prompt defines medical research assistant persona
  - Context: Compressed context or original chunks
  - Instructions: Detailed answer requirements (quantitative data, methodologies, findings, terminology)
  - Explicit instructions against placeholder citations and meta-commentary
- **API**: `ModelInference.generate(prompt=prompt, params={...})`
- **Output Cleaning**: Removes prompt artifacts (e.g., "Answer:", "Source: Context"), placeholder citations, meta-commentary paragraphs

##### Step 7: Answer Verification (Trust Layer)
- **Method**: **Natural Language Inference (NLI)** using the same LLM
- **Process**:
  1. **Claim Deconstruction**: 
     - Splits answer into sentences using regex: `re.split(r'[.;]\s+|\n+', answer)`
     - Filters for factual claims (sentences with quantitative data, findings, or length >50)
     - Excludes meta-commentary, questions, instructions
  2. **Claim Verification**:
     - For each claim, performs NLI task: "Does source support, refute, or not mention this claim?"
     - Prompts LLM with: `"Given the source text, does it support the following claim? Answer only 'Supports', 'Refutes', or 'Not Mentioned'"`
     - Checks claim against all retrieved source chunks
     - Returns best status: "Supports" (highest priority), "Refutes", or "Not Mentioned"
  3. **Answer Annotation**:
     - Matches verified claims to sentences in answer text
     - Adds visual badges: "Verified", "Refuted", or "Not Found"
     - One badge per claim (no duplicates) via matching score tracking
- **Output**: List of verification results with claim, status, and supporting chunk

#### Complex Query Path (Iterative Agentic Framework)

For complex multi-hop queries, the system uses an orchestrator-based iterative framework inspired by **i-MedRAG**:

##### Step 1: Query Complexity Detection
- **Method**: Heuristic-based detection
- **Criteria**:
  - Multiple question marks (≥2) OR
  - Multiple complex query indicators (≥2): "compare", "analyze", "synthesize", "evaluate", etc. OR
  - Long query (>50 chars) with ≥1 indicator
- **Output**: Boolean flag triggering orchestrator

##### Step 2: Query Decomposition & Routing
- **Method**: **LLM-based Query Decomposition** with routing classification
- **Model**: Granite-3-8B-Instruct, `temperature=0.2`
- **Process**:
  1. Prompts LLM to decompose query into sub-questions
  2. For each sub-question, classifies as **TEXT** (conceptual) or **TABLE** (quantitative)
  3. Returns JSON array: `[{"question": "...", "type": "TEXT|TABLE"}, ...]`
  4. Limits to 5 sub-questions max
- **Fallback**: If JSON parsing fails, uses simple decomposition (all TEXT)

##### Step 3: Iterative Retrieval Loop
- **Process**: For each sub-question:
  1. **TEXT queries**: Uses standard RAG pipeline (hybrid search → RRF → reranking → generation)
  2. **TABLE queries**: 
     - Retrieves relevant DataFrames from session state
     - LLM generates pandas code to answer question from tables
     - Executes code in sandboxed environment (`exec()` with restricted globals)
     - Returns stdout as answer
  3. Collects intermediate answers, sources, and source chunks
- **Status Updates**: Dynamic UI status text updates for each step

##### Step 4: Synthesis
- **Method**: **LLM-based Synthesis** from all collected evidence
- **Model**: Granite-3-8B-Instruct, `temperature=0.2`
- **Prompt Structure**:
  - Original query
  - Evidence from each sub-question (labeled by type: TEXT or TABLE)
  - Instructions: Synthesize comprehensive answer, include all details, avoid placeholder citations
- **Output**: Final synthesized answer integrating all evidence

##### Step 5: Verification (Same as Simple Path)
- Verifies final synthesized answer against all collected source chunks
- Displays verification badges and details

##### Step 6: Agent Trajectory Visualization
- **Purpose**: Exposes agent's "chain of thought" to user
- **Display**: Collapsible expander with step-by-step process:
  - Query Analysis (planning)
  - Query Decomposition (sub-questions with types)
  - Step-by-step retrieval and intermediate answers
  - Synthesis step
  - Verification step and results
  - Final answer
- **Storage**: `st.session_state["agent_trajectory"] = [{query, trajectory, answer}, ...]`

### Technical Stack

#### Core Libraries & Frameworks
- **Streamlit** (`>=1.37`): Web UI framework
- **Python** (`>=3.11`): Runtime environment
- **uv**: Fast Python package manager and resolver

#### IBM Cloud Services
- **watsonx.ai**: Foundation models API
  - **Embedding Model**: `sentence-transformers/all-minilm-l6-v2` (384 dim) or `ibm/granite-embedding-30m-english` (1024 dim)
  - **Generation Model**: `ibm/granite-3-8b-instruct`
  - **SDK**: `ibm-watsonx-ai>=1.1.8`
- **IBM Cloud Object Storage (COS)**: S3-compatible object storage
  - **SDK**: `ibm-cos-sdk>=2.13`
  - **Authentication**: IAM-based (recommended) or HMAC

#### Vector Search & Information Retrieval
- **FAISS** (`faiss-cpu>=1.7.4`): Facebook AI Similarity Search
  - **Index**: `IndexFlatIP` (exact search via inner product)
  - **Metric**: Cosine Similarity (via L2 normalization)
- **BM25** (`rank-bm25>=0.2.2`): Best Matching 25 algorithm
  - **Implementation**: BM25Okapi
- **NumPy** (`>=1.26`): Numerical operations for embeddings

#### Text Processing
- **PyPDF** (`>=4.2`): PDF text extraction
- **pdfplumber** (`>=0.11.0`): Enhanced PDF extraction with font size/weight analysis for metadata extraction
- **LangChain Text Splitters** (`>=0.2`): RecursiveCharacterTextSplitter for chunking
- **Camelot-py** (`>=0.11.0`): PDF table extraction
  - **Flavor**: `lattice` (for structured tables with clear boundaries)
  - **Note**: Additional system dependencies may be required (e.g., Ghostscript) depending on the backend

#### Data Processing
- **Pandas** (`>=2.2`): DataFrame manipulation for table queries

#### Document Export
- **python-docx** (`>=1.1`): DOCX export functionality for Synthesis Studio reports

#### Utilities
- **Pydantic** (`>=2.7`): Data validation and settings management
- **python-dotenv** (`>=1.0`): Environment variable management

### Session-Based Storage Architecture

All data storage is **session-based** using Streamlit session state, ensuring complete isolation between user sessions:

1. **Vector Store** (`st.session_state["faiss_store"]`):
   - `embeddings`: Raw float arrays
   - `metadata`: Chunk metadata (id, doc_id, page_num, chunk_index, text, source_uri)
   - `dim`: Embedding dimension

2. **BM25 Index**: Rebuilt from FAISS metadata on initialization (shared session key)

3. **Table Store** (`st.session_state["table_store"]`):
   - Structure: `{doc_id: [DataFrame1, DataFrame2, ...]}`

4. **Ingested Documents** (`st.session_state["ingested_docs"]`):
   - List of document IDs: `[doc_id1, doc_id2, ...]`

5. **Verification Results** (`st.session_state["verification_results"]`):
   - List of verification results per answer: `[{answer, verification, sources}, ...]`

6. **Agent Trajectory** (`st.session_state["agent_trajectory"]`):
   - List of trajectory data: `[{query, trajectory, answer}, ...]`

### Algorithm Details

#### Reciprocal Rank Fusion (RRF)
- **Purpose**: Combines ranked lists from different search methods (FAISS + BM25)
- **Formula**: `RRF_score(doc) = Σ(1 / (k + rank(doc, list_i)))` where `k=60`
- **Advantage**: Effective fusion method that doesn't require score normalization

#### Jaccard Similarity
- **Purpose**: Keyword overlap metric for reranking
- **Formula**: `J(A, B) = |A ∩ B| / |A ∪ B|`
- **Usage**: Measures word-level overlap between query and document

#### Cosine Similarity (via Inner Product)
- **Method**: L2 normalization followed by inner product
- **Formula**: `similarity = (A · B) / (||A|| × ||B||)` = `(A_normalized · B_normalized)`
- **Implementation**: FAISS `IndexFlatIP` with normalized vectors

#### BM25 Scoring
- **Formula**: `BM25(q, d) = Σ IDF(q_i) × (f(q_i, d) × (k1 + 1)) / (f(q_i, d) + k1 × (1 - b + b × |d| / avgdl))`
- **Parameters**: Standard BM25 parameters (k1=1.5, b=0.75) via `rank-bm25`
- **Purpose**: Term frequency-inverse document frequency based ranking

### Model Configuration

#### Embedding Models
- **Primary**: `sentence-transformers/all-minilm-l6-v2`
  - Dimensions: 384
  - Max tokens: 256
  - Use case: General semantic understanding
- **Alternative**: `ibm/granite-embedding-30m-english`
  - Dimensions: 1024
  - Use case: Higher-dimensional embeddings (if available)

#### Generation Model
- **Model**: `ibm/granite-3-8b-instruct`
- **Parameters**:
  - Temperature: 0.2 (default, configurable)
  - Max new tokens: 4096
  - Return options: Includes input text
- **Use cases**: Answer generation, query decomposition, synthesis, context compression, NLI verification

### Performance Optimizations

1. **Batch Embedding**: Processes chunks in batches via `embed_documents()` API
2. **Index Rebuilding**: FAISS index rebuilt from session state ensures consistency
3. **Token Limit Handling**: Automatic re-chunking prevents embedding failures
4. **Session Filtering**: `allowed_doc_ids` parameter filters searches to session-specific documents
5. **Hybrid Search**: Combines semantic and keyword search for comprehensive coverage
6. **RRF Fusion**: Efficient rank combination without score normalization
7. **Reranking**: Second-stage scoring improves precision of retrieved chunks
8. **Context Compression**: Reduces input length while preserving information

### Error Handling & Resilience

1. **Embedding Token Limits**: Automatic re-chunking and retry with smaller chunks
2. **JSON Parsing Failures**: Fallback to simple decomposition for query routing
3. **Reranking Failures**: Fallback to RRF results if reranking fails
4. **Verification Failures**: Continues without verification if NLI fails
5. **Orchestrator Failures**: Falls back to standard RAG if iterative framework fails
6. **Table Extraction Failures**: Graceful degradation (skips table extraction if camelot fails)
7. **API Failures**: Retry logic via `tenacity` library

This technical architecture ensures MedCortex provides accurate, verifiable, and comprehensive answers to medical research queries while maintaining high performance and reliability.
