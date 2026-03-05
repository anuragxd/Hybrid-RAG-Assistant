# 🧠 Hybrid Multi-Stage RAG Research Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![Groq](https://img.shields.io/badge/LLM-Groq-black)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-green)

A modular, production-grade Information Retrieval and Retrieval-Augmented Generation (RAG) system. This assistant takes a user-provided research topic, dynamically crawls the web, constructs a local Knowledge Base, and answers complex queries using a multi-stage hybrid retrieval pipeline (Lexical + Dense + Neural Reranking).

## 🌟 Key Features

* **Dynamic Web Ingestion:** Uses the Tavily API to gather sources and `trafilatura` for robust HTML boilerplate removal.
* **Semantic Chunking:** Context-aware hierarchical chunking using LangChain to preserve document coherence.
* **Dual-Index Hybrid Retrieval:**
  * **Lexical (BM25):** Captures exact keyword matches using an inverted index.
  * **Dense (FAISS HNSW):** Captures semantic meaning using local Hugging Face embeddings (`all-MiniLM-L6-v2`), with tuned `m` and `efConstruction` parameters.
* **Reciprocal Rank Fusion (RRF):** Mathematically merges BM25 and HNSW results to prevent score-scale dominance.
* **Neural Reranking:** Passes the fused candidate pool through a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for highly precise final document selection.
* **Lightning Fast LLM Synthesis:** Powered by Groq's LPU architecture for sub-second query rewriting and final answer generation with explicit URL citations.
* **Automated Evaluation:** Includes a standalone script to synthetically generate Q&A test sets and evaluate the pipeline using the **RAGAS** framework (Faithfulness, Relevancy, Precision, Recall).

---

## 🏗️ System Architecture

1. **Query Expansion:** The user's query is rewritten by an LLM into 3 distinct search queries to maximize recall.
2. **Parallel Retrieval:** Queries are sent simultaneously to the BM25 Index and the FAISS HNSW Index.
3. **RRF Fusion:** Top results (top_k=30) from both indexes are merged using the RRF formula: `score = 1 / (k + rank)`.
4. **Cross-Encoder Reranking:** The top 20 fused candidates are evaluated jointly with the original query to produce the final Top 5 contexts.
5. **Generation:** The LLM synthesizes the Top 5 chunks into a cited, factual response.

---

## 📂 Project Structure

```text
research_assistant/
├── app.py                  # Streamlit UI (Entry Point)
├── generate_testset.py     # Script to synthetically generate QA pairs
├── run_evaluation.py       # Script to evaluate the pipeline via RAGAS
├── config.yaml             # Hyperparameters & Model Selection
├── requirements.txt        # Dependencies
│
├── core/                   # The engine of the assistant
│   ├── ingestion.py        # Tavily API client
│   ├── preprocessing.py    # Trafilatura HTML cleaning
│   ├── chunking.py         # Semantic splitter
│   ├── embeddings.py       # Local HuggingFace Embeddings
│   └── llm.py              # Groq client integration
│
├── indexing/               # Knowledge Base management
│   ├── bm25_index.py       # BM25 implementation + Pickling
│   ├── hnsw_index.py       # FAISS HNSW implementation
│   └── manager.py          # Orchestrates dual-indexing and disk persistence
│
├── retrieval/              # Multi-stage logic
│   ├── query_rewriter.py   # LLM query expansion
│   ├── hybrid_retriever.py # Parallel execution logic
│   ├── rrf_fusion.py       # Reciprocal Rank Fusion
│   └── reranker.py         # Cross-encoder scoring
│
├── utils/                  # Helpers
│   ├── logger.py           # Structured logging
│   └── schemas.py          # Pydantic models (Chunks, Metadata)
