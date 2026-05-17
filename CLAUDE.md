# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# CLI chat mode
python main.py

# Streamlit web UI
streamlit run app.py
```

No test suite or linting is configured.

## Architecture

RAG pipeline: document ingestion → text chunking → embedding → FAISS vector store → hybrid retrieval → LLM answer generation.

### Singleton pattern

`utils.py:singleton(key)` caches zero-arg factory results in a module-level registry. Used by `get_embeddings()` (`vectorstore.py`), `get_llm()` (`qa_chain.py`), and `get_cross_encoder()` (`retrievers.py`). Call `utils.reset_singletons(*keys)` to invalidate cached models (e.g., after rebuilding the vectorstore via Streamlit).

### Chunking strategy

`config.py` defines three tiers based on per-source total character count (`CHUNK_TIERS`). `ingest.py:_select_tier()` picks the first tier whose `max_chars >= total_chars`. The last tier uses `float("inf")` as catch-all. Each chunk records `chunk_tier` and `source_total_chars` in metadata.

### Hybrid retrieval + reranking

`retrievers.py:HybridRetriever` fetches `top_k * 2` candidates from both FAISS (dense) and BM25 (sparse), merges via RRF (Reciprocal Rank Fusion with `k=60`), takes the top `hybrid_k` results, then reranks with a CrossEncoder (`ms-marco-MiniLM-L-6-v2`) to produce `final_k` documents.

### API provider

Despite the env var name `OPENAI_API_KEY`, the project uses **Alibaba DashScope** in OpenAI-compatible mode (`BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`). LLM is `ChatTongyi(qwen-plus)`, embeddings are `DashScopeEmbeddings(text-embedding-v4)`.

### Streamlit app flow

`app.py` initializes models once via `warmup_models()` (pre-loads embeddings, cross-encoder, LLM to avoid lazy-init latency). Knowledge base init and QA chain are wrapped in `@st.cache_resource`. File upload merges new chunks into the existing FAISS store via `merge_from()`, then clears `st.cache_resource` and singletons.

### Retry decorator

`utils.retry(max_retries, base_delay, backoff)` applies exponential backoff to flaky API calls. Used on `CrossEncoderReranker._predict_batch` and `build_vectorstore`.
