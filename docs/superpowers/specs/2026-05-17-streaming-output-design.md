# Streaming Output for Streamlit UI

**Date:** 2026-05-17
**Status:** approved

## Goal

Add real-time token-by-token answer streaming to the Streamlit web interface. CLI (`main.py`) remains unchanged.

## Current State

Both CLI and Streamlit use `chain.invoke()` — a blocking call that waits for the full LLM response before displaying anything. The user sees a spinner the entire time.

## Target Data Flow

```
User input
  ↓
Show user message
  ↓
st.spinner("正在检索...")
  ↓
HybridRetriever.get_relevant_documents(query)  → List[Document]
  ↓
Format PROMPT_TEMPLATE with context + question
  ↓
llm.stream(prompt)  → token iterator
  ↓
Hide spinner, stream tokens into st.empty() container
  ↓
Show sources in expander
```

## Changes

### `qa_chain.py` — new `stream_qa()` function

```python
def stream_qa(query, vectorstore, chunks, top_k=4):
    # Build retriever (same params as build_qa_chain)
    # Retrieve documents
    # Format prompt
    # Return (generator, source_docs)
```

- Builds retriever internally (vector + BM25 + CrossEncoder rerank)
- Formats PROMPT_TEMPLATE with retrieved context
- Returns a tuple: (token generator, source documents)
- Existing `build_qa_chain()` kept for CLI backward compatibility

### `app.py` — modify `handle_user_input()`

- Store `vectorstore` and `chunks` in `st.session_state` so the handler can access them
- Replace `chain.invoke()` with:
  1. `spinner` wrapping only the retrieval call
  2. `st.empty()` placeholder for streaming tokens
  3. Iterate token generator, updating placeholder with `full_answer + "▌"` cursor
  4. Final render without cursor
  5. Sources in expander (unchanged behavior)

### Files NOT changed

- `retrievers.py` — retrieval logic unchanged
- `vectorstore.py` — vector store unchanged
- `main.py` — CLI stays non-streaming
- `config.py`, `ingest.py`, `utils.py` — no impact

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Retrieval fails | Show `st.error`, log to session |
| Streaming fails mid-generation | Keep partial output visible, append error message, show sources (retrieval already done) |
| Empty knowledge base | Prompt handles it — LLM says "未找到相关内容" via streaming |

## Boundary Cases

- **Rapid successive questions:** `st.chat_input` is synchronous, no concurrency issue
- **Model warmup:** unchanged — `warmup_models()` still pre-loads LLM singleton
- **Upload-then-ask flow:** `st.cache_resource` clear + singleton reset still works; new handler picks up fresh `vectorstore` from session_state

## Scope

Streamlit only. Single-session implementation. No test suite changes (none exists).
