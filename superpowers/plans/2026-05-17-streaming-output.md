# Streaming Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real-time token-by-token LLM answer streaming to the Streamlit UI.

**Architecture:** Split retrieval from generation — add a `stream_qa()` function in `qa_chain.py` that retrieves documents then returns a token generator, and modify `handle_user_input()` in `app.py` to consume it. Existing `build_qa_chain()` and CLI remain untouched.

**Tech Stack:** Python, Streamlit, LangChain Community (ChatTongyi), FAISS

---

### Task 1: Add `stream_qa()` to `qa_chain.py`

**Files:**
- Modify: `qa_chain.py:38-73` (append after `build_qa_chain`)

- [ ] **Step 1: Add import for `ChatGenerationChunk` (needed for type check)**

At line 1-11 in `qa_chain.py`, add `ChatGenerationChunk` to the existing `langchain_core` imports. The current imports are:

```python
import os
import logging
from typing import List

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from retrievers import HybridRetriever, get_cross_encoder
from utils import singleton
```

Change the `langchain_core` imports line to:

```python
from langchain_core.documents import Document
from langchain_core.outputs import ChatGenerationChunk
```

- [ ] **Step 2: Add `stream_qa()` function after `build_qa_chain()` (after line 72)**

Append this function to `qa_chain.py`:

```python
def stream_qa(query: str, vectorstore: FAISS, chunks: List[Document], top_k: int = 4):
    llm = get_llm()

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k * 2},
    )

    from langchain_community.retrievers import BM25Retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k * 2

    reranker = get_cross_encoder()

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        hybrid_k=top_k * 3,
        final_k=top_k,
    )

    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt_text = PROMPT_TEMPLATE.format(context=context, question=query)

    def generate():
        for chunk in llm.stream(prompt_text):
            if isinstance(chunk, ChatGenerationChunk):
                yield chunk.content
            elif hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)

    return generate(), docs
```

- [ ] **Step 3: Commit**

```bash
git add qa_chain.py
git commit -m "feat: add stream_qa() for token-by-token LLM generation"
```

---

### Task 2: Modify `app.py` for streaming

**Files:**
- Modify: `app.py:45-49` (initialize_session_state)
- Modify: `app.py:131-171` (handle_user_input)
- Modify: `app.py:267` (call to handle_user_input)
- Modify: `app.py:4` (add import for stream_qa)

- [ ] **Step 1: Add import for `stream_qa`**

At line 9 in `app.py`, change:

```python
from qa_chain import build_qa_chain, get_llm
```

to:

```python
from qa_chain import build_qa_chain, stream_qa, get_llm
```

- [ ] **Step 2: Store `vectorstore` and `chunks` in session_state**

In `initialize_session_state()` (lines 45-49), add two new session_state keys:

Current:
```python
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
```

Change to:
```python
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
```

- [ ] **Step 3: Populate session_state after knowledge base init**

After line 267 where `vectorstore, chunks = result` is set, add session_state writes. This block (lines 265-267):

```python
    vectorstore, chunks = result

    chain = create_qa_chain(vectorstore, chunks)
```

Change to:
```python
    vectorstore, chunks = result

    st.session_state.vectorstore = vectorstore
    st.session_state.chunks = chunks

    chain = create_qa_chain(vectorstore, chunks)
```

- [ ] **Step 4: Replace `handle_user_input` body with streaming version**

Replace the entire `handle_user_input` function (lines 131-171):

Current function:
```python
def handle_user_input(chain):
    if user_input := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考..."):
                try:
                    result = chain.invoke({"query": user_input})
                    answer = result["result"]
                    sources = list(set(
                        doc.metadata.get("source", "未知")
                        for doc in result["source_documents"]
                    ))

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 查看来源文档"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**来源 {i}:** {source}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    logger.exception("问答处理失败")
                    error_msg = f"❌ 发生错误: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
```

Replace with:
```python
def handle_user_input():
    if user_input := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 正在检索..."):
                try:
                    token_gen, source_docs = stream_qa(
                        user_input,
                        st.session_state.vectorstore,
                        st.session_state.chunks,
                    )
                except Exception as e:
                    logger.exception("检索失败")
                    error_msg = f"❌ 检索失败: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
                    return

            # Streaming phase
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                for token in token_gen:
                    full_answer += token
                    answer_placeholder.markdown(full_answer + "▌")
            except Exception as e:
                logger.exception("流式生成中断")
                full_answer += f"\n\n❌ 生成中断: {str(e)}"
            finally:
                answer_placeholder.markdown(full_answer)

            sources = list(set(
                doc.metadata.get("source", "未知") for doc in source_docs
            ))

            if sources:
                with st.expander("📎 查看来源文档"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**来源 {i}:** {source}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
            })
```

- [ ] **Step 5: Update the call site**

At line 273, change:

```python
    handle_user_input(chain)
```

to:

```python
    handle_user_input()
```

- [ ] **Step 6: Also update `add_files_to_knowledge_base` to refresh session_state**

In `add_files_to_knowledge_base()` after a successful merge (around line 110), the session_state vectorstore and chunks need updating. After the `st.cache_resource.clear()` line 103 within the merge path, add:

```python
                # Refresh session_state with updated vectorstore
                new_vs, new_chunks = load_vectorstore()
                st.session_state.vectorstore = new_vs
                st.session_state.chunks = new_chunks
```

And after the `st.cache_resource.clear()` on line 109 in the first-build path, add:

```python
                new_vs, new_chunks = load_vectorstore()
                st.session_state.vectorstore = new_vs
                st.session_state.chunks = new_chunks
```

- [ ] **Step 7: Remove unused import**

The `build_qa_chain` import in `app.py` (line 9) is still used by `create_qa_chain()`. Keep it. No change needed.

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "feat: add streaming answer output to Streamlit UI"
```

---

### Task 3: Manual verification

- [ ] **Step 1: Start the Streamlit app**

```bash
streamlit run app.py
```

- [ ] **Step 2: Verify streaming behavior**

1. Wait for the app to load and warm up models
2. Type a question in the chat input
3. **Verify:** "正在检索..." spinner appears briefly
4. **Verify:** Answer text appears token by token with a `▌` cursor
5. **Verify:** After completion, cursor disappears
6. **Verify:** "📎 查看来源文档" expander shows sources

- [ ] **Step 3: Verify error handling**

1. Temporarily change `OPENAI_API_KEY` to an invalid value and restart
2. **Verify:** Error message displays cleanly in chat
3. Restore the correct API key

- [ ] **Step 4: Verify CLI is unaffected**

```bash
python main.py
```

1. **Verify:** CLI still works with `chain.invoke()` — no streaming, full answer printed at once
