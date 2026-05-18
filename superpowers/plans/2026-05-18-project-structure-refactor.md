# 项目结构重构 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除重复代码、拆分 app.py、引入 retrieval/ 子包，不改变任何业务行为。

**Architecture:** 7 个独立任务，按依赖顺序执行。每个任务都是自包含的 — 先改代码，再验证导入正确性，最后提交。纯结构重构，无测试套件，验证方式为导入检查 + 运行应用。

**Tech Stack:** Python, LangChain, FAISS, Streamlit

---

### Task 1: 重构 ingest.py — 消除 loader 分发重复

**Files:**
- Modify: `ingest.py`

- [ ] **Step 1: 新增 `_get_loader()` 辅助函数**

在 `ingest.py` 的 import 块之后、`load_documents()` 之前插入：

```python
def _get_loader(fpath: str):
    """根据文件扩展名返回对应 LangChain loader，不支持则返回 None。"""
    if fpath.endswith(".pdf"):
        return PyPDFLoader(fpath)
    elif fpath.endswith(".docx"):
        return Docx2txtLoader(fpath)
    elif fpath.endswith(".txt") or fpath.endswith(".md"):
        return TextLoader(fpath, encoding="utf-8")
    return None
```

- [ ] **Step 2: 重构 `load_documents()` 使用 `_get_loader()`**

将现有 `load_documents()` 替换为：

```python
def load_documents(path: str) -> List[Document]:
    docs = []
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        loader = _get_loader(fpath)
        if loader:
            docs.extend(loader.load())
    return docs
```

- [ ] **Step 3: 新增 `load_files()` 函数**

在 `load_documents()` 之后追加：

```python
def load_files(file_paths: List[str]) -> List[Document]:
    """加载指定文件路径列表（用于上传流程，而非扫描目录）。"""
    docs = []
    for fpath in file_paths:
        loader = _get_loader(fpath)
        if loader:
            docs.extend(loader.load())
    return docs
```

- [ ] **Step 4: 验证导入**

```bash
python -c "from ingest import load_documents, split_documents, load_files, _get_loader; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add ingest.py
git commit -m "refactor: extract _get_loader/load_files from ingest.py to eliminate loader dispatch duplication"
```

---

### Task 2: 创建 retrieval/ 子包

**Files:**
- Create: `retrieval/__init__.py`
- Move: `vectorstore.py` → `retrieval/vectorstore.py`
- Move: `retrievers.py` → `retrieval/retrievers.py`
- Modify: `retrieval/vectorstore.py` (remove duplicate VECTORSTORE_PATH)
- Modify: `app.py` (更新 imports)
- Modify: `main.py` (更新 imports)
- Modify: `qa_chain.py` (更新 imports)
- Delete: `vectorstore.py` (root)
- Delete: `retrievers.py` (root)

- [ ] **Step 1: 创建 retrieval/ 目录**

```bash
mkdir retrieval
```

- [ ] **Step 2: 移动文件**

```bash
git mv vectorstore.py retrieval/vectorstore.py
git mv retrievers.py retrieval/retrievers.py
```

- [ ] **Step 3: 修复 retrieval/vectorstore.py 的重复常量**

`retrieval/vectorstore.py` 第 13 行有 `VECTORSTORE_PATH = "vectorstore"`，删除该行，在 imports 中加入：

```python
from config import VECTORSTORE_PATH
```

最终 `retrieval/vectorstore.py` 的 import 部分为：

```python
import os
import logging
from typing import List, Tuple

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import VECTORSTORE_PATH
from utils import singleton, retry
```

（其余函数不变）

- [ ] **Step 4: 创建 retrieval/__init__.py**

```python
from retrieval.vectorstore import get_embeddings, build_vectorstore, load_vectorstore
from retrieval.retrievers import HybridRetriever, CrossEncoderReranker, get_cross_encoder

__all__ = [
    "get_embeddings",
    "build_vectorstore",
    "load_vectorstore",
    "HybridRetriever",
    "CrossEncoderReranker",
    "get_cross_encoder",
]
```

- [ ] **Step 5: 更新 qa_chain.py 的 imports**

`qa_chain.py` 第 11 行：

```python
from retrievers import HybridRetriever, get_cross_encoder
```

改为：

```python
from retrieval import HybridRetriever, get_cross_encoder
```

- [ ] **Step 6: 更新 app.py 的 imports**

`app.py` 第 7-9 行：

```python
from vectorstore import build_vectorstore, load_vectorstore, VECTORSTORE_PATH, get_embeddings
from qa_chain import stream_qa, get_llm
from retrievers import get_cross_encoder
```

改为：

```python
from retrieval import build_vectorstore, load_vectorstore, get_embeddings, get_cross_encoder
from qa_chain import stream_qa, get_llm
from config import VECTORSTORE_PATH
```

- [ ] **Step 7: 更新 main.py 的 imports**

`main.py` 第 5 行：

```python
from vectorstore import build_vectorstore, load_vectorstore, VECTORSTORE_PATH
```

改为：

```python
from retrieval import build_vectorstore, load_vectorstore
from config import VECTORSTORE_PATH
```

- [ ] **Step 8: 验证所有导入**

```bash
python -c "from retrieval import HybridRetriever, get_cross_encoder, get_embeddings, build_vectorstore, load_vectorstore; print('retrieval OK')"
python -c "from qa_chain import build_qa_chain, stream_qa, get_llm; print('qa_chain OK')"
python -c "from config import VECTORSTORE_PATH; print('config OK')"
```

Expected: 三行 OK

- [ ] **Step 9: Commit**

```bash
git add retrieval/ app.py main.py qa_chain.py
git commit -m "refactor: move retrievers and vectorstore into retrieval/ subpackage"
```

---

### Task 3: 消除 qa_chain.py 检索器构建重复

**Files:**
- Modify: `qa_chain.py`

- [ ] **Step 1: 将 BM25Retriever 导入移到文件顶部**

在 `qa_chain.py` 顶部 import 区域加入：

```python
from langchain_community.retrievers import BM25Retriever
```

同时删除 `build_qa_chain()` 和 `stream_qa()` 各自内部的 `from langchain_community.retrievers import BM25Retriever` 局部导入。

- [ ] **Step 2: 新增 `_build_hybrid_retriever()` 函数**

在 `get_llm()` 之后、`build_qa_chain()` 之前插入：

```python
def _build_hybrid_retriever(vectorstore: FAISS, chunks: List[Document], top_k: int = 4) -> HybridRetriever:
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k * 2},
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k * 2
    reranker = get_cross_encoder()
    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        hybrid_k=top_k * 3,
        final_k=top_k,
    )
```

- [ ] **Step 3: 精简 `build_qa_chain()`**

将现有 `build_qa_chain()` 替换为：

```python
def build_qa_chain(vectorstore: FAISS, chunks: List[Document], top_k: int = 4):
    llm = get_llm()
    retriever = _build_hybrid_retriever(vectorstore, chunks, top_k)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain
```

- [ ] **Step 4: 精简 `stream_qa()`**

将现有 `stream_qa()` 替换为：

```python
def stream_qa(query: str, vectorstore: FAISS, chunks: List[Document], top_k: int = 4):
    llm = get_llm()
    retriever = _build_hybrid_retriever(vectorstore, chunks, top_k)

    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt_text = PROMPT_TEMPLATE.format(context=context, question=query)

    def generate():
        for chunk in llm.stream(prompt_text):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)

    return generate(), docs
```

- [ ] **Step 5: 验证导入**

```bash
python -c "from qa_chain import build_qa_chain, stream_qa, get_llm, _build_hybrid_retriever; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add qa_chain.py
git commit -m "refactor: extract _build_hybrid_retriever to eliminate duplication in qa_chain.py"
```

---

### Task 4: 新建 knowledge_base.py

**Files:**
- Create: `knowledge_base.py`

- [ ] **Step 1: 创建 knowledge_base.py**

写入完整文件：

```python
import os
import logging
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ingest import load_documents, split_documents, load_files
from retrieval import build_vectorstore, load_vectorstore, get_embeddings, get_cross_encoder
from qa_chain import get_llm
from config import VECTORSTORE_PATH
from utils import reset_singletons

logger = logging.getLogger("rag")


def warmup_models():
    """Pre-load all singletons to avoid lazy-init latency on first request."""
    try:
        logger.info("预加载模型...")
        get_embeddings()
        get_cross_encoder()
        get_llm()
        logger.info("模型预加载完成")
    except Exception as e:
        logger.warning("模型预加载部分失败 (将延迟初始化): %s", e)


def init_knowledge_base(docs_path: str) -> Optional[Tuple[FAISS, List[Document]]]:
    """Initialize knowledge base. Returns (vectorstore, chunks) or None on failure."""
    if not os.path.exists(VECTORSTORE_PATH):
        docs = load_documents(docs_path)
        chunks = split_documents(docs)

        if not chunks:
            return None

        vectorstore, chunks = build_vectorstore(chunks)
        return vectorstore, chunks
    else:
        return load_vectorstore()


def add_files_to_knowledge_base(uploaded_files) -> dict:
    """Add uploaded files to the knowledge base.

    Returns:
        {"success": bool, "files": int, "chunks": int, "error": str|None}
    """
    if not uploaded_files:
        return {"success": False, "files": 0, "chunks": 0, "error": "no files"}

    docs_dir = "./docs"
    os.makedirs(docs_dir, exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(docs_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)

    docs = load_files(saved_files)
    if not docs:
        return {"success": False, "files": len(saved_files), "chunks": 0, "error": "无法解析上传的文件"}

    chunks = split_documents(docs)
    if not chunks:
        return {"success": False, "files": len(saved_files), "chunks": 0, "error": "文档片段为空"}

    if os.path.exists(VECTORSTORE_PATH):
        existing_vectorstore, _ = load_vectorstore()
        new_vectorstore, _ = build_vectorstore(chunks)
        existing_vectorstore.merge_from(new_vectorstore)
        existing_vectorstore.save_local(VECTORSTORE_PATH)
    else:
        build_vectorstore(chunks)

    return {"success": True, "files": len(saved_files), "chunks": len(chunks), "error": None}
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from knowledge_base import warmup_models, init_knowledge_base, add_files_to_knowledge_base; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add knowledge_base.py
git commit -m "feat: add knowledge_base.py with shared init/add_files/warmup logic"
```

---

### Task 5: 精简 app.py（纯视图层）

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 更新 imports**

`app.py` 当前 imports（第 1-10 行）替换为：

```python
import os
import shutil
import logging

import streamlit as st
from knowledge_base import warmup_models, init_knowledge_base, add_files_to_knowledge_base
from qa_chain import stream_qa
from retrieval import load_vectorstore
from config import VECTORSTORE_PATH
from utils import reset_singletons
```

- [ ] **Step 2: 删除已迁移到 knowledge_base.py 的函数**

删除以下三个函数（原位置）：
- `init_knowledge_base()` — 第 26-37 行的 `@st.cache_resource` 装饰的函数
- `add_files_to_knowledge_base()` — 第 49-123 行的函数
- `warmup_models()` — 第 194-203 行的函数

- [ ] **Step 3: 新增带缓存的 init 包装函数**

在原 `init_knowledge_base` 位置，添加：

```python
@st.cache_resource
def _cached_init_kb(docs_path: str):
    return init_knowledge_base(docs_path)
```

- [ ] **Step 4: 重写 `add_files_to_knowledge_base` 调用点**

将 `handle_user_input` 附近的文件上传处理（原第 241-246 行附近）替换为使用新 API：

```python
if st.button("📤 添加到知识库", use_container_width=True, type="primary"):
    with st.spinner("⏳ 正在处理文档..."):
        result = add_files_to_knowledge_base(uploaded_files)
    if result["success"]:
        st.cache_resource.clear()
        st.session_state.upload_success = True
        st.success(f"✅ 成功添加 {result['files']} 个文件，新增 {result['chunks']} 个片段！")

        new_vs, new_chunks = load_vectorstore()
        st.session_state.vectorstore = new_vs
        st.session_state.chunks = new_chunks

        st.rerun()
    else:
        st.error(f"❌ {result['error']}")
```

- [ ] **Step 5: 更新 warmup_models 调用点**

`main()` 中第 210-213 行，`warmup_models` 现在从 `knowledge_base` 导入，调用方式不变：

```python
if "models_warmed" not in st.session_state:
    with st.spinner("🔥 正在初始化模型 (首次加载较慢)..."):
        warmup_models()
    st.session_state.models_warmed = True
```

- [ ] **Step 6: 更新 init_knowledge_base 调用点**

`main()` 中第 273-281 行，将 `init_knowledge_base(docs_path)` 改为 `_cached_init_kb(docs_path)`：

```python
if not os.path.exists(VECTORSTORE_PATH):
    with st.spinner("🔢 正在构建向量库，请稍候..."):
        result = _cached_init_kb(docs_path)
    if result is not None:
        _, chunks = result
        st.success(f"✅ 向量库构建成功！共 {len(chunks)} 个片段")
else:
    with st.spinner("📦 加载已有向量库..."):
        result = _cached_init_kb(docs_path)
```

- [ ] **Step 7: 验证导入**

```bash
python -c "import app; print('OK')"
```

Expected: `OK`（Streamlit 脚本语法正确，不会因为 import 而启动服务器）

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "refactor: slim app.py to view-only by delegating to knowledge_base.py"
```

---

### Task 6: 精简 main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 1: 更新 imports**

`main.py` 第 4-6 行替换为：

```python
from knowledge_base import init_knowledge_base
from qa_chain import build_qa_chain
from config import VECTORSTORE_PATH
```

- [ ] **Step 2: 删除本地 `init_knowledge_base()` 函数**

删除 `main.py` 第 15-28 行的 `init_knowledge_base()` 函数定义。

- [ ] **Step 3: 更新调用点**

`main.py` 第 53-57 行的 `__main__` 块，将 `init_knowledge_base("./docs")` 改为使用导入的版本（签名相同，调用方式不变）：

```python
if __name__ == "__main__":
    if not os.path.exists(VECTORSTORE_PATH):
        vectorstore, chunks = init_knowledge_base("./docs")
    else:
        logger.info("加载已有向量库...")
        vectorstore, chunks = load_vectorstore()
```

注意：需要额外导入 `load_vectorstore`：

```python
from retrieval import load_vectorstore
```

最终 `main.py` 的完整内容为：

```python
import os
import logging

from knowledge_base import init_knowledge_base
from retrieval import load_vectorstore
from qa_chain import build_qa_chain
from config import VECTORSTORE_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag")


def chat(chain):
    print("\n💬 知识库已就绪，输入 quit 退出\n")
    while True:
        query = input("你：").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        try:
            result = chain.invoke({"query": query})
            print(f"\n助手：{result['result']}")

            sources = {doc.metadata.get("source", "未知")
                       for doc in result["source_documents"]}
            print(f"📎 来源：{', '.join(sources)}\n")
        except Exception as e:
            logger.exception("问答失败")
            print(f"❌ 错误: {e}\n")


if __name__ == "__main__":
    if not os.path.exists(VECTORSTORE_PATH):
        vectorstore, chunks = init_knowledge_base("./docs")
    else:
        logger.info("加载已有向量库...")
        vectorstore, chunks = load_vectorstore()

    chain = build_qa_chain(vectorstore, chunks)
    chat(chain)
```

- [ ] **Step 4: 验证导入**

```bash
python -c "import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "refactor: simplify main.py by using shared knowledge_base.init_knowledge_base"
```

---

### Task 7: 端到端验证

**Files:** 无修改，仅验证

- [ ] **Step 1: 全面导入检查**

```bash
python -c "
from config import VECTORSTORE_PATH
from utils import retry, singleton, reset_singletons
from ingest import load_documents, split_documents, load_files, _get_loader
from retrieval import HybridRetriever, CrossEncoderReranker, get_cross_encoder, get_embeddings, build_vectorstore, load_vectorstore
from qa_chain import build_qa_chain, stream_qa, get_llm, _build_hybrid_retriever
from knowledge_base import warmup_models, init_knowledge_base, add_files_to_knowledge_base
import main
import app
print('ALL IMPORTS OK')
"
```

Expected: `ALL IMPORTS OK`

- [ ] **Step 2: 检查根目录无冗余文件**

```bash
ls *.py
```

Expected: `app.py  config.py  ingest.py  knowledge_base.py  main.py  qa_chain.py  utils.py`（无 `vectorstore.py` 和 `retrievers.py`）

- [ ] **Step 3: 确认 retrieval/ 子包结构**

```bash
ls retrieval/
```

Expected: `__init__.py  retrievers.py  vectorstore.py`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: final verification after structure refactor"
```
