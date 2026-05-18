# 项目结构重构设计

## 目标

中等程度重构：消除重复代码、拆分 `app.py`、引入 `retrieval/` 子包内聚检索模块。

## 新文件结构

```
RAG/
├── main.py              CLI 入口
├── app.py               Streamlit UI (精简到 ~130 行，纯视图)
├── config.py            配置常量 (不变)
├── utils.py             工具函数 (不变)
├── knowledge_base.py    [新] 知识库管理 — 被 CLI 和 Streamlit 共用
├── ingest.py            文档加载 + 分块
├── qa_chain.py          QA 链 + 流式 (消除重复)
└── retrieval/           [新] 检索子包
    ├── __init__.py       对外导出
    ├── retrievers.py     HybridRetriever + CrossEncoderReranker
    └── vectorstore.py    向量库构建/加载
```

## 四个核心改动

### 1. 消除 loader 分发重复

`ingest.py` 新增 `_get_loader()` 和 `load_files()`：

- `_get_loader(fpath)` — 根据扩展名返回对应 LangChain loader，统一所有 loader 分发逻辑
- `load_files(file_paths)` — 加载指定文件列表（而非整个目录），供上传流程使用
- `load_documents(path)` 内部改用 `_get_loader()`

`app.py` (现在在 `knowledge_base.py`) 的上传流程调用 `load_files()`，不再内联重复的分发代码。

### 2. 消除检索器构建重复

`qa_chain.py` 提取 `_build_hybrid_retriever(vectorstore, chunks, top_k=4)`：

- 构建 FAISS retriever + BM25 retriever + CrossEncoder reranker → `HybridRetriever`
- `build_qa_chain()` 和 `stream_qa()` 都调用此函数
- 两个查询路径的检索器构造逻辑不再重复

### 3. 新建 `knowledge_base.py`

从 `app.py` 抽出三个函数，供 CLI 和 Streamlit 共用：

- `warmup_models()` — 预加载 embeddings / cross-encoder / LLM
- `init_knowledge_base(docs_path)` — 统一 `main.py` 和 `app.py` 的两个版本，返回 `(vectorstore, chunks)` 或 `None`
- `add_files_to_knowledge_base(uploaded_files)` — 文件上传处理，内部调用 `ingest.load_files()` + `split_documents()`

### 4. 新建 `retrieval/` 子包

- `retrievers.py` 和 `vectorstore.py` 移入 `retrieval/`
- `vectorstore.py` 删除重复的 `VECTORSTORE_PATH` 常量，改为从 `config` 导入
- `__init__.py` 导出所有对外接口：`HybridRetriever`, `CrossEncoderReranker`, `get_cross_encoder`, `get_embeddings`, `build_vectorstore`, `load_vectorstore`

## 模块依赖

```
config.py  ←──── 被所有模块导入（常量唯一来源）
utils.py   ←──── 被 retrieval/, knowledge_base 导入

retrieval/
├── vectorstore.py  ← config, utils
├── retrievers.py   ← utils
└── __init__.py     导出所有公开接口

ingest.py           ← config
qa_chain.py         ← retrieval
knowledge_base.py   ← ingest, retrieval, qa_chain, config, utils

main.py             ← knowledge_base, qa_chain, config
app.py              ← knowledge_base, qa_chain, config, utils
```

## 效果

| 指标 | 重构前 | 重构后 |
|------|--------|--------|
| `app.py` | 299 行 | ~130 行 |
| 重复代码 | 3 处 | 0 |
| 重复常量 | `VECTORSTORE_PATH` ×2 | 1 处 |
| 检索模块 | 散落根目录 | 内聚于 `retrieval/` |
