import os
import logging
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ingest import load_documents, split_documents, load_files
from retrieval import build_vectorstore, load_vectorstore, get_embeddings, get_cross_encoder
from qa_chain import get_llm
from config import VECTORSTORE_PATH

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
