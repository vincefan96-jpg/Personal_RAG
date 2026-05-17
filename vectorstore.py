import os
import logging
from typing import List, Tuple

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils import singleton, retry

logger = logging.getLogger("rag")

VECTORSTORE_PATH = "vectorstore"


@singleton("embeddings")
def get_embeddings() -> DashScopeEmbeddings:
    logger.info("初始化 DashScope embeddings (text-embedding-v4)")
    return DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=os.getenv("OPENAI_API_KEY"),
    )


@retry(max_retries=3, base_delay=1.0, backoff=2.0)
def build_vectorstore(chunks: List[Document]) -> Tuple[FAISS, List[Document]]:
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    logger.info("向量库已保存，共 %d 个片段", len(chunks))
    return vectorstore, chunks


def load_vectorstore() -> Tuple[FAISS, List[Document]]:
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    docs = list(vectorstore.docstore._dict.values())
    logger.info("已加载向量库，共 %d 个文档片段", len(docs))
    return vectorstore, docs
