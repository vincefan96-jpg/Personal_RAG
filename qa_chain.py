import os
import logging
from typing import List

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever

from retrieval import HybridRetriever, get_cross_encoder
from utils import singleton

logger = logging.getLogger("rag")


PROMPT_TEMPLATE = """你是一个专业的知识库助手，只根据以下提供的上下文回答问题。
如果上下文中没有相关信息，请说"我在知识库中未找到相关内容"，不要编造答案。

上下文：
{context}

问题：{question}

请给出清晰、准确的回答："""


@singleton("llm")
def get_llm() -> ChatTongyi:
    logger.info("初始化 ChatTongyi LLM (qwen-plus)")
    return ChatTongyi(
        model="qwen-plus",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        streaming=True,
    )


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
