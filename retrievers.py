import logging
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from utils import retry

logger = logging.getLogger("rag")

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder: Optional["CrossEncoderReranker"] = None


def get_cross_encoder() -> "CrossEncoderReranker":
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("加载 CrossEncoder 模型: %s", CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoderReranker()
    return _cross_encoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @retry(max_retries=2, base_delay=0.5, backoff=2.0)
    def _predict_batch(self, pairs: List[List[str]]) -> List[float]:
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]
        return scores

    def predict(self, pairs: List[List[str]]) -> List[float]:
        return self._predict_batch(pairs)

    def rerank(self, query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
        if not documents:
            return documents

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.predict(pairs)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored[:top_k]]


def _make_dedup_key(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    return f"{source}::{doc.page_content}"


class HybridRetriever(BaseRetriever):
    vector_retriever: BaseRetriever = None
    bm25_retriever: BM25Retriever = None
    reranker: CrossEncoderReranker = None
    hybrid_k: int = 10
    final_k: int = 4

    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def _rrf_merge(vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        scores: dict[str, float] = {}
        for rank, doc in enumerate(vector_docs):
            key = _make_dedup_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        for rank, doc in enumerate(bm25_docs):
            key = _make_dedup_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

        seen: set[str] = set()
        merged: list[Document] = []
        for doc in vector_docs + bm25_docs:
            key = _make_dedup_key(doc)
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        merged.sort(key=lambda d: scores.get(_make_dedup_key(d), 0.0), reverse=True)
        return merged

    def _get_relevant_documents(self, query: str) -> List[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        merged = self._rrf_merge(vector_docs, bm25_docs)
        top_hybrid = merged[: self.hybrid_k]

        reranked = self.reranker.rerank(query, top_hybrid, top_k=self.final_k)
        return reranked
