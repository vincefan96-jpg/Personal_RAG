import os
import logging
from collections import defaultdict
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

logger = logging.getLogger("rag")


def load_documents(path: str) -> List[Document]:
    docs = []
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        if fname.endswith(".pdf"):
            docs += PyPDFLoader(fpath).load()
        elif fname.endswith(".docx"):
            docs += Docx2txtLoader(fpath).load()
        elif fname.endswith(".txt") or fname.endswith(".md"):
            docs += TextLoader(fpath, encoding="utf-8").load()
    return docs


def _select_tier(total_chars: int) -> dict:
    for tier_name, tier_cfg in config.CHUNK_TIERS.items():
        if total_chars <= tier_cfg["max_chars"]:
            return {**tier_cfg, "name": tier_name}

    last_name = list(config.CHUNK_TIERS.keys())[-1]
    last_cfg = config.CHUNK_TIERS[last_name]
    return {**last_cfg, "name": last_name}


def split_documents(docs: List[Document]) -> List[Document]:
    if not docs:
        return []

    source_docs: dict[str, List[Document]] = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        source_docs[source].append(doc)

    separators = config.CHUNK_SEPARATORS

    all_chunks: List[Document] = []

    for source, src_docs in source_docs.items():
        total_chars = sum(len(d.page_content) for d in src_docs)
        tier = _select_tier(total_chars)

        logger.info(
            "Chunking '%s': total_chars=%d -> tier=%s (chunk_size=%d, overlap=%d)",
            source,
            total_chars,
            tier["name"],
            tier["chunk_size"],
            tier["chunk_overlap"],
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=tier["chunk_size"],
            chunk_overlap=tier["chunk_overlap"],
            separators=separators,
        )

        chunks = splitter.split_documents(src_docs)
        for chunk in chunks:
            chunk.metadata["chunk_tier"] = tier["name"]
            chunk.metadata["source_total_chars"] = total_chars

        all_chunks.extend(chunks)

    logger.info(
        "Split %d source(s) into %d total chunks across %d tier(s)",
        len(source_docs),
        len(all_chunks),
        len({c.metadata.get("chunk_tier") for c in all_chunks}),
    )

    return all_chunks
