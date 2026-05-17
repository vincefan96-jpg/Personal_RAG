import os
import logging

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logging.getLogger("rag").warning("OPENAI_API_KEY 未设置，API 调用将失败")

VECTORSTORE_PATH = "vectorstore"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Dynamic chunking tiers — selected by per-source total character count.
# Evaluated in dict insertion order; first tier with max_chars >= total_chars wins.
# The last tier MUST use float("inf") as a catch-all.
CHUNK_TIERS = {
    "small": {
        "max_chars": 2_000,
        "chunk_size": 500,
        "chunk_overlap": 100,
    },
    "medium": {
        "max_chars": 100_000,
        "chunk_size": 900,
        "chunk_overlap": 135,
    },
    "large": {
        "max_chars": float("inf"),
        "chunk_size": 1300,
        "chunk_overlap": 150,
    },
}

CHUNK_SEPARATORS = ["\n\n", "\n", "。", ".", " ", ""]
