import os
import sys
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
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.path.exists(VECTORSTORE_PATH):
        vectorstore, chunks = init_knowledge_base("./docs")
    else:
        logger.info("加载已有向量库...")
        vectorstore, chunks = load_vectorstore()

    chain = build_qa_chain(vectorstore, chunks)
    chat(chain)
