import os
import logging

from ingest import load_documents, split_documents
from vectorstore import build_vectorstore, load_vectorstore, VECTORSTORE_PATH
from qa_chain import build_qa_chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag")


def init_knowledge_base(docs_path: str):
    logger.info("加载文档...")
    docs = load_documents(docs_path)
    logger.info("加载到 %d 个文档对象", len(docs))

    logger.info("切分文本...")
    chunks = split_documents(docs)
    logger.info("切分后 %d 个 chunk", len(chunks))

    if not chunks:
        raise ValueError("chunks 为空，请检查文档路径和格式")

    vectorstore, _ = build_vectorstore(chunks)
    return vectorstore, chunks


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
