import os
import shutil
import logging

import streamlit as st
from ingest import load_documents, split_documents
from vectorstore import build_vectorstore, load_vectorstore, VECTORSTORE_PATH, get_embeddings
from qa_chain import stream_qa, get_llm
from retrievers import get_cross_encoder
from utils import reset_singletons

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag")

st.set_page_config(
    page_title="RAG 知识库助手",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def init_knowledge_base(docs_path: str):
    if not os.path.exists(VECTORSTORE_PATH):
        docs = load_documents(docs_path)
        chunks = split_documents(docs)

        if not chunks:
            return None

        vectorstore, chunks = build_vectorstore(chunks)
        return vectorstore, chunks
    else:
        return load_vectorstore()


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None


def add_files_to_knowledge_base(uploaded_files):
    if not uploaded_files:
        return False

    docs_dir = "./docs"
    os.makedirs(docs_dir, exist_ok=True)

    saved_files = []
    try:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)

        with st.spinner("📂 正在加载上传的文档..."):
            docs = []
            for file_path in saved_files:
                fname = os.path.basename(file_path)
                if fname.endswith(".pdf"):
                    from langchain_community.document_loaders import PyPDFLoader
                    docs += PyPDFLoader(file_path).load()
                elif fname.endswith(".docx"):
                    from langchain_community.document_loaders import Docx2txtLoader
                    docs += Docx2txtLoader(file_path).load()
                elif fname.endswith(".txt") or fname.endswith(".md"):
                    from langchain_community.document_loaders import TextLoader
                    docs += TextLoader(file_path, encoding="utf-8").load()

            if not docs:
                st.error("❌ 无法解析上传的文件")
                return False

        with st.spinner(f"✂️ 正在切分文本... (共 {len(docs)} 个文档)"):
            chunks = split_documents(docs)

        if not chunks:
            st.error("❌ 文档片段为空")
            return False

        if os.path.exists(VECTORSTORE_PATH):
            with st.spinner("📦 加载现有向量库..."):
                existing_vectorstore, _ = load_vectorstore()

            with st.spinner("🔢 构建新文档的向量..."):
                new_vectorstore, _ = build_vectorstore(chunks)

            with st.spinner("🔄 合并向量库..."):
                existing_vectorstore.merge_from(new_vectorstore)
                existing_vectorstore.save_local(VECTORSTORE_PATH)

            st.cache_resource.clear()

            st.success(f"✅ 成功添加 {len(saved_files)} 个文件，新增 {len(chunks)} 个片段！")

            # Refresh session_state
            new_vs, new_chunks = load_vectorstore()
            st.session_state.vectorstore = new_vs
            st.session_state.chunks = new_chunks
        else:
            with st.spinner("🔢 首次构建向量库..."):
                build_vectorstore(chunks)
                st.cache_resource.clear()
                st.success(f"✅ 向量库已创建，共 {len(chunks)} 个片段！")

                new_vs, new_chunks = load_vectorstore()
                st.session_state.vectorstore = new_vs
                st.session_state.chunks = new_chunks

        return True

    except Exception as e:
        logger.exception("文档上传处理失败")
        st.error(f"❌ 处理失败: {str(e)}")
        return False


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📎 查看来源文档"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**来源 {i}:** {source}")


def handle_user_input():
    if user_input := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🤔 正在检索..."):
                try:
                    token_gen, source_docs = stream_qa(
                        user_input,
                        st.session_state.vectorstore,
                        st.session_state.chunks,
                    )
                except Exception as e:
                    logger.exception("检索失败")
                    error_msg = f"❌ 检索失败: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
                    return

            # Streaming phase
            answer_placeholder = st.empty()
            full_answer = ""
            try:
                for token in token_gen:
                    full_answer += token
                    answer_placeholder.markdown(full_answer + "▌")
            except Exception as e:
                logger.exception("流式生成中断")
                full_answer += f"\n\n❌ 生成中断: {str(e)}"
            finally:
                answer_placeholder.markdown(full_answer)

            sources = list(set(
                doc.metadata.get("source", "未知") for doc in source_docs
            ))

            if sources:
                with st.expander("📎 查看来源文档"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**来源 {i}:** {source}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
            })


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


def main():
    st.title("🤖 RAG 知识库助手")
    st.markdown("---")

    if "models_warmed" not in st.session_state:
        with st.spinner("🔥 正在初始化模型 (首次加载较慢)..."):
            warmup_models()
        st.session_state.models_warmed = True

    with st.sidebar:
        st.header("⚙️ 设置")

        st.markdown("---")
        st.header("📁 上传文档")

        if st.session_state.get("upload_success", False):
            st.success("✅ 文档已成功添加到知识库！")
            st.info("💡 您可以继续上传其他文件，或开始对话")

            if st.button("🔄 继续上传更多文件", use_container_width=True):
                st.session_state.upload_success = False
                st.rerun()
        else:
            uploaded_files = st.file_uploader(
                "选择文件上传到知识库",
                type=["pdf", "txt", "md", "docx"],
                accept_multiple_files=True,
                help="支持 PDF、DOCX、TXT、MD 格式",
            )

            if uploaded_files:
                st.info(f"已选择 {len(uploaded_files)} 个文件")
                for f in uploaded_files:
                    st.text(f"📄 {f.name}")

                if st.button("📤 添加到知识库", use_container_width=True, type="primary"):
                    with st.spinner("⏳ 正在处理文档..."):
                        success = add_files_to_knowledge_base(uploaded_files)
                    if success:
                        st.session_state.upload_success = True
                        st.rerun()

        st.markdown("---")
        st.markdown("### 📊 系统信息")
        st.info(f"向量库路径: `{VECTORSTORE_PATH}`")
        st.info(f"向量库存在: {'✅' if os.path.exists(VECTORSTORE_PATH) else '❌'}")

        docs_path = st.text_input(
            "文档路径",
            value="./docs",
            help="放置知识库文档的文件夹路径",
        )

        if st.button("🔄 重新构建向量库", use_container_width=True):
            if os.path.exists(VECTORSTORE_PATH):
                shutil.rmtree(VECTORSTORE_PATH)
                st.cache_resource.clear()
                reset_singletons()
                st.session_state.models_warmed = False
                st.rerun()

        if st.button("🗑️ 清空聊天历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    initialize_session_state()

    if not os.path.exists(VECTORSTORE_PATH):
        with st.spinner("🔢 正在构建向量库，请稍候..."):
            result = init_knowledge_base(docs_path)
        if result is not None:
            _, chunks = result
            st.success(f"✅ 向量库构建成功！共 {len(chunks)} 个片段")
    else:
        with st.spinner("📦 加载已有向量库..."):
            result = init_knowledge_base(docs_path)
    if result is None:
        st.error("❌ 知识库初始化失败")
        return

    vectorstore, chunks = result

    st.session_state.vectorstore = vectorstore
    st.session_state.chunks = chunks

    st.markdown("### 💬 开始对话")
    st.markdown("您可以询问关于知识库中的任何问题，我会基于提供的文档给您回答。")

    display_chat_history()
    handle_user_input()


if __name__ == "__main__":
    main()
