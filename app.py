import os
import shutil
import logging

import streamlit as st
from knowledge_base import warmup_models, init_knowledge_base, add_files_to_knowledge_base
from qa_chain import stream_qa
from retrieval import load_vectorstore
from config import VECTORSTORE_PATH
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
def _cached_init_kb(docs_path: str):
    return init_knowledge_base(docs_path)


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None



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
                        result = add_files_to_knowledge_base(uploaded_files)
                    if result["success"]:
                        st.cache_resource.clear()
                        st.session_state.upload_success = True
                        st.success(f"✅ 成功添加 {result['files']} 个文件，新增 {result['chunks']} 个片段！")

                        new_vs, new_chunks = load_vectorstore()
                        st.session_state.vectorstore = new_vs
                        st.session_state.chunks = new_chunks

                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")

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
            result = _cached_init_kb(docs_path)
        if result is not None:
            _, chunks = result
            st.success(f"✅ 向量库构建成功！共 {len(chunks)} 个片段")
    else:
        with st.spinner("📦 加载已有向量库..."):
            result = _cached_init_kb(docs_path)
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
