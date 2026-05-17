# RAG 知识库助手

基于 Retrieval-Augmented Generation (RAG) 架构的智能知识库问答系统，支持多格式文档的语义检索与智能问答，提供 CLI 和 Web 两种交互方式。

## 功能模块

### 文档处理
- 支持 **PDF**、**TXT**、**Markdown** 三种格式的文档导入
- **动态分块策略**：根据文档长度自动选择分块大小（短文档细粒度、长文档粗粒度），优化检索效果
- 递归文本切分，保留语义边界

### 向量存储
- 基于 **FAISS** 的高效向量索引
- 使用 DashScope **text-embedding-v1** 模型生成文本嵌入
- 向量库本地持久化，支持增量合并

### 混合检索
- **Dense 检索**：FAISS 语义相似度搜索
- **Sparse 检索**：BM25 关键词匹配
- **RRF 融合**（Reciprocal Rank Fusion）：合并两路检索结果
- **CrossEncoder 重排序**：使用 `ms-marco-MiniLM-L-6-v2` 模型对候选文档精排

### 智能问答
- LLM 基于检索到的上下文生成答案，严格限定知识库范围，不编造内容
- **流式输出**：Web 界面支持 token-by-token 实时流式生成，提升交互体验
- 返回答案的同时标注引用来源
- 支持 CLI 命令行交互和 Streamlit Web 界面

### Web 界面 (Streamlit)
- 对话式 Chat UI
- 侧边栏文件上传，支持增量添加文档到知识库
- 一键重建向量库
- 聊天历史管理

## 技术栈

| 分类 | 技术 |
|------|------|
| **LLM** | 阿里云通义千问 qwen-plus (DashScope API) |
| **Embedding** | DashScope text-embedding-v1 |
| **向量数据库** | FAISS (CPU) |
| **检索框架** | LangChain (Community + Core + Classic) |
| **重排序** | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| **Web 框架** | Streamlit |
| **深度学习** | PyTorch, HuggingFace Transformers, Sentence-Transformers |
| **文档解析** | PyPDF2, python-docx |
| **语言** | Python 3.14 |

## 项目结构

```
RAG/
├── main.py          # CLI 入口，命令行交互问答（非流式）
├── app.py           # Streamlit Web 应用入口（流式输出）
├── config.py        # 配置文件（API Key、分块参数等）
├── ingest.py        # 文档加载与文本切分
├── vectorstore.py   # 向量库构建、加载与嵌入模型管理
├── qa_chain.py      # QA Chain（含流式 stream_qa 与非流式 build_qa_chain）
├── retrievers.py    # 混合检索器（Dense + BM25 + RRF + CrossEncoder）
├── utils.py         # 工具函数（重试机制、单例模式）
├── docs/            # 知识库文档存放目录
└── vectorstore/     # FAISS 向量库持久化目录
```

## 快速开始

### 环境要求

- Python 3.10+
- 阿里云 DashScope API Key

### 安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install langchain langchain-community langchain-classic langchain-core
pip install dashscope faiss-cpu streamlit
pip install transformers torch sentence-transformers
pip install PyPDF2 python-docx python-dotenv
```

### 配置

编辑 `.env` 文件，设置你的 DashScope API Key：

```env
OPENAI_API_KEY=your-dashscope-api-key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 准备知识库文档

将需要检索的 PDF、TXT、MD 文件放入 `./docs` 目录。

### 启动 CLI 模式

```bash
python main.py
```

### 启动 Web 界面

```bash
streamlit run app.py
```

在浏览器中打开显示的地址（默认 `http://localhost:8501`），即可通过 Web 界面上传文档并进行问答。

## 核心特性

- **分层分块**：根据文档字数（≤2k / ≤100k / >100k）自动选择 chunk_size（500 / 900 / 1300），小块文档精细切分，大块文档粗粒度切分
- **混合检索 + 精排**：结合语义搜索和关键词匹配的优势，再通过 CrossEncoder 进行精确重排序
- **单例模型管理**：Embedding、LLM、CrossEncoder 均采用单例模式，避免重复加载
- **指数退避重试**：API 调用失败自动重试（最多 3 次）
- **流式输出**：Web 界面实现 LLM token-by-token 实时流式生成，用户可即时看到回答内容，无需等待完整生成
- **来源追溯**：每次回答均标注引用文档，确保可追溯性
