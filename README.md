# 🧠 shopkeeper_brain (掌柜智库)

> 掌柜智库是一个基于大语言模型（LLM）与检索增强生成（RAG）技术的企业级智能知识库问答系统。

## 📖 项目简介

本项目旨在构建一个高精度的私有化知识库问答平台。通过引入**双路混合检索（Dense + Sparse）**、**RRF 倒数排名融合算法**以及**Agent 工作流**，解决传统大模型在垂直领域“易产生幻觉”和“缺乏私有数据”的痛点。

## ✨ 核心特性

- 🔍 **混合检索引擎**：深度集成 Milvus 向量数据库，支持 BGE-M3 稠密向量与稀疏向量双路检索，结合 RRF 算法重排，大幅提升召回准确率。
- 🧩 **Agent 智能体编排**：基于 LangGraph 构建有状态的工作流，支持人工介入（Human-in-the-loop）与复杂任务拆解。
- 📄 **多格式文档解析**：支持 PDF、Word、Markdown 等格式的自动化解析与智能分块（Chunking）。
- ⚡ **极速依赖管理**：全面拥抱下一代 Python 包管理工具 `uv`，实现环境的秒级重建。
- 🐳 **一键式基建部署**：依赖 Docker Compose 实现 Milvus、MongoDB、MinIO 等核心组件的开箱即用。

## 🛠️ 技术栈

*   **AI 框架**：LangChain 1.x / LangGraph / OpenAI SDK
*   **大模型 (LLM)**：通义千问 (qwen-flash)
*   **Embedding 模型**：BGE-M3 (支持 GPU 加速)
*   **基础设施**：
    *   向量数据库：Milvus v2.5.5
    *   元数据存储：MongoDB 7.0
    *   对象存储：MinIO
    *   可视化面板：Attu
*   **开发环境**：Python 3.12 / `uv` / PyCharm / Docker Desktop

