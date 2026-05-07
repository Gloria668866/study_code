# 简历项目描述

## 项目名称

中医药知识图谱增强问答系统

## 项目简介

基于 DeepSeek、LangGraph 思路、Neo4j、FAISS、FastAPI 和 Streamlit 构建中医药知识图谱增强问答系统，实现中药/方剂知识的结构化管理、语义检索和 RAG/GraphRAG 问答。系统支持 Neo4j 图谱模式与本地向量检索模式自动降级，保证演示稳定性。

## 技术栈

Python, FastAPI, Streamlit, DeepSeek API, Neo4j, FAISS, Sentence-Transformers, Pydantic

## 项目亮点

- 构建中医药实体与方剂关系数据模型，支持中药、方剂、症状、疾病等实体统一管理。
- 基于 Sentence-Transformers + FAISS 实现中医药知识语义检索。
- 接入 DeepSeek 兼容 OpenAI API，将检索结果作为上下文生成可解释回答。
- 设计 Local RAG 与 Neo4j GraphRAG 双模式降级机制，提升系统可用性。
- 使用 FastAPI 暴露问答、检索、健康检查与图谱状态接口。
- 使用 Streamlit 实现可交互 Web 演示页面，展示检索依据与运行模式。
