# EduRAG 简历与面试讲解稿

## 简历项目名

教育领域 RAG 智能问答系统 | FastAPI + Milvus + BGE-M3 + DeepSeek

## 简历描述

- 设计并实现教育知识库 RAG 问答系统，支持 AI、Java、测试、运维、大数据等学科知识问答。
- 构建 Redis/MySQL/BM25 精确问答与 Milvus 混合向量检索双路由，结合 BERT 意图识别判断是否进入 RAG。
- 使用 BGE-M3 生成 dense/sparse 向量，Milvus WeightedRanker 混合召回，BGE-Reranker 对父块文档重排，提升上下文相关性。
- 基于 FastAPI + SSE 实现流式问答和检索过程可视化，展示意图分类、检索策略、召回文档与响应耗时。
- 使用 RAGAS 构建评估流程，覆盖 faithfulness、answer_relevancy、context_precision、context_recall；当前汇总分数约为 0.8034、0.8876、1.0、0.875。

## 项目流程

1. 用户输入 query，前端携带 session_id、学科过滤条件请求 `/api/chat/stream`。
2. 后端先查 Redis answer cache，命中则直接流式返回标准答案。
3. Redis 未命中后走 MySQL QA 表的 BM25 检索，softmax 分数超过 0.85 则返回标准问答。
4. BM25 未命中后使用 BERT 二分类器判断 query 是“通用知识”还是“专业咨询”。
5. 通用知识不进入知识库检索，直接由 LLM 回答；专业咨询进入 RAG。
6. RAG 阶段由策略选择器选择直接检索、HyDE、子查询或回溯问题检索。
7. BGE-M3 生成 dense/sparse 查询向量，Milvus 混合召回，BGE-Reranker 重排父块。
8. 取 top M 父块拼接上下文，结合最近 5 轮历史构造 prompt，由 DeepSeek 流式生成。

## 高频面试回答

### 你们如何设计意图识别？

项目里意图识别不是简单关键词规则，而是用 BERT 做二分类，把用户问题分成“通用知识”和“专业咨询”。这样做的原因是：通用寒暄、开放知识问题不一定需要检索内部知识库，直接检索反而会引入无关上下文；专业咨询才进入 RAG，能降低延迟、成本和上下文污染。

### 用户 query 怎么判断要不要 RAG？

系统是分层判断。第一层是 Redis/MySQL/BM25 标准问答，如果已有高置信答案就不走 RAG。第二层是 BERT 意图分类，如果属于通用知识也不走 RAG。只有 BM25 未命中且分类为专业咨询时，才进入 Milvus 检索和生成链路。

### 用户输入后完整检索流程是什么？

query 进入 RAG 后，先由策略选择器选择检索方式。直接检索会把原 query 送入 BGE-M3；HyDE 会先生成假设答案；子查询会拆成多个问题；回溯问题会把复杂问题简化。随后 BGE-M3 生成 dense/sparse 向量，Milvus 分别检索 dense_vector 和 sparse_vector，用 WeightedRanker 融合，再把子块映射回父块去重，最后用 BGE-Reranker 重排，把 top M 父块作为上下文交给 DeepSeek。

### RAG 评估怎么实现？

项目使用 RAGAS。评估数据包含 question、answer、contexts、ground_truth 四类字段，DeepSeek 作为评估 LLM，本地 BGE-M3 作为 embedding，计算 faithfulness、answer_relevancy、context_precision、context_recall。结果保存为 CSV 和 JSON，便于做 badcase 分析和版本对比。

### RAG 检索效果不好怎么优化？

我会按链路分层排查：先看原始文档质量、清洗、OCR 和切块是否合理；再调 parent/child chunk size、overlap、topK、dense/sparse 权重；然后看 query rewrite、HyDE、子查询是否真的改善召回；再引入 metadata filter、reranker、人工 badcase 集和 RAGAS 回归评估。最后才考虑换 embedding 模型或微调领域向量模型。

### 数据如何处理并写入 Milvus？

数据按学科目录组织，loader 负责加载 md/txt/pdf/docx/ppt/img 等文件，其中 PDF、图片、PPT、Word 支持 OCR 或专用 loader。每个文档打上 source、file_path、timestamp 元数据后，先切父块，再切子块；子块写入 Milvus，metadata 保留 parent_id 和 parent_content。写入时使用 BGE-M3 同时生成 dense_vector 和 sparse_vector，用内容 hash 做主键，通过 upsert 保证可重复导入。

## 后续学习方向

- RAG 工程：Self-RAG、CRAG、LightRAG、GraphRAG、query rewrite、hybrid search、reranking。
- RAG 评估：RAGAS 指标、离线 badcase 集、在线反馈采样、AB 对比。
- 部署工程：Docker Compose、Nginx、systemd、日志采集、密钥管理、GPU/CPU 推理部署。
- 大模型应用：Function Calling、Agent、MCP、工具调用、长上下文压缩和多轮记忆。
