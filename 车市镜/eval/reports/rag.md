# RAG（RAGAS 口径）评测报告

- 样本数：**60**　分类：{'single': 26, 'multi': 16, 'conflict': 6, 'none': 12}

## 一、可答题四指标（single+multi+conflict）

| 指标 | 值 |
|---|---|
| context precision | 0.9167 |
| context recall | 0.894 |
| faithfulness（忠实度/防幻觉） | 0.7973 |
| answer relevancy（答案相关性） | 0.9444 |
| 检索命中率 hit-recall（确定性） | 0.7604 |
| 应答率 answer_rate | 0.75 |

## 二、防幻觉（none 类，库里没有应拒答）

- 拒答正确率 abstention_rate：**1.0**（n=12）

## 三、多源冲突处理（conflict 类）

- 并列两口径正确率 handle_rate：**1.0**（n=6）

## 四、检索消融（hit-recall，量化父子分块+rerank 增益）

| 配置 | hit-recall |
|---|---|
| 纯向量召回 | 0.75 |
| 混合召回(向量+全文 RRF) | 0.75 |
| 混合召回 + rerank（完整系统） | 0.7619 |
（消融样本 n=42）