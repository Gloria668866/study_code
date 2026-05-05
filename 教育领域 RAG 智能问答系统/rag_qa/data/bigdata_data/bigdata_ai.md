# 大数据与人工智能的结合应用

## 大数据与AI的关系

大数据和人工智能是相辅相成的技术。大数据为AI模型提供海量训练数据，AI技术为大数据处理提供智能化分析和自动化决策能力。两者结合形成了从数据采集、存储、处理到模型训练、推理部署的完整技术闭环。

## 大模型训练中的数据工程

### 数据采集与预处理

训练大语言模型需要TB-PB级别的文本数据：
- Common Crawl网页数据采集
- 维基百科、书籍、论文等结构化数据源
- 代码仓库数据（GitHub）
- 经过质量过滤、去重、去毒的清洗流程

**数据清洗Pipeline**：
```
原始数据 -> 语言检测 -> 质量评分 -> 去重 -> 
毒性过滤 -> PII去除 -> 格式标准化 -> 训练数据
```

使用Hadoop/Spark进行大规模数据清洗：
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col

spark = SparkSession.builder.appName("DataClean").getOrCreate()
df = spark.read.text("hdfs:///raw_corpus/*.txt")

# 去重
df = df.dropDuplicates(["value"])

# 质量过滤
@udf
def quality_filter(text):
    if len(text) < 100: return None
    if text.count('\n') > len(text) * 0.3: return None
    return text

df = df.withColumn("cleaned", quality_filter(col("value"))).filter(col("cleaned").isNotNull())
```

### 特征工程平台

推荐系统和搜索系统依赖大规模特征工程：
- 用户画像特征（年龄、性别、行为偏好）
- 物品特征（类目、价格、标签）
- 上下文特征（时间、位置、设备）
- 交叉特征（用户-物品交互历史）

使用Flink实时特征计算：
```
Kafka用户行为 -> Flink聚合窗口特征 -> Redis/Feature Store
```

## ML Pipeline与大数据

### 模型训练的数据准备

以推荐系统为例，使用Spark MLlib进行训练数据准备：

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# 特征工程Pipeline
indexer = StringIndexer(inputCol="category", outputCol="category_idx")
encoder = OneHotEncoder(inputCol="category_idx", outputCol="category_vec")
assembler = VectorAssembler(inputCols=["age", "price", "category_vec"], outputCol="features")

pipeline = Pipeline(stages=[indexer, encoder, assembler])
model = pipeline.fit(train_df)
train_features = model.transform(train_df)
```

### 分布式训练

对于大规模模型，使用分布式训练框架：
- **Parameter Server**：参数服务器存储模型参数，Worker节点计算梯度
- **Ring AllReduce**：GPU之间高效通信（Horovod框架）
- **数据并行**：每个Worker持有完整模型副本，处理不同数据分片

## 实时推理架构

### 模型服务化
```
用户请求 -> API网关 -> 模型推理服务 -> 返回预测结果
                |              |
            Kafka日志     Feature Store
                |              |
        Flink实时指标   Redis/在线存储
```

### 向量检索在RAG中的应用
将文档通过Embedding模型（如BGE-M3）编码为向量存入Milvus向量数据库，用户查询时检索最相关的文档作为LLM的上下文。这就是RAG（检索增强生成）技术的核心流程：

```
文档 -> Embedding Model -> 向量 -> Milvus
查询 -> Embedding Model -> 向量 -> Milvus检索 -> 
TopK文档 -> LLM Prompt -> 生成回答
```

## 数据治理与MLOps

### 数据版本管理
- 使用DVC（Data Version Control）或LakeFS管理训练数据版本
- 记录每次模型训练使用的数据版本和超参数

### 模型监控
- 数据漂移检测：监控生产数据分布与训练数据分布的偏差
- 模型性能监控：准确率、召回率、延迟
- 使用Prometheus + Grafana构建监控看板

### 数据合规
- GDPR/个人信息保护法合规要求
- 数据脱敏和匿名化处理
- 数据访问权限控制和审计日志
