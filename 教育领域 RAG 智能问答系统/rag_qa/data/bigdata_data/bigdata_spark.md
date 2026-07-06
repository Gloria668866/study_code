# Apache Spark核心概念

## Spark概述

Apache Spark是一个快速、通用的分布式计算引擎，由UC Berkeley AMPLab于2009年开发。Spark基于内存计算，比Hadoop MapReduce快10-100倍。

## RDD（Resilient Distributed Dataset）

RDD是Spark的核心抽象，是一个不可变的、可分区的、可并行操作的分布式数据集合。

### RDD特性
- **不可变性（Immutable）**：创建后不能修改，只能通过转换生成新的RDD
- **分区（Partition）**：数据分布在集群不同节点上
- **容错性（Fault-Tolerant）**：通过血统（Lineage）自动重建丢失的分区

### RDD操作

**Transformation（转换）**：延迟计算，返回新RDD：
```python
rdd = sc.parallelize([1, 2, 3, 4, 5])
filtered = rdd.filter(lambda x: x > 2)     # [3, 4, 5]
mapped = rdd.map(lambda x: x * 2)           # [2, 4, 6, 8, 10]
flat = rdd.flatMap(lambda x: [x, x*10])    # [1,10, 2,20, ...]
```

**Action（动作）**：触发计算，返回值：
```python
rdd.collect()     # 收集所有元素到Driver
rdd.count()       # 计数
rdd.take(3)       # 取前3个
rdd.reduce(lambda a, b: a + b)  # 聚合
```

## DataFrame与Spark SQL

DataFrame是比RDD更高层的API，带有Schema信息：

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("demo").getOrCreate()

df = spark.read.csv("data.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("users")

# SQL查询
result = spark.sql("SELECT department, AVG(salary) FROM users GROUP BY department")

# DataFrame API
from pyspark.sql.functions import col, avg
result = df.groupBy("department").agg(avg(col("salary")))
```

## Spark优化配置

### 内存管理
- `spark.executor.memory`：Executor内存
- `spark.driver.memory`：Driver内存
- `spark.memory.fraction`：执行+存储内存占比（默认0.6）

### 并行度
- `spark.sql.shuffle.partitions`：Shuffle分区数（默认200）
- 建议设置为Executor核心数的2-3倍

### 数据倾斜处理
- 加盐（Salting）：给倾斜的key加随机前缀
- 广播Join：小表广播到每个Executor
- 调整并行度：增加分区数

## Spark vs MapReduce

| 对比维度 | Spark | MapReduce |
|----------|-------|-----------|
| 计算模型 | 内存计算 + 磁盘 | 磁盘为主 |
| 速度 | 快10-100倍 | 慢 |
| 编程模型 | RDD/DataFrame/SQL | Map + Reduce |
| 交互式查询 | 支持 | 不支持 |
| 流处理 | Spark Streaming | 不支持 |
| 适用场景 | 迭代计算、ML、实时 | 批量处理 |
