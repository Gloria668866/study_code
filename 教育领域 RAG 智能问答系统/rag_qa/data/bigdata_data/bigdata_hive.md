# Hive数据仓库

## Hive概述

Apache Hive是建立在Hadoop之上的数据仓库基础设施，由Facebook开发。它提供了SQL-like查询语言（HiveQL，简称HQL），将SQL查询转换为MapReduce/Tez/Spark作业执行，使不熟悉MapReduce编程的分析师也能使用大数据平台。

## Hive架构

### 核心组件
- **Metastore**：存储元数据（表结构、分区信息、列类型等），通常使用MySQL/PostgreSQL
- **Driver**：接收查询、解析和编译SQL
- **Compiler**：将SQL转换为逻辑执行计划、物理执行计划，最终生成MR/Tez/Spark任务
- **Execution Engine**：执行编译后的任务

### Hive与传统RDBMS区别

| 特性 | Hive | RDBMS |
|------|------|-------|
| 数据存储 | HDFS | 本地文件系统 |
| 查询语言 | HQL | SQL |
| 延迟 | 高（秒到分钟级） | 低（毫秒到秒） |
| 事务 | ACID（3.0+支持） | ACID |
| 适用场景 | 批量分析 | 在线事务处理 |
| 扩展性 | 水平扩展（1000+节点） | 垂直扩展为主 |

## 表类型

### 内部表（Managed Table）
Hive管理数据生命周期，删除表时数据也被删除：
```sql
CREATE TABLE employees (
    id INT, name STRING, salary DOUBLE
) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';
```

### 外部表（External Table）
Hive只管理元数据，删除表时数据保留在HDFS：
```sql
CREATE EXTERNAL TABLE page_views (
    user_id INT, page_url STRING, view_time TIMESTAMP
) LOCATION '/user/hive/external/page_views';
```

## 分区与分桶

### 分区（Partition）
按字段值将数据划分为子目录，查询时可跳过无关分区：
```sql
CREATE TABLE sales (
    id INT, amount DOUBLE
) PARTITIONED BY (year INT, month INT);

-- 查询时分区剪裁
SELECT SUM(amount) FROM sales WHERE year=2024 AND month=1;
```

### 分桶（Bucket）
将数据Hash分割为固定数量的文件，便于Sampling和Join优化：
```sql
CREATE TABLE users (
    id INT, name STRING
) CLUSTERED BY (id) INTO 32 BUCKETS;
```

## 文件格式

| 格式 | 特点 | 适用场景 |
|------|------|----------|
| TEXTFILE | 纯文本，可读性好 | 数据交换、调试 |
| ORC | 列式存储，高压缩率，索引 | 分析查询（推荐） |
| Parquet | 列式存储，跨平台兼容 | Spark/Presto联合查询 |
| Avro | 行式存储，Schema演化 | 数据采集、序列化 |

## 性能优化

1. **分区剪裁（Partition Pruning）**：WHERE条件中指定分区字段
2. **使用ORC/Parquet列式格式**，查询时只读取需要的列
3. **Map Join**：小表放入内存，避免Shuffle
4. **Vectorization**：批量处理数据行而非逐行处理
5. **CBO（Cost-Based Optimizer）**：基于统计信息选择最优执行计划
6. **合理设置Reducer数量**：`set mapreduce.job.reduces=N`
