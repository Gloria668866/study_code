# ETL流程设计与常用工具

## ETL概述

ETL（Extract-Transform-Load）是将数据从源系统抽取、转换后加载到目标系统的过程，是数据仓库建设的核心环节。

在数据量大、时效性要求高的场景下，也出现了ELT（Extract-Load-Transform）模式，先加载原始数据再在目标系统中进行转换，充分利用大数据平台的分布式计算能力。

## 数据抽取（Extract）

### 全量抽取
首次同步或定期全量同步全部数据：
- 适合数据量小或变更频率低的场景
- 简单但开销大，容易影响源库性能

### 增量抽取
只同步变化的数据，常用方法：

**时间戳方式**：
```sql
-- 抽取昨天更新的数据
SELECT * FROM orders WHERE updated_at >= '2024-01-01' AND updated_at < '2024-01-02';
```

**CDC（Change Data Capture）方式**：
- 基于日志：解析MySQL Binlog、Oracle Redo Log
- 基于触发器：在源表上创建触发器记录变更
- 基于快照：定期比较快照差异

常用CDC工具：Canal（阿里开源，MySQL Binlog解析）、Debezium（支持多种数据库）、Flink CDC

## 数据转换（Transform）

### 数据清洗
- 空值处理：填充默认值、丢弃或标记
- 格式统一：日期格式、编码转换、单位统一
- 无效值过滤：超出合理范围的异常值

### 数据标准化
- 维度退化：将维度属性合并到事实表
- 代码转换：将编码转为人可读的值
- 聚合计算：SUM、COUNT、AVG等指标计算

### 数据关联
```sql
-- ETL中的典型Join操作
INSERT INTO dwd_order_detail
SELECT 
    o.order_id,
    o.user_id,
    u.user_name,
    o.order_amount,
    o.created_at
FROM ods_orders o
LEFT JOIN dim_users u ON o.user_id = u.user_id
WHERE o.created_at >= '${yesterday}';
```

## 数据加载（Load）

### 加载策略
- **Insert**：新增数据
- **Update**：更新已有数据（需确定更新键）
- **Upsert（Merge）**：有则更新，无则插入
- **Delete**：标记或物理删除

### 加载优化
- 批量插入代替逐条插入
- 关闭约束检查和外键校验
- 使用分区表，加载完后再添加分区

## ETL工具

### 离线ETL工具

**DataX（阿里开源）**
- 异构数据源同步工具
- 支持MySQL、Oracle、HDFS、Hive、HBase等
- 通过Reader-Transformer-Writer插件化架构

**Sqoop**
- Hadoop生态的传统数据传输工具
- 通过MapReduce实现并行导入导出

### 实时ETL

**Flink CDC + Kafka**
```java
// Flink CDC读取MySQL Binlog
MySqlSource<String> mySqlSource = MySqlSource.<String>builder()
    .hostname("localhost")
    .port(3306)
    .databaseList("business_db")
    .tableList("business_db.orders")
    .username("user")
    .password("password")
    .deserializer(new JsonDebeziumDeserializationSchema())
    .build();

// 写入Kafka
stream.addSource(mySqlSource).addSink(kafkaSink);
```

## ETL最佳实践

1. **幂等设计**：ETL任务可重跑且结果一致
2. **监控与告警**：监控数据量波动、延迟、异常值
3. **分区分批处理**：避免一次处理过多数据
4. **小文件合并**：避免HDFS小文件问题
5. **数据校验**：ETL完成后检查源和目标数据量是否匹配
