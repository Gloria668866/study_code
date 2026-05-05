# 大数据开发面试高频考点

## Hadoop基础

### Q: HDFS读写流程是怎样的？
**读流程**：
1. Client向NameNode请求文件元数据
2. NameNode返回Block列表及所在DataNode位置
3. Client选择最近的DataNode读取Block
4. 读取完所有Block后合并为完整文件

**写流程**：
1. Client向NameNode请求创建文件
2. NameNode检查权限和命名空间，返回DataNode列表
3. Client将数据写入第一个DataNode
4. 第一个DataNode将数据转发给第二个，以此类推（Pipeline）
5. 所有副本写入完成后返回确认

### Q: NameNode HA如何实现？
通过Active NameNode和Standby NameNode实现高可用：
- 两个NameNode共享JournalNode集群（通常3个）
- Active写入EditLog到JournalNode，Standby读取并同步
- ZooKeeper监控NameNode状态，故障时自动切换
- 使用QJM（Quorum Journal Manager）保证日志一致性

### Q: MapReduce Shuffle过程详解？
Shuffle是MapReduce的核心阶段，发生在Map输出和Reduce输入之间：
1. Map端输出先写入环形内存缓冲区（默认100MB）
2. 达到阈值（80%）后Spill到磁盘，同时进行分区和排序
3. 合并多个Spill文件（Combiner可在此阶段执行局部聚合）
4. Reduce端通过HTTP拉取对应分区的数据
5. 合并来自不同Map Task的数据，排序后输入Reduce函数

## Spark

### Q: RDD的依赖关系有哪些？
- **窄依赖（Narrow Dependency）**：父RDD的每个分区最多被一个子RDD分区使用（map、filter、union）
- **宽依赖（Wide Dependency / Shuffle Dependency）**：父RDD的每个分区可能被多个子RDD分区使用（groupByKey、reduceByKey、join）

窄依赖支持Pipeline执行，宽依赖需要Shuffle，会划分Stage。

### Q: Spark为什么比MapReduce快？
1. 内存计算：中间结果缓存在内存中，避免重复磁盘I/O
2. DAG执行引擎：优化执行计划，减少不必要的Shuffle
3. 线程模型：Executor使用线程池复用，MR是进程级别
4. 数据本地性优化：尽量在数据所在节点执行计算

### Q: 如何解决Spark数据倾斜？
- 加盐：给倾斜Key加随机前缀，分散到多个分区
- 自定义分区器：根据数据分布设计分区策略
- 广播Join：将小表广播，避免Shuffle
- 拆分任务：将倾斜Key单独处理

## Flink

### Q: Flink的Checkpoint机制如何保证Exactly-Once？
基于Chandy-Lamport分布式快照算法和Barrier机制：
1. JobManager发出Checkpoint Barrier
2. Barrier流经所有算子，触发状态快照
3. 所有算子完成快照后，Checkpoint完成
4. 发生故障时从最近的Checkpoint恢复

配合二阶段提交（2PC）实现端到端的Exactly-Once（Source + Flink + Sink）。

## 大数据面试常见场景题

### 海量数据去重怎么处理？
- 数据量不大：使用HashSet或数据库唯一索引
- 数据量很大：使用BloomFilter（允许少量误判）
- 需要精确去重：使用HBase RowKey去重或Hive DISTINCT

### 如何设计一个实时数据仓库？
1. ODS层：Flink CDC采集MySQL Binlog + Kafka日志采集
2. DWD层：Flink实时清洗、关联维度
3. DWS层：Flink SQL实时聚合写入ClickHouse/Doris
4. ADS层：BI工具直接连接OLAP引擎查询
5. 使用数据湖（Hudi/Iceberg）管理历史数据

### MapReduce处理TopN问题
Map阶段：每个Mapper维护大小为N的最小堆（本地TopN）
Reduce阶段：合并所有Mapper的TopN结果，维护大小为N的最小堆
