# Apache Flink流处理引擎

## Flink概述

Apache Flink是一个分布式流处理框架，由德国柏林工业大学的研究项目Stratosphere发展而来。Flink以"有状态的流处理"为核心，支持事件时间（Event Time）语义和精确一次（Exactly-Once）状态一致性。

## 核心概念

### 流处理 vs 批处理
Flink认为批处理是流处理的特殊形式（有界流）。DataStream API处理无界流数据，DataSet API处理有界批数据，Flink SQL则统一了两种处理模式。

### 时间语义

三种时间概念：
- **Event Time**：事件实际发生的时间（最常用）
- **Processing Time**：事件被Flink处理的时间
- **Ingestion Time**：事件进入Flink系统的时间

### Watermark水位线
用于处理乱序事件的机制。Watermark表示"所有时间戳小于t的事件都已经到达"，允许Flink在事件延迟和结果完整性之间做出权衡。

```java
// 设置Watermark策略
stream.assignTimestampsAndWatermarks(
    WatermarkStrategy
        .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
        .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
);
```

## Window窗口

### 窗口类型

**时间窗口**：
- 滚动窗口（Tumbling Window）：固定大小，互不重叠
- 滑动窗口（Sliding Window）：固定大小，有重叠（由滑动步长决定）
- 会话窗口（Session Window）：根据活动间隔动态合并

```java
// 滚动窗口示例
stream
    .keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .sum("count");

// 滑动窗口示例
stream
    .keyBy(Event::getUserId)
    .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(5)))
    .aggregate(new MyAggregateFunction());
```

## 状态管理与Checkpoint

Flink的有状态计算依赖两个核心机制：

### State状态
- **Keyed State**：每个key独立维护状态
- **Operator State**：整个算子共享状态
- State可作为内存、RocksDB存储后端

### Checkpoint
定期保存全局一致性快照，基于Chandy-Lamport分布式快照算法：

```
checkpointCoordinator触发 -> Source注入barrier -> 
Operator收到barrier后对齐 -> 各节点snapshot完成 ->
checkpointCoordinator确认完成
```

## 反压机制

Flink通过Credit-based Flow Control处理反压：
1. 下游通过Netty向上游传递可用buffer数量
2. 上游根据credit决定发送数据量
3. 下游消费慢时credit减少，上游自动降速

## Flink vs Spark Streaming

| 对比维度 | Flink | Spark Streaming |
|----------|-------|-----------------|
| 处理模型 | 原生流处理 | 微批处理 |
| 延迟 | 毫秒级 | 秒级 |
| Exactly-Once | 原生支持 | 需要外部支持 |
| 事件时间 | 强大支持 | 有限支持 |
| 生态系统 | 相对年轻 | 更成熟 |

## Flink应用场景
- 实时数仓：Flink CDC + Kafka + Flink SQL
- 实时风控：低延迟规则引擎
- 实时推荐：用户行为流 + 模型推理
- 日志分析：实时聚合和告警
