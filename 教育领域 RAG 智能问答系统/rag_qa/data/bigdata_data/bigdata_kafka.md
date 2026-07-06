# Kafka分布式消息队列

## Kafka概述

Apache Kafka是由LinkedIn开发、后捐献给Apache基金会的分布式消息系统。由Jay Kreps、Neha Narkhede等人创建。Kafka以高吞吐量、低延迟、可持久化和水平扩展能力著称，单集群可支持每秒百万级消息处理。

## 核心概念

### 基础架构

- **Producer**：消息生产者，向Topic发送消息
- **Consumer**：消息消费者，从Topic拉取消息
- **Broker**：Kafka服务器节点
- **Topic**：消息的逻辑分类，类似数据库的表
- **Partition**：Topic的物理分区，每个Partition是有序的不可变消息序列
- **Consumer Group**：消费者组，组内每个Consumer负责不同Partition
- **ZooKeeper**：集群元数据管理和协调（新版本已转向KRaft模式）

### 消息模型

Kafka采用Pull模型：
1. Producer将消息写入Leader Partition
2. Follower从Leader同步数据（ISR机制）
3. Consumer从Broker拉取消息，自己管理Offset

## 分区与副本

### 分区策略
- Producer默认使用Hash(key) % partitionCount决定写入分区
- 同一Key的消息保证顺序性（在同一分区内）
- 增加分区可提升并行度，但分区数只能增加不能减少

### 副本机制
- 每个分区有一个Leader和多个Follower
- 读写都通过Leader进行
- ISR（In-Sync Replicas）：与Leader保持同步的副本集合
- `min.insync.replicas`：最少同步副本数，保证消息可靠性

## 消费者组

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "order-consumer-group");
props.put("enable.auto.commit", "false");
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Arrays.asList("orders"));

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process(record);
    }
    consumer.commitSync();
}
```

## 消息可靠性保证

### Producer端
- `acks=0`：不等待确认（最高吞吐，可能丢数据）
- `acks=1`：Leader写入即确认（可能丢数据若Leader宕机）
- `acks=all` 或 `acks=-1`：所有ISR确认（最高可靠）

### 幂等性与事务

**幂等Producer**：`enable.idempotence=true`，保证单分区Exactly-Once语义

**事务Producer**：跨分区原子写入：
```java
producer.initTransactions();
producer.beginTransaction();
producer.send(record1);
producer.send(record2);
producer.commitTransaction();
```

## 性能优化

### Producer优化
- `batch.size`：批量发送大小（默认16KB）
- `linger.ms`：等待更多消息的时间
- `compression.type`：压缩类型（gzip/snappy/lz4/zstd）

### Consumer优化
- 合理设置`fetch.min.bytes`和`fetch.max.wait.ms`
- 增加Consumer实例数（不超过Partition数）
- 使用异步提交Offset

### Broker优化
- 使用SSD磁盘
- Page Cache利用（Kafka重度依赖OS Page Cache）
- JVM堆内存设置（通常4-6GB即可，剩余内存给Page Cache）
