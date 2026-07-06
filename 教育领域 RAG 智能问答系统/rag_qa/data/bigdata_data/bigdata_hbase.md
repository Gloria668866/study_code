# HBase分布式NoSQL数据库

## HBase概述

Apache HBase是一个分布式的、面向列的NoSQL数据库，运行在HDFS之上，由Powerset公司于2007年创建，灵感来自Google BigTable论文。HBase提供对大规模数据的实时读写访问能力。

## 数据模型

### 核心概念
- **Table**：表，由多行组成
- **RowKey**：行键，唯一标识一行，按字典序排序
- **Column Family**：列族，创建表时定义，是存储和访问控制的基本单元
- **Column Qualifier**：列限定符，可在运行时动态添加
- **Cell**：由{RowKey, Column Family, Column Qualifier, Timestamp}唯一确定的单元格
- **Timestamp**：版本号，HBase自动维护，支持多版本存储

### RowKey设计原则

RowKey是HBase最重要的设计决策：

1. **长度适中**：过长浪费存储，过短难以均匀分布。建议16-64字节
2. **散列化**：避免热点，使用MD5/SHA1哈希或反转时间戳
3. **业务查询优先**：将常用查询条件编码到RowKey中

```
// 好的RowKey设计：Hash(UserID)[0:4] + UserID + Timestamp.reverse()
// 散列前缀避免热点，ID便于查询，反转时间戳使最新数据排在前面

// 坏的设计：直接使用时间戳作为RowKey
// 会导致所有写入集中在一个Region，造成热点
```

## HBase架构

### HMaster
- 管理RegionServer的负载均衡
- 处理表结构变更（DDL）
- 管理Region的分配和故障恢复

### RegionServer
- 处理客户端读写请求
- 管理Region的分裂（Split）
- 与HDFS交互进行实际的数据读写

### Region
表按RowKey范围水平切分为多个Region，每个Region存储连续范围的RowKey数据：
```
Region 1: RowKey [0000 - 5555]
Region 2: RowKey [5556 - aaaa]
Region 3: RowKey [aaab - ffff]
```

### ZooKeeper
- 维护集群元数据
- HMaster选举
- 监控RegionServer状态

## 读写流程

### 写入流程
1. Client向ZooKeeper查询META表位置
2. 根据RowKey找到对应RegionServer
3. 先写WAL（Write-Ahead Log）确保持久性
4. 再写入MemStore（内存）
5. MemStore达到阈值后Flush为HFile

### 读取流程
1. Client查询META表定位RegionServer
2. RegionServer合并读取BlockCache、MemStore和HFile
3. BlockCache（LRU缓存）命中直接返回
4. MemStore未命中，扫描HFile
5. BloomFilter快速判断RowKey是否在HFile中

## 常见问题与优化

### 热点问题
某个Region承载过多读写请求：
- 解决：RowKey加盐、随机前缀、反转Key

### Full GC问题
MemStore占用过多堆内存导致GC停顿：
- 增大堆内存或开启MSLAB（MemStore Local Allocation Buffer）

### Compaction策略
HFile数量过多影响读性能，需要Compaction合并：
- Minor Compaction：合并少量HFile
- Major Compaction：合并所有HFile，清理删除标记
- 建议在低峰期进行Major Compaction
