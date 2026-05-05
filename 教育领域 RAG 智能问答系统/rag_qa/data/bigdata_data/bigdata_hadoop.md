# Hadoop生态系统

## Hadoop概述

Apache Hadoop是一个开源的分布式计算框架，由Doug Cutting和Mike Cafarella于2005年创建，以Lucene和Nutch项目为基础。Hadoop的核心设计理念是将计算任务分布到多台廉价机器上并行处理。

## 核心组件

### HDFS（Hadoop Distributed File System）
分布式文件系统，提供高吞吐量的数据访问：

- **NameNode**：管理文件系统命名空间和元数据，记录每个文件被分割成的Block及其所在DataNode
- **DataNode**：存储实际数据块，定期向NameNode发送心跳和Block报告
- **Secondary NameNode**：辅助NameNode合并FsImage和EditLog，不是热备份

数据块默认128MB，每个Block默认3个副本，策略为：第一个副本在本地机架，第二个副本在同机架不同节点，第三个副本在不同机架。

### MapReduce
分布式计算编程模型：

**Map阶段**：读取输入数据，转换成键值对（key-value pair），对每条记录执行map函数

**Shuffle阶段**：按key对map输出进行分区、排序、合并，传输到Reduce节点

**Reduce阶段**：对每个key的值集合执行reduce函数，输出最终结果

```java
// WordCount示例（MapReduce）
public class WordCount {
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
        
        public void map(Object key, Text value, Context context) {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one);
            }
        }
    }
}
```

### YARN（Yet Another Resource Negotiator）
资源管理和作业调度框架：

- **ResourceManager**：全局资源管理器，接收作业提交，分配资源
- **NodeManager**：每个节点上的代理，监控容器资源使用
- **ApplicationMaster**：每个应用一个，向RM申请资源，管理任务执行

## Hadoop生态工具

- **Hive**：数据仓库工具，提供SQL-like查询语言（HQL）
- **HBase**：分布式NoSQL列式数据库，实时读写性能优秀
- **ZooKeeper**：分布式协调服务，提供配置管理、命名服务、分布式锁
- **Sqoop**：关系数据库与Hadoop之间的数据传输工具
- **Flume**：日志数据采集和聚合系统
- **Oozie**：工作流调度引擎

## 小文件问题

HDFS不适合存储大量小文件，原因：
1. NameNode在内存中存储元数据，每个文件约占用150字节
2. 大量小文件导致NameNode内存压力大
3. MapReduce处理小文件效率低

解决方案：使用Hadoop Archive（HAR）、SequenceFile、或合并小文件后再上传。
