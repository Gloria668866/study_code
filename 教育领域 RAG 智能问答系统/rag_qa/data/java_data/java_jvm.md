# JVM内存模型、垃圾回收机制与性能调优实战

## 一、JVM架构概述

Java虚拟机（JVM）是Java程序运行的基石，负责执行编译后的字节码文件（.class）。JVM的主要组成部分包括类加载器（ClassLoader）、运行时数据区（Runtime Data Area）、执行引擎（Execution Engine）和本地方法接口（JNI）。

JVM的核心职责：
- **加载字节码**：通过类加载器将.class文件加载到内存
- **内存管理**：分配和回收内存空间
- **执行字节码**：通过解释器和JIT编译器执行字节码
- **垃圾回收**：自动回收不再使用的对象
- **线程管理**：支持多线程并发执行

## 二、JVM内存模型（运行时数据区）

### 2.1 内存区域划分

JVM运行时数据区分为线程共享和线程私有两大类：

**线程共享区域**：
- **堆（Heap）**：所有对象实例和数组分配的内存，GC主要工作区域。细分为新生代（Young Generation）和老年代（Old Generation）
- **方法区（Method Area）/元空间（Metaspace）**：存储类信息、常量、静态变量、即时编译器编译后的代码缓存。JDK 8之前称为永久代（PermGen），JDK 8之后移至本地内存称为元空间

**线程私有区域**：
- **程序计数器（Program Counter Register）**：当前线程执行的字节码行号指示器
- **Java虚拟机栈（JVM Stack）**：每个方法执行时创建栈帧（Stack Frame），存储局部变量表、操作数栈、动态链接、返回地址等
- **本地方法栈（Native Method Stack）**：为本地Native方法服务

### 2.2 堆内存详解

```java
// JVM堆内存参数配置示例
// -Xms2g          初始堆大小2G
// -Xmx4g          最大堆大小4G
// -Xmn1g          新生代大小1G
// -XX:NewRatio=2  老年代/新生代比例=2（老年代:新生代=2:1）
// -XX:SurvivorRatio=8  Eden/Survivor比例=8（Eden:From:To=8:1:1）

// 查看JVM内存信息
Runtime runtime = Runtime.getRuntime();
long totalMemory = runtime.totalMemory();  // JVM总内存
long freeMemory = runtime.freeMemory();    // JVM空闲内存
long maxMemory = runtime.maxMemory();      // JVM最大可用内存
System.out.println("已使用内存: " + (totalMemory - freeMemory) / 1024 / 1024 + "MB");
```

**对象内存分配流程**：
1. 新对象首先尝试在栈上分配（逃逸分析优化）
2. 无法栈上分配则尝试TLAB（Thread Local Allocation Buffer）分配
3. TLAB无法分配则在Eden区分配
4. Eden区满时触发Minor GC，存活对象进入Survivor区
5. 达到年龄阈值的对象进入老年代
6. 大对象（可通过`-XX:PretenureSizeThreshold`设置）直接进入老年代

### 2.3 元空间（Metaspace）

JDK 8之后，方法区改用元空间实现，使用本地内存而非JVM堆内存，有效解决了永久代的OOM问题。

```java
// 元空间参数配置
// -XX:MetaspaceSize=128m          元空间初始大小
// -XX:MaxMetaspaceSize=256m       元空间最大大小
// -XX:MinMetaspaceFreeRatio=40    GC后最小空闲比例
// -XX:MaxMetaspaceFreeRatio=70    GC后最大空闲比例
// -XX:+UseCompressedClassPointers  使用压缩类指针（默认开启）
```

**类卸载条件**：
- 该类的所有实例已被回收
- 加载该类的ClassLoader已被回收
- 该类的Class对象没有在任何地方被引用

## 三、垃圾回收机制（GC）

### 3.1 如何判断对象已死

**引用计数法**：
- 每个对象维护一个引用计数器，引用加1，释放减1
- 缺点：无法解决循环引用问题
- Java未采用此方法

**可达性分析算法（Java采用）**：
- 以GC Roots为起点，通过引用链向下搜索
- 不可达的对象被标记为可回收对象
- GC Roots包括：虚拟机栈中引用的对象、方法区中静态属性引用的对象、常量引用的对象、JNI引用的对象、活跃线程对象

### 3.2 引用类型

Java提供四种引用类型，用于更灵活地控制对象生命周期：

```java
// 1. 强引用（Strong Reference）：默认引用类型，GC永不回收
Object obj = new Object();

// 2. 软引用（Soft Reference）：内存不足时回收，适合做缓存
SoftReference<byte[]> softRef = new SoftReference<>(new byte[1024 * 1024]);
byte[] data = softRef.get();  // 可能返回null

// 3. 弱引用（Weak Reference）：GC时必定回收
WeakReference<Object> weakRef = new WeakReference<>(new Object());
// 常用于ThreadLocal中的Entry

// 4. 虚引用（Phantom Reference）：无法通过它获取对象，仅用于跟踪回收
ReferenceQueue<Object> queue = new ReferenceQueue<>();
PhantomReference<Object> phantomRef = new PhantomReference<>(new Object(), queue);
```

### 3.3 垃圾回收算法

**标记-清除算法（Mark-Sweep）**：
- 先标记所有需要回收的对象，然后统一清除
- 缺点：产生内存碎片，效率不高

**标记-复制算法（Mark-Copy）**：
- 将内存分为两块，每次使用一块；存活对象复制到另一块，然后清空当前块
- 优点：无内存碎片；缺点：内存利用率只有50%
- HotSpot新生代使用此算法的优化版（Eden:Survivor0:Survivor1 = 8:1:1）

**标记-整理算法（Mark-Compact）**：
- 标记后将存活对象向一端移动，然后清理边界外的内存
- 优点：无内存碎片；缺点：移动对象需要STW（Stop The World），耗时较长
- 老年代通常使用此算法

### 3.4 垃圾回收器详解

| 回收器 | 工作区域 | 算法 | 特点 | 适用场景 |
|--------|---------|------|------|---------|
| Serial | 新生代 | 复制 | 单线程，STW | 客户端应用 |
| ParNew | 新生代 | 复制 | 多线程版Serial | 配合CMS使用 |
| Parallel Scavenge | 新生代 | 复制 | 吞吐量优先 | 后台计算 |
| Serial Old | 老年代 | 整理 | 单线程 | 客户端 |
| Parallel Old | 老年代 | 整理 | 多线程 | 配合Parallel Scavenge |
| CMS | 老年代 | 清除 | 低延迟 | 响应时间优先 |
| G1 | 全部 | 混合 | 平衡吞吐量与延迟 | 服务端默认（JDK9+） |
| ZGC | 全部 | - | 超低延迟(<10ms) | 大堆低延迟 |
| Shenandoah | 全部 | - | 超低延迟 | 大堆低延迟 |

**G1（Garbage First）回收器核心原理**：
- 将堆划分为多个大小相等的Region（默认2048个）
- 每个Region可以是Eden、Survivor、Old、Humongous（大对象）中的一种
- 优先回收垃圾比例最高的Region（Garbage First的由来）
- 通过Remembered Set记录跨Region引用，避免全堆扫描
- 支持可预测的停顿时间（`-XX:MaxGCPauseMillis=200`）

### 3.5 CMS与G1对比

```java
// CMS GC相关参数
// -XX:+UseConcMarkSweepGC        启用CMS
// -XX:CMSInitiatingOccupancyFraction=70  老年代使用率达70%时触发CMS GC
// -XX:+UseCMSCompactAtFullCollection     Full GC后进行内存碎片整理
// -XX:CMSFullGCsBeforeCompaction=5       每5次Full GC后进行一次碎片整理

// G1 GC相关参数
// -XX:+UseG1GC                    启用G1
// -XX:MaxGCPauseMillis=200        期望最大停顿时间200ms
// -XX:G1HeapRegionSize=4m         Region大小4MB（1MB~32MB，必须是2的幂）
// -XX:InitiatingHeapOccupancyPercent=45  堆使用率达到45%时触发并发标记周期
```

**CMS的主要问题**：
- 使用标记-清除算法，会产生内存碎片
- 对CPU资源敏感（并发阶段占用CPU）
- 无法处理浮动垃圾（并发标记期间新产生的垃圾）
- 预留空间不足时退化为Serial Old（单线程，STW时间长）

**G1的优势**：
- 整体基于标记-整理，Region间基于复制算法（G1更优）
- 可预测的停顿时间模型
- 不会因为内存碎片导致Full GC

## 四、JVM性能调优实战

### 4.1 常用JVM参数

```bash
# 基础内存配置
-Xms4g -Xmx4g                      # 初始堆与最大堆一致，避免动态扩容
-Xss256k                            # 每个线程栈大小256K
-XX:NewRatio=2                      # 老年代:新生代=2:1
-XX:SurvivorRatio=8                 # Eden:Survivor=8:1

# GC日志配置（JDK 8）
-XX:+PrintGCDetails                 # 打印GC详细信息
-XX:+PrintGCDateStamps              # 打印GC时间戳
-Xloggc:/var/log/gc-%t.log          # GC日志文件路径
-XX:+UseGCLogFileRotation           # GC日志滚动
-XX:NumberOfGCLogFiles=10           # 保留10个GC日志文件
-XX:GCLogFileSize=50M               # 每个GC日志50M

# GC日志配置（JDK 9+）
-Xlog:gc*:file=/var/log/gc.log:time,level,tags:filecount=10,filesize=50M

# OOM排查
-XX:+HeapDumpOnOutOfMemoryError     # OOM时自动生成堆转储
-XX:HeapDumpPath=/var/log/heapdump.hprof  # 堆转储文件路径
-XX:ErrorFile=/var/log/hs_err_pid%p.log   # 致命错误日志
-XX:OnOutOfMemoryError="kill -9 %p"       # OOM时执行脚本

# 其他常用参数
-XX:+DisableExplicitGC              # 禁用System.gc()调用
-Dsun.net.client.defaultConnectTimeout=5000   # Socket连接超时
-Dsun.net.client.defaultReadTimeout=10000      # Socket读超时
```

### 4.2 调优流程

**步骤1：确定调优目标**
- 吞吐量优先：单位时间内完成更多任务（如批量计算）
- 延迟优先：每次请求响应时间尽可能短（如Web服务）
- 内存占用优先：尽可能减少内存使用

**步骤2：选择垃圾回收器**
- 响应时间优先且堆小于4G：CMS
- 响应时间优先且堆大于4G：G1
- 超高吞吐量：Parallel Scavenge + Parallel Old
- 超低延迟且堆大于16G：ZGC

**步骤3：收集GC日志并分析**

使用工具分析GC日志：GCViewer、GCEasy、JProfiler

**步骤4：调优参数**

```bash
# 示例：4核8G Web应用G1调优配置
-Xms4g -Xmx4g
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200
-XX:ParallelGCThreads=4
-XX:ConcGCThreads=2
-XX:InitiatingHeapOccupancyPercent=45
-XX:G1ReservePercent=10
-XX:+PrintGCDetails
-XX:+PrintGCDateStamps
-Xloggc:/var/log/gc.log
-XX:+HeapDumpOnOutOfMemoryError
```

### 4.3 常见JVM问题排查

**OOM问题排查步骤**：
1. 分析堆转储文件（JProfiler、Eclipse MAT、jhat）
2. 定位大对象或内存泄漏点
3. 检查是否存在集合类无限增长（如Map未清理、ThreadLocal未remove）
4. 检查数据库连接、文件流是否正确关闭

```java
// 常见内存泄漏代码
// 1. ThreadLocal未清理
public class UserContext {
    private static final ThreadLocal<User> userHolder = new ThreadLocal<>();
    
    public static void setUser(User user) {
        userHolder.set(user);
    }
    // 缺少remove方法！线程池中线程复用会导致无法回收
    public static void clear() {
        userHolder.remove();  // 必须调用
    }
}

// 2. 静态集合无限增长
public class CacheManager {
    private static final Map<String, Object> cache = new HashMap<>();
    // 缺少淘汰机制，导致OOM
    // 应使用WeakHashMap或Guava Cache
}

// 3. 未关闭资源
public String readFile(String path) {
    FileInputStream fis = null;
    try {
        fis = new FileInputStream(path);
        // 读取操作
    } finally {
        if (fis != null) {
            fis.close();  // 必须关闭
        }
    }
}
```

## 五、类加载机制

### 5.1 类加载过程

类加载的生命周期包括7个阶段：加载、验证、准备、解析、初始化、使用、卸载。

- **加载**：通过类的全限定名获取二进制字节流，在内存中生成Class对象
- **验证**：确保字节码符合JVM规范（文件格式、元数据、字节码、符号引用验证）
- **准备**：为类变量（static）分配内存并设置类型零值
- **解析**：将符号引用替换为直接引用
- **初始化**：执行类构造器`<clinit>()`方法，赋值静态变量，执行静态代码块

### 5.2 双亲委派模型

```java
// 类加载器层次结构（JDK 9+）
// Bootstrap ClassLoader（启动类加载器）
//   └─ Platform ClassLoader（平台类加载器，原Extension ClassLoader）
//       └─ Application ClassLoader（应用类加载器）
//           └─ 自定义类加载器

// 双亲委派模型的核心逻辑
protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
    synchronized (getClassLoadingLock(name)) {
        // 1. 检查是否已加载
        Class<?> c = findLoadedClass(name);
        if (c == null) {
            try {
                if (parent != null) {
                    c = parent.loadClass(name, false);  // 2. 委托父加载器
                } else {
                    c = findBootstrapClassOrNull(name); // 3. 交给启动类加载器
                }
            } catch (ClassNotFoundException e) {
                // 父加载器无法加载
            }
            if (c == null) {
                c = findClass(name);  // 4. 自己尝试加载
            }
        }
        return c;
    }
}
```

**双亲委派模型的意义**：避免核心类库被篡改，保证Java类型体系安全。例如，用户自定义的java.lang.String不会被加载，因为会先被Bootstrap ClassLoader拦截。

## 六、总结

JVM内存模型和垃圾回收机制是Java开发者进阶的必经之路。理解堆内存分代结构、掌握主流GC回收器特点、学会进行性能调优和问题排查，对于构建高性能、高可用的Java应用至关重要。在实际工作中，需要根据业务场景合理选择GC策略，通过监控和日志分析及时发现和解决JVM层面的问题。
