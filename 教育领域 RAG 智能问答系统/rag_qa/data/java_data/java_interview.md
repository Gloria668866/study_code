# Java开发面试高频考点与解答精编

## 一、Java基础面试题

### 1.1 Java基础核心问题

**Q1: String、StringBuilder、StringBuffer的区别？**

String是不可变类，每次修改都会创建新对象，适用于少量字符串操作。StringBuilder是可变类，非线程安全，单线程环境下性能最高。StringBuffer是可变类，所有方法使用synchronized修饰，线程安全但性能较低。

```java
// 性能对比：循环拼接场景
String result = "";
for (int i = 0; i < 10000; i++) {
    result += i;  // 创建10000个String对象，OOM风险
}

StringBuilder sb = new StringBuilder();
for (int i = 0; i < 10000; i++) {
    sb.append(i);  // 仅1个对象，性能远超String
}

// 字符串常量池
String s1 = "hello";
String s2 = "hello";
String s3 = new String("hello");
System.out.println(s1 == s2);     // true（指向常量池同一对象）
System.out.println(s1 == s3);     // false（new在堆上创建新对象）
System.out.println(s1.equals(s3));  // true（内容相同）
```

**Q2: equals()和==的区别？**

`==`比较的是引用地址（基本类型比较值），`equals()`默认比较引用地址，但String、Integer等已重写equals()用于比较内容。自定义类需要重写equals()和hashCode()来实现内容比较。

```java
// equals和hashCode重写原则
public class Person {
    private String name;
    private int age;
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Person)) return false;
        Person person = (Person) o;
        return age == person.age && Objects.equals(name, person.name);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(name, age);
    }
    // hashCode约定：equals相等的对象，hashCode必须相等
    // hashCode相等的对象，equals不一定相等
}
```

**Q3: HashMap的底层实现原理？**

JDK 8中HashMap采用数组+链表+红黑树实现。通过key的hashCode经过扰动函数处理后确定在数组中的位置。当链表长度超过8且数组长度大于64时，链表转为红黑树（时间复杂度O(n)到O(log n)）。当红黑树节点数小于6时，转回链表。

```java
// HashMap的put流程
// 1. 计算hash值：h = key.hashCode() ^ (key.hashCode() >>> 16)
// 2. 定位桶位置：(n - 1) & hash
// 3. 桶为空 -> 直接放入
// 4. 桶不为空 -> 判断第一个元素是否相同 -> 相同则覆盖
// 5. 判断是否为红黑树节点 -> 是则执行红黑树插入
// 6. 遍历链表 -> 找到相同key则覆盖，否则尾插法插入（JDK 7是头插法）
// 7. 插入后判断是否需要扩容（size > threshold）

// 扩容机制
// 默认容量16，负载因子0.75，12个元素时触发扩容
// 扩容为原来的2倍，重新计算元素位置（高位判断法，效率高）
```

**Q4: ArrayList和LinkedList的区别及使用场景？**

| 特性 | ArrayList | LinkedList |
|------|-----------|------------|
| 底层数据结构 | Object[]数组 | 双向链表 |
| 随机访问 | O(1) | O(n) |
| 头部增删 | O(n) | O(1) |
| 尾部增删 | O(1)平均 | O(1) |
| 中间增删 | O(n) | O(1)（定位到节点后） |
| 内存占用 | 连续内存，尾部预留空间 | 每个节点额外存储前后指针 |
| 实现接口 | List、RandomAccess | List、Deque |

### 1.2 面向对象核心

**Q5: 重载(Overload)和重写(Override)的区别？**

- **重载**：同一类中方法名相同，参数列表不同（类型、个数、顺序），与返回值无关。编译时多态。
- **重写**：子类重写父类方法，方法签名完全相同，运行时多态。访问修饰符不能更严格，返回类型可以是父类返回类型的子类型（协变返回类型）。

**Q6: 抽象类和接口的区别？什么时候使用？**

| 区别点 | 抽象类 | 接口 |
|-------|--------|------|
| 关键字 | abstract class | interface |
| 构造方法 | 可以有 | 不可以有 |
| 成员变量 | 无限制 | 默认为public static final |
| 方法 | 可以有抽象和具体方法 | JDK 8+可以有default和static方法 |
| 继承 | 单继承 | 多实现 |
| 关系 | is-a | can-do / has-a |
| 设计层面 | 模板设计 | 行为规范 |

使用场景：抽象类适合有共同属性和行为的类族（如动物->猫、狗）；接口适合定义跨类族的能力（如飞行、游泳）。

## 二、并发编程面试题

**Q7: synchronized和Lock的区别？**

| 方面 | synchronized | Lock（ReentrantLock） |
|------|-------------|----------------------|
| 实现 | JVM层面，Monitor机制 | JDK层面，基于AQS |
| 释放 | 自动释放 | 手动释放（finally unlock） |
| 可中断 | 不支持 | 支持（lockInterruptibly） |
| 公平锁 | 非公平 | 两者都可 |
| 条件变量 | wait/notify | 多个Condition |
| 尝试获取 | 不支持 | tryLock() |
| Java版本 | 早期就有 | JDK 5引入 |

```java
// Condition实现生产者消费者模型
public class BoundedBuffer {
    private final Lock lock = new ReentrantLock();
    private final Condition notFull = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();
    private final Object[] buffer = new Object[100];
    private int putPtr, takePtr, count;
    
    public void put(Object x) throws InterruptedException {
        lock.lock();
        try {
            while (count == buffer.length) {
                notFull.await();  // 等待不满
            }
            buffer[putPtr] = x;
            if (++putPtr == buffer.length) putPtr = 0;
            count++;
            notEmpty.signal();  // 通知不空
        } finally {
            lock.unlock();
        }
    }
    
    public Object take() throws InterruptedException {
        lock.lock();
        try {
            while (count == 0) {
                notEmpty.await();
            }
            Object x = buffer[takePtr];
            if (++takePtr == buffer.length) takePtr = 0;
            count--;
            notFull.signal();
            return x;
        } finally {
            lock.unlock();
        }
    }
}
```

**Q8: volatile关键字的作用？**

volatile保证变量的**可见性**和**禁止指令重排序**，但不保证原子性。

- **可见性**：一个线程修改了volatile变量，修改会立即刷新到主内存，其他线程读取时从主内存获取最新值
- **禁止指令重排序**：通过内存屏障防止JVM对操作重排序，常用于单例DCL模式

```java
// VOLATILE的典型应用场景
// 1. 状态标记（不依赖当前值）
volatile boolean running = true;
// 线程1修改，线程2立即可见

// 2. DCL单例（防止指令重排序）
class Singleton {
    private volatile static Singleton instance;  // 必须有volatile
    // new Singleton()分三步：1.分配内存 2.初始化对象 3.引用指向内存
    // 2和3可能重排序，造成其他线程获取到未初始化对象
}

// 3. volatile不保证原子性的示例
volatile int count = 0;
// count++不是原子操作（读-改-写），多线程下会出错
// 应使用AtomicInteger或synchronized
```

**Q9: ThreadLocal实现原理及内存泄漏问题？**

ThreadLocal为每个线程提供独立的变量副本，底层通过ThreadLocalMap实现。每个Thread对象内部维护一个ThreadLocalMap，其Entry的key是ThreadLocal的弱引用。

```java
// ThreadLocal使用
public class UserContext {
    private static final ThreadLocal<User> currentUser = new ThreadLocal<>();
    
    public static void setUser(User user) {
        currentUser.set(user);
    }
    
    public static User getUser() {
        return currentUser.get();
    }
    
    // 关键：必须清理，否则内存泄漏
    public static void clear() {
        currentUser.remove();
    }
}

// 内存泄漏原因：
// ThreadLocalMap的Entry中key(ThreadLocal)是弱引用，value(用户对象)是强引用
// 当ThreadLocal对象被GC回收后，key变为null，但value无法被访问也无法回收
// 如果线程长期存在（如线程池），value会一直占用内存
// 解决方案：使用完后必须调用remove()
```

## 三、JVM面试题

**Q10: JVM内存模型和GC回收机制？**

（详细内容参考java_jvm.md文档）

核心考点：
- 堆内存分代：新生代（Eden + Survivor0 + Survivor1）、老年代
- 方法区/元空间：JDK 8后从永久代移到本地内存
- GC算法：标记-复制（新生代）、标记-整理（老年代）、标记-清除（CMS）
- GC回收器：Serial、Parallel、CMS、G1（JDK 9+默认）、ZGC（JDK 11+）
- GC Roots：栈帧中的引用、静态变量、常量、JNI引用
- 常见OOM场景：堆溢出、元空间溢出、直接内存溢出

```bash
# 如何排查OOM
# 1. 配置JVM参数自动dump
-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/var/log/heapdump.hprof
# 2. 使用MAT/JProfiler分析dump文件
# 3. 检查大对象（byte[]、String等）
# 4. 检查集合类无限增长
# 5. 检查ThreadLocal未清理
```

**Q11: 类加载机制和双亲委派模型？**

类加载的7个阶段：加载 -> 验证 -> 准备 -> 解析 -> 初始化 -> 使用 -> 卸载

双亲委派模型：Bootstrap ClassLoader -> Platform ClassLoader -> Application ClassLoader。首先委托父加载器加载，父加载器无法加载时才自己加载。目的是防止核心类库被篡改（如自定义java.lang.String不会被加载）。

```java
// 破坏双亲委派模型的场景
// 1. JDBC驱动加载（通过SPI机制，调用Thread Context ClassLoader）
// 2. Tomcat Web应用类加载（每个Web应用独立加载器，支持热部署）
// 3. OSGi模块化类加载
```

## 四、Spring面试题

**Q12: Spring Bean的生命周期？**

（详细参考java_spring.md文档）

核心阶段：实例化 -> 属性注入 -> Aware接口回调 -> BeanPostProcessor前置处理 -> @PostConstruct -> InitializingBean -> BeanPostProcessor后置处理（AOP代理创建） -> Bean就绪 -> @PreDestroy -> DisposableBean

**Q13: Spring事务传播行为和失效场景？**

事务传播行为（7种）：REQUIRED（默认）、REQUIRES_NEW、SUPPORTS、NOT_SUPPORTED、MANDATORY、NEVER、NESTED

@Transactional失效的6大场景：
1. 非public方法（Spring AOP基于代理，默认只代理public方法）
2. 同类内部方法调用（this.xxx()没有经过代理对象）
3. 异常被catch后未抛出
4. 配置了noRollbackFor的异常类型
5. 数据库引擎不支持事务（如MyISAM）
6. 多线程环境中（事务绑定在线程上）

```java
// 解决同类方法调用失效
@Service
public class ServiceA {
    @Autowired
    private ServiceA self;  // 注入自己的代理对象
    
    @Transactional
    public void methodA() {
        self.methodB();  // 通过代理调用，事务生效
    }
    
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void methodB() {
        // 新事务
    }
}
```

**Q14: Spring AOP的实现原理？**

Spring AOP基于动态代理实现：
- **JDK动态代理**：目标类必须实现接口，通过Proxy和InvocationHandler
- **CGLIB代理**：通过继承目标类生成代理子类，不能代理final类和方法

Spring Boot 2.x默认使用CGLIB代理（`spring.aop.proxy-target-class=true`）。

## 五、数据库与Redis面试题

**Q15: MySQL索引类型及优化策略？**

```sql
-- 索引类型
-- 1. B+树索引（最常用）
-- 2. 哈希索引（Memory引擎，等值查询快）
-- 3. 全文索引（InnoDB 5.6+支持）
-- 4. 空间索引（地理位置数据）

-- 联合索引最左前缀原则
-- 创建联合索引 (a, b, c)
-- 以下查询能使用索引：WHERE a=1、WHERE a=1 AND b=2、WHERE a=1 AND b=2 AND c=3
-- 以下不能使用：WHERE b=2、WHERE c=3、WHERE a=1 AND c=3（c会失效，只有a能用）

-- 索引优化策略
-- 1. 选择区分度高的列建索引（如唯一ID优于性别）
-- 2. 避免在索引列上使用函数或运算（WHERE YEAR(date)=2024 不会走索引）
-- 3. 使用覆盖索引避免回表查询
-- 4. 遵守最左前缀原则
-- 5. 避免创建过多索引（影响写入性能）
```

**Q16: Redis缓存穿透、击穿、雪崩？**

- **缓存穿透**：查询不存在的数据，请求直达数据库。解决方案：布隆过滤器、缓存空值（设置短过期时间）
- **缓存击穿**：热点key过期，大量请求直达数据库。解决方案：分布式锁、永不过期（后台异步更新）
- **缓存雪崩**：大量key同时过期。解决方案：过期时间加随机值、多级缓存、限流降级

```java
// 缓存击穿解决方案：分布式锁
public String getData(String key) {
    String value = redis.get(key);
    if (value != null) return value;
    
    String lockKey = "lock:" + key;
    try {
        if (redis.setnx(lockKey, "1", 30, TimeUnit.SECONDS)) {
            value = db.query(key);
            redis.set(key, value, 60, TimeUnit.SECONDS);
        } else {
            Thread.sleep(100);
            return getData(key);  // 重试
        }
    } finally {
        redis.del(lockKey);
    }
    return value;
}
```

**Q17: 缓存与数据库一致性方案？**

- **Cache Aside Pattern**（旁路缓存模式，最常用）：先更新数据库，再删除缓存
- **Update Immediately**：更新数据库 + 更新缓存（并发问题多，不推荐）
- **双写+过期时间**：设置合理过期时间兜底
- **延迟双删**：先删缓存 -> 更新数据库 -> 延迟删除缓存
- **Canal + MQ**：通过binlog异步更新缓存（最终一致性）

## 六、分布式系统面试题

**Q18: 如何保证分布式系统数据一致性？**

- **强一致性**：2PC（两阶段提交）、3PC（三阶段提交）
- **最终一致性**：Base理论、TCC事务（Try-Confirm-Cancel）、Saga模式
- **CAP定理**：一致性、可用性、分区容错性三者最多满足两者。分布式系统必须保证P（分区容错性），在C和A之间平衡
- **分布式锁**：Redis（SETNX + Lua）、Zookeeper（临时顺序节点）、Redisson（看门狗自动续期）

```java
// Redisson分布式锁使用
@Service
public class OrderService {
    @Autowired
    private RedissonClient redisson;
    
    public void createOrder(Long userId, Long productId) {
        String lockKey = "lock:order:" + userId;
        RLock lock = redisson.getLock(lockKey);
        
        try {
            // 尝试加锁，等待10秒，锁30秒自动释放（看门狗会自动续期）
            if (lock.tryLock(10, 30, TimeUnit.SECONDS)) {
                // 业务逻辑
                checkStock(productId);
                createOrderRecord(userId, productId);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            if (lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }
}
```

**Q19: 消息队列的使用场景和选型？**

| 场景 | 推荐MQ | 原因 |
|------|--------|------|
| 高吞吐量 | Kafka | 顺序写入，零拷贝 |
| 低延迟 | RocketMQ | 阿里系，事务消息完善 |
| 简单解耦 | RabbitMQ | 生态成熟，管理方便 |
| 流计算 | Kafka | 搭配Flink/Spark Streaming |

常见应用场景：异步处理、应用解耦、流量削峰、日志收集、事件驱动。

## 七、总结

Java面试涵盖的范围非常广，从基础语法到高并发、JVM调优、分布式架构。除了理论知识外，面试官通常还会考察：
1. **实际问题解决能力**：让你描述排查过的线上问题及解决方案
2. **系统设计能力**：如何设计一个高并发秒杀系统、短链接服务等
3. **代码阅读能力**：给出有问题的代码，让你找出bug
4. **技术广度与深度**：对自己擅长的领域要有深入理解

建议采用"理论+实践+系统"三位一体的学习方法，每个知识点都要能讲出"是什么、为什么、怎么用、有什么坑"。
