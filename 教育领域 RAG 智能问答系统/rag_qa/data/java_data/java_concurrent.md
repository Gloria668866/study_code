# Java并发编程：线程池、锁机制与JUC工具类

## 一、Java并发编程基础

### 1.1 线程基础

Java从诞生之初就内置了对多线程的支持。线程是操作系统调度的最小单位，Java中的线程通过`java.lang.Thread`类来表示。

```java
// 创建线程的三种方式

// 方式1：继承Thread类
public class MyThread extends Thread {
    @Override
    public void run() {
        System.out.println("线程运行: " + Thread.currentThread().getName());
    }
}
new MyThread().start();

// 方式2：实现Runnable接口（推荐，更灵活）
public class MyRunnable implements Runnable {
    @Override
    public void run() {
        System.out.println("线程运行: " + Thread.currentThread().getName());
    }
}
new Thread(new MyRunnable()).start();

// 方式3：实现Callable接口（有返回值，可抛异常）
public class MyCallable implements Callable<String> {
    @Override
    public String call() throws Exception {
        Thread.sleep(1000);
        return "任务完成";
    }
}
FutureTask<String> futureTask = new FutureTask<>(new MyCallable());
new Thread(futureTask).start();
String result = futureTask.get();  // 阻塞获取结果
```

### 1.2 线程的生命周期

Java线程有6种状态，定义在`Thread.State`枚举中：

| 状态 | 说明 | 进入方式 |
|------|------|---------|
| NEW | 初始状态，线程创建但未启动 | new Thread() |
| RUNNABLE | 运行中（包括就绪和运行） | thread.start() |
| BLOCKED | 阻塞状态，等待获取锁 | 进入synchronized块 |
| WAITING | 无限等待状态 | wait()、join()、LockSupport.park() |
| TIMED_WAITING | 限时等待状态 | sleep()、wait(timeout)、join(timeout) |
| TERMINATED | 终止状态 | run()执行完毕或异常退出 |

```java
// 线程状态切换示例
public class ThreadStateDemo {
    public static void main(String[] args) throws InterruptedException {
        Thread thread = new Thread(() -> {
            try {
                Thread.sleep(2000);  // TIMED_WAITING
                synchronized (ThreadStateDemo.class) {
                    ThreadStateDemo.class.wait();  // WAITING
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });
        
        System.out.println(thread.getState());  // NEW
        thread.start();
        System.out.println(thread.getState());  // RUNNABLE
        Thread.sleep(500);
        System.out.println(thread.getState());  // TIMED_WAITING
    }
}
```

## 二、线程池详解

### 2.1 为什么使用线程池

线程池的核心优势：
- **降低资源消耗**：复用已创建的线程，减少创建和销毁线程的开销
- **提高响应速度**：任务到达时无需等待线程创建
- **提高线程可管理性**：统一分配、调优和监控
- **防止资源耗尽**：限制线程数量，避免无限制创建线程

### 2.2 ThreadPoolExecutor核心参数

```java
/**
 * 线程池构造方法完整参数
 */
public ThreadPoolExecutor(int corePoolSize,          // 核心线程数
                          int maximumPoolSize,        // 最大线程数
                          long keepAliveTime,         // 空闲线程存活时间
                          TimeUnit unit,              // 时间单位
                          BlockingQueue<Runnable> workQueue,  // 工作队列
                          ThreadFactory threadFactory,        // 线程工厂
                          RejectedExecutionHandler handler) {  // 拒绝策略
```

**核心参数详解**：

- **corePoolSize**：核心线程数，即使空闲也保留在线程池中（除非设置了allowCoreThreadTimeOut）
- **maximumPoolSize**：最大线程数，工作队列满且核心线程都在忙时，新任务会创建新线程直到达到此值
- **keepAliveTime**：非核心线程空闲等待新任务的最长时间，超时后销毁
- **workQueue**：任务队列，存储等待执行的任务。常见选择：
  - `SynchronousQueue`：不存储任务，直接提交给线程（建议搭配较大maximumPoolSize）
  - `LinkedBlockingQueue`：无界队列，任务无限排队（可能导致OOM）
  - `ArrayBlockingQueue`：有界队列，队列满时触发拒绝策略
  - `PriorityBlockingQueue`：优先级队列
- **handler**：拒绝策略

### 2.3 线程池工作流程

任务提交到线程池后的处理流程：

1. 如果当前线程数 < corePoolSize，创建新线程执行任务
2. 如果当前线程数 >= corePoolSize 且工作队列未满，任务加入工作队列
3. 如果工作队列已满 且 当前线程数 < maximumPoolSize，创建新线程执行任务
4. 如果工作队列已满 且 当前线程数 >= maximumPoolSize，执行拒绝策略

### 2.4 拒绝策略

| 策略 | 说明 |
|------|------|
| AbortPolicy（默认） | 抛出RejectedExecutionException |
| CallerRunsPolicy | 由调用者线程执行该任务（反馈机制） |
| DiscardPolicy | 直接丢弃任务，不抛异常 |
| DiscardOldestPolicy | 丢弃队列中最旧的任务，然后重新提交 |

```java
// 线程池最佳实践配置
int cpuCores = Runtime.getRuntime().availableProcessors();

// CPU密集型任务：线程数 = CPU核心数 + 1
ExecutorService cpuPool = new ThreadPoolExecutor(
    cpuCores + 1,
    cpuCores + 1,
    60L, TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(1000),
    new ThreadPoolExecutor.CallerRunsPolicy()
);

// IO密集型任务：线程数 = CPU核心数 * 2（或CPU核心数 / (1 - 阻塞系数)）
ExecutorService ioPool = new ThreadPoolExecutor(
    cpuCores * 2,
    cpuCores * 4,
    60L, TimeUnit.SECONDS,
    new LinkedBlockingQueue<>(10000),
    new ThreadFactoryBuilder().setNameFormat("io-pool-%d").build(),
    new ThreadPoolExecutor.CallerRunsPolicy()
);
```

### 2.5 Executors工厂类及其陷阱

```java
// Executors提供的便捷工厂方法（但阿里巴巴开发手册不推荐使用）
// 陷阱1：FixedThreadPool使用无界队列，可能导致OOM
ExecutorService fixedPool = Executors.newFixedThreadPool(10);
// 内部：new ThreadPoolExecutor(n, n, 0L, TimeUnit.MILLISECONDS,
//                               new LinkedBlockingQueue<Runnable>());  // 无界！

// 陷阱2：CachedThreadPool允许创建无数线程，可能导致OOM
ExecutorService cachedPool = Executors.newCachedThreadPool();
// 内部：new ThreadPoolExecutor(0, Integer.MAX_VALUE, 60L, ...);

// 推荐：手动创建线程池，精确控制参数
ExecutorService pool = new ThreadPoolExecutor(
    10, 20, 60L, TimeUnit.SECONDS,
    new ArrayBlockingQueue<>(500),
    new ThreadFactoryBuilder().setNameFormat("biz-pool-%d").build(),
    new ThreadPoolExecutor.CallerRunsPolicy()
);
```

### 2.6 线程池监控

```java
public class MonitoredThreadPool extends ThreadPoolExecutor {
    
    public void printStatus() {
        System.out.println("=== 线程池状态 ===");
        System.out.println("核心线程数: " + getCorePoolSize());
        System.out.println("最大线程数: " + getMaximumPoolSize());
        System.out.println("当前活跃线程: " + getActiveCount());
        System.out.println("当前池大小: " + getPoolSize());
        System.out.println("历史最大池大小: " + getLargestPoolSize());
        System.out.println("已完成任务数: " + getCompletedTaskCount());
        System.out.println("总任务数: " + getTaskCount());
        System.out.println("队列大小: " + getQueue().size());
        System.out.println("队列剩余容量: " + getQueue().remainingCapacity());
    }
}
```

## 三、锁机制详解

### 3.1 synchronized关键字

`synchronized`是Java内置的同步机制，基于JVM的Monitor对象实现。

```java
public class SynchronizedDemo {
    
    // 1. 修饰实例方法：锁是当前实例对象
    public synchronized void instanceMethod() {
        // 同一实例的该方法在同一时刻只能有一个线程执行
    }
    
    // 2. 修饰静态方法：锁是当前类的Class对象
    public static synchronized void staticMethod() {
        // 该类的所有实例共享此锁
    }
    
    // 3. 修饰代码块：指定锁对象
    private final Object lock = new Object();
    
    public void blockMethod() {
        synchronized (lock) {
            // 精确控制同步范围，减少锁粒度
        }
    }
}
```

**synchronized锁升级过程（JDK 6+）**：
- **无锁**（刚创建时状态）
  - 当第一个线程访问时，进入偏向锁
- **偏向锁**（Biased Locking）
  - 线程ID记录在对象头中，该线程后续进入无需CAS操作
  - 当另一个线程尝试获取锁时，偏向锁撤销，升级为轻量级锁
- **轻量级锁**（Lightweight Locking）
  - 通过CAS操作和自旋获取锁，不阻塞线程
  - 自旋失败一定次数后，升级为重量级锁
- **重量级锁**（Heavyweight Locking）
  - 线程阻塞，进入等待队列，需要操作系统的Mutex Lock

### 3.2 Lock接口与ReentrantLock

`java.util.concurrent.locks.Lock`接口提供了比synchronized更灵活的锁操作。

```java
public class LockDemo {
    private final ReentrantLock lock = new ReentrantLock();
    private int count = 0;
    
    // 标准用法：lock() + try-finally
    public void increment() {
        lock.lock();
        try {
            count++;
        } finally {
            lock.unlock();  // 必须在finally中释放锁
        }
    }
    
    // 公平锁：按等待时间顺序获取锁（默认非公平）
    private final ReentrantLock fairLock = new ReentrantLock(true);
    
    // tryLock：尝试获取锁，获取不到立即返回
    public boolean tryIncrement() {
        try {
            if (lock.tryLock(1, TimeUnit.SECONDS)) {  // 等待1秒
                try {
                    count++;
                    return true;
                } finally {
                    lock.unlock();
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return false;
    }
    
    // lockInterruptibly：可中断的锁获取
    public void interruptibleLock() throws InterruptedException {
        lock.lockInterruptibly();  // 可被Thread.interrupt()中断
        try {
            // 业务逻辑
        } finally {
            lock.unlock();
        }
    }
}
```

**synchronized vs ReentrantLock**：

| 特性 | synchronized | ReentrantLock |
|------|-------------|---------------|
| 实现方式 | JVM层面，基于Monitor | JDK层面，基于AQS |
| 锁释放 | 自动释放（代码块退出或异常） | 手动释放（finally中unlock） |
| 可中断 | 不可中断 | 可中断（lockInterruptibly） |
| 公平锁 | 非公平 | 可公平可非公平（构造器参数） |
| 条件变量 | 单一条件（wait/notify） | 多条件（Condition） |
| 性能 | JDK 6优化后两者接近 | - |
| 选择建议 | 简单同步场景 | 需要高级特性时 |

### 3.3 ReadWriteLock读写锁

读写锁允许多个读线程同时访问，但写线程独占访问，提高了读多写少场景的并发性能。

```java
public class CacheService {
    private final ReadWriteLock rwLock = new ReentrantReadWriteLock();
    private final Map<String, Object> cache = new HashMap<>();
    
    // 读操作使用读锁，允许并发读取
    public Object get(String key) {
        rwLock.readLock().lock();
        try {
            return cache.get(key);
        } finally {
            rwLock.readLock().unlock();
        }
    }
    
    // 写操作使用写锁，独占访问
    public void put(String key, Object value) {
        rwLock.writeLock().lock();
        try {
            cache.put(key, value);
        } finally {
            rwLock.writeLock().unlock();
        }
    }
    
    // 降级：写锁降级为读锁
    public Object getAndClear(String key) {
        rwLock.writeLock().lock();
        try {
            Object value = cache.remove(key);
            rwLock.readLock().lock();  // 获取读锁
            // 释放写锁后仍持有读锁，保证了数据一致性
            return value;
        } finally {
            rwLock.writeLock().unlock();
            try {
                // 使用读锁做其他操作
            } finally {
                rwLock.readLock().unlock();
            }
        }
    }
}
```

### 3.4 死锁与解决方案

```java
// 死锁示例：两个线程互相等待对方释放锁
public class DeadLockDemo {
    private static final Object lock1 = new Object();
    private static final Object lock2 = new Object();
    
    public static void main(String[] args) {
        new Thread(() -> {
            synchronized (lock1) {
                sleep(100);
                synchronized (lock2) {  // 等待lock2
                    System.out.println("Thread1 got both locks");
                }
            }
        }).start();
        
        new Thread(() -> {
            synchronized (lock2) {
                sleep(100);
                synchronized (lock1) {  // 等待lock1
                    System.out.println("Thread2 got both locks");
                }
            }
        }).start();
    }
}

// 解决方案1：固定加锁顺序
public void safeMethod() {
    synchronized (lock1) {
        synchronized (lock2) {
            // 所有线程都以相同顺序获取锁
        }
    }
}

// 解决方案2：使用tryLock设置超时时间
public void tryLockMethod() {
    while (true) {
        if (lock1.tryLock()) {
            try {
                if (lock2.tryLock(1, TimeUnit.SECONDS)) {
                    try { return; }
                    finally { lock2.unlock(); }
                }
            } finally {
                lock1.unlock();
            }
        }
        Thread.sleep(ThreadLocalRandom.current().nextInt(100));
    }
}
```

## 四、JUC工具类

### 4.1 CountDownLatch

计数器，允许一个或多个线程等待其他线程完成操作。

```java
// 场景：等待所有子任务完成后，主线程再执行
public class CountDownLatchDemo {
    public void process(int taskCount) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(taskCount);
        ExecutorService executor = Executors.newFixedThreadPool(5);
        
        for (int i = 0; i < taskCount; i++) {
            final int taskId = i;
            executor.submit(() -> {
                try {
                    doTask(taskId);
                } finally {
                    latch.countDown();  // 完成一个任务，计数减1
                }
            });
        }
        
        latch.await();  // 等待所有任务完成
        System.out.println("所有任务完成");
        executor.shutdown();
    }
}
```

### 4.2 CyclicBarrier

可循环使用的屏障，让一组线程达到某个屏障点时互相等待。

```java
// 场景：多个线程分段计算，每段完成后等待其他线程一起进入下一段
public class CyclicBarrierDemo {
    private final CyclicBarrier barrier = new CyclicBarrier(3, () -> {
        System.out.println("所有线程到达屏障点，执行汇总操作");
    });
    
    public void startThreads() {
        for (int i = 0; i < 3; i++) {
            final int threadId = i;
            new Thread(() -> {
                for (int round = 0; round < 5; round++) {
                    doWork(threadId, round);
                    try {
                        barrier.await();  // 等待其他线程
                    } catch (Exception e) {
                        Thread.currentThread().interrupt();
                    }
                }
            }).start();
        }
    }
}
```

### 4.3 Semaphore

信号量，用于控制同时访问某个资源的线程数量。

```java
// 场景：限流，控制数据库连接并发数
public class ConnectionPool {
    private final Semaphore semaphore = new Semaphore(10);  // 最多10个并发连接
    
    public Connection getConnection() throws InterruptedException {
        semaphore.acquire();  // 获取许可，获取不到则阻塞
        try {
            return createConnection();
        } catch (Exception e) {
            semaphore.release();  // 异常时释放许可
            throw e;
        }
    }
    
    public void releaseConnection(Connection conn) {
        closeConnection(conn);
        semaphore.release();  // 归还许可
    }
}
```

### 4.4 CompletableFuture

Java 8引入的异步编程利器，提供强大的Future增强功能。

```java
public class CompletableFutureDemo {
    
    // 异步执行
    public void asyncDemo() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            sleep(1000);
            return "结果";
        });
        
        // 回调处理
        future.thenAccept(result -> System.out.println("得到结果: " + result));
        // 不阻塞主线程
    }
    
    // 组合多个异步任务
    public void combineDemo() {
        CompletableFuture<String> task1 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> task2 = CompletableFuture.supplyAsync(() -> "World");
        
        // 等待两个任务都完成
        CompletableFuture<String> combined = task1.thenCombine(task2, (r1, r2) -> r1 + " " + r2);
        
        // 任一任务完成即返回
        CompletableFuture<Object> any = CompletableFuture.anyOf(task1, task2);
    }
    
    // 异常处理
    public void exceptionDemo() {
        CompletableFuture.supplyAsync(() -> {
            if (Math.random() > 0.5) throw new RuntimeException("随机异常");
            return "成功";
        }).exceptionally(ex -> {
            System.err.println("异常: " + ex.getMessage());
            return "降级值";
        }).thenAccept(System.out::println);
    }
    
    // 链式调用
    public void chainDemo() {
        CompletableFuture.supplyAsync(() -> getUserId("张三"))
            .thenApplyAsync(id -> getUserInfo(id))
            .thenApplyAsync(info -> enrichInfo(info))
            .thenAcceptAsync(result -> saveResult(result))
            .whenComplete((v, ex) -> {
                if (ex != null) System.err.println("处理异常: " + ex.getMessage());
            });
    }
}
```

### 4.5 ConcurrentHashMap

线程安全的HashMap，JDK 8后采用CAS+synchronized实现。

```java
// ConcurrentHashMap的常用方法
ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();

// putIfAbsent：仅当key不存在时放入
map.putIfAbsent("key", 1);

// computeIfAbsent：如果key不存在则计算value并放入
map.computeIfAbsent("counter", k -> 0);

// compute：原子地更新value
map.compute("counter", (k, v) -> v == null ? 1 : v + 1);

// merge：合并value
map.merge("key", 1, Integer::sum);

// 搜索（并行操作）
String result = map.search(100, (k, v) -> v > 10 ? k : null);

// reduce（聚合操作）
int total = map.reduceValues(100, Integer::sum);
```

## 五、并发编程最佳实践

1. **优先使用不可变对象**：不可变对象天然线程安全（如String、LocalDate）
2. **减小锁粒度**：使用分段锁（如ConcurrentHashMap早期版本）或细粒度锁
3. **避免在锁内执行耗时操作**：减少锁持有时间
4. **使用并发集合替代手动同步**：ConcurrentHashMap > Collections.synchronizedMap(new HashMap<>())
5. **正确使用ThreadLocal**：记得在finally中remove，避免内存泄漏
6. **线程池命名**：给线程池起有意义的名字，方便排查问题
7. **合理设置线程数**：CPU密集型 N+1，IO密集型 2N
8. **使用JUC工具类而非自己实现**：CountDownLatch、Semaphore等都已经过严格测试

## 六、总结

Java并发编程是高级开发者的必备技能。理解线程池的工作原理和参数配置，掌握synchronized和Lock的适用场景，熟练使用JUC工具类，能够编写出高效、安全的并发程序。在实际项目中，应优先使用现有的并发工具类，避免重复造轮子，同时注意避免死锁、内存泄漏等常见并发陷阱。
