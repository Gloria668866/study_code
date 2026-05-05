# Java 8-21新特性：Lambda、Stream、模块化系统与现代化Java

## 一、Java版本演进概述

Java经历了重大变革，从JDK 8到JDK 21的演进带来了函数式编程、模块化系统、记录类、模式匹配、虚拟线程等革命性特性。以下是各主要版本的里程碑特性：

| 版本 | 发布年份 | 核心特性 |
|------|---------|---------|
| Java 8 | 2014.03 | Lambda、Stream、Optional、新日期API |
| Java 9 | 2017.09 | 模块化系统(JPMS)、JShell、集合工厂方法 |
| Java 10 | 2018.03 | 局部变量类型推断(var) |
| Java 11 | 2018.09 | LTS版本、HTTP Client API、字符串增强 |
| Java 14 | 2020.03 | Records（预览）、Switch表达式 |
| Java 16 | 2021.03 | Records正式版、Pattern Matching for instanceof |
| Java 17 | 2021.09 | LTS版本、Sealed Classes |
| Java 19 | 2022.09 | 虚拟线程（预览） |
| Java 21 | 2023.09 | LTS版本、虚拟线程正式版、Record Patterns |

## 二、Lambda表达式与函数式接口

### 2.1 Lambda表达式基础

Lambda表达式本质上是一个匿名函数，允许将函数作为方法参数传递。它是函数式编程在Java中的核心实现。

```java
// Lambda表达式语法：(参数列表) -> { 方法体 }

// 演进过程
// 1. 传统匿名内部类
Runnable r1 = new Runnable() {
    @Override
    public void run() {
        System.out.println("Hello");
    }
};

// 2. Lambda表达式
Runnable r2 = () -> System.out.println("Hello");

// 3. 方法引用
Runnable r3 = System.out::println;

// Lambda的各种形式
Consumer<String> c1 = (String s) -> System.out.println(s);  // 完整形式
Consumer<String> c2 = (s) -> System.out.println(s);         // 省略类型
Consumer<String> c3 = s -> System.out.println(s);           // 单参数省略括号

Comparator<Integer> comp1 = (a, b) -> { return a - b; };   // 多行方法体
Comparator<Integer> comp2 = (a, b) -> a - b;                // 单行省略return
```

### 2.2 函数式接口

函数式接口是只包含一个抽象方法的接口（SAM - Single Abstract Method），使用`@FunctionalInterface`注解标注。

```java
// Java内置四大函数式接口

// 1. Function<T, R>：接收T返回R（转换）
Function<String, Integer> strLength = String::length;
Function<Integer, String> intToStr = i -> "数字: " + i;
String result = intToStr.apply(42);  // "数字: 42"

// compose（先应用before，再应用当前）
Function<Integer, Integer> doubleIt = n -> n * 2;
Function<Integer, Integer> square = n -> n * n;
Function<Integer, Integer> composed = square.compose(doubleIt);  // square(doubleIt(n))
composed.apply(3);  // 先double(3)=6, 再square(6)=36

// andThen（先应用当前，再应用after）
Function<Integer, Integer> andThen = doubleIt.andThen(square);  // doubleIt(square(n))
andThen.apply(3);  // 先square(3)=9, 再doubleIt(9)=18

// 2. Consumer<T>：接收T无返回（消费）
Consumer<String> print = System.out::println;
Consumer<String> log = s -> log.info(s);
Consumer<String> printAndLog = print.andThen(log);  // 链式执行

// 3. Supplier<T>：无参数返回T（提供）
Supplier<LocalDateTime> now = LocalDateTime::now;
Supplier<Double> random = Math::random;

// 4. Predicate<T>：接收T返回boolean（判断）
Predicate<String> isEmpty = String::isEmpty;
Predicate<String> isLong = s -> s.length() > 10;
Predicate<String> isLongAndNotEmpty = isEmpty.negate().and(isLong);  // 组合

// 自定义函数式接口
@FunctionalInterface
public interface ThrowingFunction<T, R> {
    R apply(T t) throws Exception;
    
    static <T, R> Function<T, R> wrap(ThrowingFunction<T, R> func) {
        return t -> {
            try {
                return func.apply(t);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }
}
```

### 2.3 方法引用

方法引用是Lambda表达式的简洁写法，有四种形式：

```java
// 1. 静态方法引用：ClassName::staticMethod
Function<String, Integer> parseInt = Integer::parseInt;

// 2. 实例方法引用（特定对象）：instance::method
String prefix = "Hello ";
Function<String, String> concat = prefix::concat;

// 3. 实例方法引用（任意对象）：ClassName::instanceMethod
// 第一个参数作为调用对象
Function<String, String> toLower = String::toLowerCase;
// 等价于 s -> s.toLowerCase()

// 4. 构造方法引用：ClassName::new
Supplier<List<String>> listFactory = ArrayList::new;  // 无参构造
Function<Integer, List<String>> sizedList = ArrayList::new;  // 带参构造
```

## 三、Stream API

Stream API是Java 8引入的处理集合数据的声明式编程方式，支持链式操作和并行处理。

### 3.1 Stream创建

```java
// 从集合创建
Stream<String> stream1 = list.stream();
Stream<String> parallelStream = list.parallelStream();

// 从数组创建
Stream<String> stream2 = Arrays.stream(array);
Stream<Integer> stream3 = Stream.of(1, 2, 3, 4, 5);

// 无限流
Stream<Double> randoms = Stream.generate(Math::random);
Stream<Integer> iterate = Stream.iterate(0, n -> n + 2);  // 0,2,4,6...

// 范围流
IntStream range = IntStream.range(0, 10);      // 0-9
IntStream rangeClosed = IntStream.rangeClosed(0, 10);  // 0-10

// 从文件
try (Stream<String> lines = Files.lines(Path.of("file.txt"))) {
    lines.filter(line -> !line.isEmpty()).forEach(System.out::println);
}
```

### 3.2 中间操作

```java
// filter：过滤
list.stream()
    .filter(s -> s.length() > 3)
    .filter(s -> s.startsWith("A"))
    .collect(Collectors.toList());

// map：转换元素
list.stream()
    .map(String::toUpperCase)
    .map(String::trim)
    .collect(Collectors.toList());

// flatMap：将多个流合并为一个流（扁平化）
List<List<String>> nested = Arrays.asList(
    Arrays.asList("a", "b"),
    Arrays.asList("c", "d")
);
List<String> flat = nested.stream()
    .flatMap(Collection::stream)
    .collect(Collectors.toList());  // [a, b, c, d]

// distinct：去重（基于equals）
list.stream().distinct().collect(Collectors.toList());

// sorted：排序
list.stream()
    .sorted(Comparator.comparing(String::length)
                      .thenComparing(Comparator.naturalOrder()))
    .collect(Collectors.toList());

// limit / skip：限制/跳过
list.stream().skip(5).limit(10).collect(Collectors.toList());

// peek：调试（查看中间结果）
list.stream()
    .peek(s -> System.out.println("处理后: " + s))
    .collect(Collectors.toList());

// mapToInt / mapToLong / mapToDouble：转换为基本类型流
int sum = list.stream()
    .mapToInt(String::length)
    .sum();
```

### 3.3 终端操作

```java
// collect：收集器（最常用）
// 转List
List<String> toList = stream.collect(Collectors.toList());
// 转Set
Set<String> toSet = stream.collect(Collectors.toSet());
// 转Map（注意key重复会抛异常）
Map<Long, User> toMap = stream.collect(
    Collectors.toMap(User::getId, Function.identity())
);
// 转Map（处理重复key）
Map<String, List<User>> groupBy = stream.collect(
    Collectors.toMap(
        User::getDept,
        Function.identity(),
        (existing, replacement) -> existing  // 保留前者
    )
);

// 分组
Map<String, List<User>> grouped = users.stream()
    .collect(Collectors.groupingBy(User::getDept));

// 多级分组
Map<String, Map<Integer, List<User>>> multiGroup = users.stream()
    .collect(Collectors.groupingBy(User::getDept,
             Collectors.groupingBy(User::getAge)));

// 分区（分为true/false两组）
Map<Boolean, List<User>> partitioned = users.stream()
    .collect(Collectors.partitioningBy(u -> u.getAge() >= 18));

// joining：字符串连接
String names = users.stream()
    .map(User::getName)
    .collect(Collectors.joining(", ", "[", "]"));  // [Alice, Bob, Charlie]

// 聚合
long count = stream.count();
String max = stream.max(Comparator.naturalOrder()).orElse(null);
int sum = stream.mapToInt(Integer::intValue).sum();

// reduce：归约
Optional<Integer> product = numbers.stream()
    .reduce((a, b) -> a * b);  // 累积乘法

Integer sumWithIdentity = numbers.stream()
    .reduce(0, Integer::sum);  // 有初始值

// findFirst / findAny
Optional<String> first = stream.findFirst();
Optional<String> any = stream.findAny();  // 并行流更快

// allMatch / anyMatch / noneMatch
boolean allAdult = users.stream().allMatch(u -> u.getAge() >= 18);
```

### 3.4 Collectors进阶用法

```java
// summarizing（统计）
IntSummaryStatistics stats = users.stream()
    .collect(Collectors.summarizingInt(User::getAge));
System.out.println("平均年龄: " + stats.getAverage());
System.out.println("最大年龄: " + stats.getMax());

// mapping（嵌套映射）
Map<String, List<String>> deptNames = users.stream()
    .collect(Collectors.groupingBy(User::getDept,
             Collectors.mapping(User::getName, Collectors.toList())));

// collectingAndThen（收集后再转换）
List<User> unmodifiableList = users.stream()
    .collect(Collectors.collectingAndThen(
        Collectors.toList(),
        Collections::unmodifiableList
    ));

// teeing（JDK 12+）：同时执行两个收集器并合并结果
double avgSalary = employees.stream()
    .collect(Collectors.teeing(
        Collectors.summingDouble(Employee::getSalary),
        Collectors.counting(),
        (sum, count) -> sum / count
    ));
```

### 3.5 并行流

```java
// 使用并行流（注意线程安全问题）
// ArrayList、数组适合并行；LinkedList不适合（难以分割）
List<Integer> result = list.parallelStream()
    .map(this::expensiveOperation)
    .collect(Collectors.toList());  // collect是线程安全的

// 避免在并行流中使用有状态操作
// 错误示例
List<Integer> values = new ArrayList<>();
list.parallelStream()
    .forEach(values::add);  // ArrayList非线程安全！
// 正确示例
List<Integer> values = list.parallelStream()
    .collect(Collectors.toList());
```

## 四、Optional类

Optional是一个容器类，代表一个值存在或不存在，用于避免NullPointerException。

```java
// 创建Optional
Optional<String> opt1 = Optional.of("Hello");     // 值不能为null
Optional<String> opt2 = Optional.ofNullable(null); // 值可为null
Optional<String> opt3 = Optional.empty();          // 空Optional

// 获取值
String value = opt1.get();              // 为null时抛NoSuchElementException
String value2 = opt1.orElse("默认值");   // 为null时返回默认值
String value3 = opt1.orElseGet(() -> computeDefault());  // 延迟计算默认值
String value4 = opt1.orElseThrow(() -> new BusinessException("值不存在"));

// 条件操作
opt1.ifPresent(v -> System.out.println("存在: " + v));
opt1.ifPresentOrElse(
    v -> System.out.println("存在: " + v),
    () -> System.out.println("不存在")
);

// 链式操作
String result = Optional.ofNullable(user)
    .map(User::getAddress)
    .map(Address::getCity)
    .filter(city -> !city.isEmpty())
    .orElse("未知城市");

// flatMap用于返回Optional的方法
String deptName = Optional.ofNullable(user)
    .flatMap(User::getDepartment)    // getDepartment返回Optional<Department>
    .map(Department::getName)
    .orElse("未分配部门");

// Optional使用原则
// 1. 不要用Optional作为字段类型
// 2. 不要用Optional作为方法参数
// 3. 主要用于方法的返回类型
// 4. 避免Optional.of(null)，始终使用Optional.ofNullable
// 5. 使用isPresent()不如使用orElse/orElseGet
```

## 五、新日期API（java.time）

Java 8引入了全新的日期时间API，解决了旧版Date和Calendar的线程安全问题和使用复杂性。

```java
// 三大核心类
LocalDate date = LocalDate.now();           // 2024-01-15
LocalTime time = LocalTime.now();           // 14:30:00.123
LocalDateTime dateTime = LocalDateTime.now();  // 2024-01-15T14:30:00.123

// 创建指定日期时间
LocalDate specificDate = LocalDate.of(2024, 1, 15);
LocalDate parseDate = LocalDate.parse("2024-01-15");

// 日期运算（返回新对象，不可变）
LocalDate tomorrow = LocalDate.now().plusDays(1);
LocalDate lastMonth = LocalDate.now().minusMonths(1);
LocalDate nextMonday = LocalDate.now().with(TemporalAdjusters.next(DayOfWeek.MONDAY));

// 时间比较
boolean isAfter = date1.isAfter(date2);
boolean isBefore = date1.isBefore(date2);
long daysBetween = ChronoUnit.DAYS.between(date1, date2);

// Duration（时间间隔）和 Period（日期间隔）
Duration duration = Duration.between(time1, time2);
Period period = Period.between(date1, date2);

// 格式化
DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
String formatted = dateTime.format(formatter);
LocalDateTime parsed = LocalDateTime.parse("2024-01-15 14:30:00", formatter);

// 时区处理
ZonedDateTime zonedDateTime = ZonedDateTime.now();  // 带时区
ZoneId shanghai = ZoneId.of("Asia/Shanghai");
ZonedDateTime shanghaiTime = ZonedDateTime.now(shanghai);

// Instant（时间戳，以Unix元年1970-01-01 00:00:00开始）
Instant now = Instant.now();
long timestamp = now.toEpochMilli();  // 毫秒级时间戳

// 与旧API互转
Date oldDate = Date.from(Instant.now());
Instant instant = oldDate.toInstant();
LocalDateTime fromDate = LocalDateTime.ofInstant(instant, ZoneId.systemDefault());
```

## 六、JDK 9-21新特性精选

### 6.1 模块化系统（JPMS - JDK 9）

```java
// module-info.java
module com.example.mylibrary {
    requires java.sql;           // 声明依赖
    requires transitive java.xml; // 传递依赖（使用方也可见）
    
    exports com.example.api;     // 导出包
    exports com.example.model to com.example.client;  // 限定导出
    
    opens com.example.internal to spring.core;  // 反射访问
}
```

### 6.2 var局部变量类型推断（JDK 10）

```java
// 编译器自动推断变量类型
var list = new ArrayList<String>();       // ArrayList<String>
var map = new HashMap<String, Integer>(); // HashMap<String, Integer>
var stream = list.stream();               // Stream<String>

// 限制（仅在局部变量中使用）
// var不能在以下位置使用：类字段、方法参数、方法返回类型
// 不适合推断类型不明显的情况
var result = someMethod();  // 不推荐，返回类型不明确
```

### 6.3 Switch表达式（JDK 14）

```java
// 传统switch（容易遗漏break）
String result;
switch (day) {
    case MONDAY: result = "周一"; break;
    case TUESDAY: result = "周二"; break;
    default: result = "未知";
}

// 新switch表达式（更简洁、安全）
String result = switch (day) {
    case MONDAY -> "周一";
    case TUESDAY, WEDNESDAY -> "工作日";
    case SATURDAY, SUNDAY -> {
        System.out.println("周末处理...");
        yield "周末";  // 使用yield返回值（代码块形式）
    }
};

// 模式匹配（JDK 17+）
Object obj = "Hello";
String description = switch (obj) {
    case Integer i -> "整数: " + i;
    case String s -> "字符串: " + s;
    case Long l -> "长整数: " + l;
    case null -> "null值";
    default -> "未知类型";
};
```

### 6.4 Record类（JDK 16）

Record是一种特殊的类，用于透明地承载不可变数据。自动生成构造器、getter、equals、hashCode、toString。

```java
// 传统POJO（大量样板代码）
public class User {
    private final String name;
    private final int age;
    // 构造器、getter、equals、hashCode、toString...
}

// Record（简洁替代）
public record User(String name, int age) {
    // 紧凑构造器（参数校验）
    public User {
        if (age < 0) throw new IllegalArgumentException("年龄不能为负");
        if (name == null || name.isBlank()) throw new IllegalArgumentException("名称不能为空");
    }
    
    // 可添加静态方法和实例方法
    public static User of(String name, int age) {
        return new User(name, age);
    }
    
    public String fullInfo() {
        return name + "(" + age + "岁)";
    }
}

// 使用
User user = new User("张三", 25);
System.out.println(user.name());  // "张三"（访问器方法，非getName()）
System.out.println(user);         // "User[name=张三, age=25]"
```

### 6.5 文本块（Text Blocks - JDK 15）

```java
// JDK 15之前
String json = "{\n" +
    "  \"name\": \"张三\",\n" +
    "  \"age\": 25\n" +
    "}";

// 文本块（使用 """ 分隔符）
String json = """
    {
      "name": "张三",
      "age": 25
    }
    """;

// S情文本块（去除尾部缩进，使用\续行）
String sql = """
    SELECT u.id, u.name, d.dept_name \
    FROM t_user u \
    JOIN t_department d ON u.dept_id = d.id \
    WHERE u.status = 1 \
    ORDER BY u.create_time DESC
    """;
```

### 6.6 密封类（Sealed Classes - JDK 17）

限制哪些类可以继承或实现该类/接口。

```java
// 密封类（使用permits控制继承）
public sealed class Shape 
    permits Circle, Rectangle, Triangle {
}

public final class Circle extends Shape { }
public non-sealed class Rectangle extends Shape { }  // 允许进一步扩展
public sealed class Triangle extends Shape permits RightTriangle { }

// 密封接口
public sealed interface PaymentMethod 
    permits CreditCard, DebitCard, WechatPay {}

record CreditCard(String cardNumber) implements PaymentMethod {}
record DebitCard(String bankName) implements PaymentMethod {}
record WechatPay(String openId) implements PaymentMethod {}
```

### 6.7 虚拟线程（Virtual Threads - JDK 21）

虚拟线程是轻量级线程，由JVM管理而非操作系统，可以高效地创建数百万个。

```java
// 传统平台线程（1:1映射到OS线程）
Thread platformThread = new Thread(() -> {
    System.out.println("平台线程");
});
platformThread.start();

// 虚拟线程（M:N映射，JDK 21）
Thread virtualThread = Thread.startVirtualThread(() -> {
    try {
        Thread.sleep(1000);
        System.out.println("虚拟线程完成");
    } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
    }
});

// 使用虚拟线程的ExecutorService（每个任务一个虚拟线程）
try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
    List<Future<String>> futures = new ArrayList<>();
    for (int i = 0; i < 10000; i++) {
        final int taskId = i;
        futures.add(executor.submit(() -> {
            // 模拟IO操作
            Thread.sleep(Duration.ofSeconds(1));
            return "Task " + taskId + " done";
        }));
    }
    
    // 等待所有任务完成
    for (Future<String> future : futures) {
        System.out.println(future.get());
    }
}

// 虚拟线程优势：
// 1. 轻量级：可创建数百万个
// 2. 阻塞操作几乎零成本（挂起时释放载体线程）
// 3. 适合IO密集型任务（Web请求、数据库调用、文件操作）
// 4. 与ThreadLocal兼容
// 注意：不适合CPU密集型任务（会争抢载体线程）
```

## 七、Java现代化编程最佳实践

1. **使用Lambda和方法引用替代匿名内部类**：提高代码可读性
2. **优先使用Stream API处理集合数据**：声明式编程，支持并行处理
3. **用Optional取代null返回值**：减少NPE风险
4. **使用新的日期时间API**：线程安全，API设计更合理
5. **用Record替代简单POJO**：减少样板代码
6. **使用var进行类型推断**：在类型明确时简化代码
7. **拥抱虚拟线程**：在高并发IO场景中使用虚拟线程替代传统线程池
8. **使用模块化系统组织大型项目**：增强封装性
9. **使用try-with-resources管理资源**：避免资源泄漏
10. **升级到LTS版本**：Java 8 -> Java 17 -> Java 21

## 八、总结

Java从8到21的演进体现了语言现代化的趋势：更简洁的语法（Lambda、Record）、更强的类型系统（泛型增强、密封类）、更好的并发支持（虚拟线程）、更安全的编程方式（Optional）。作为Java开发者，掌握这些新特性不仅能够提升开发效率，还能写出更健壮、更可维护的代码。在实际项目中，应根据项目JDK版本合理使用新特性，保持团队编码风格的一致性。
