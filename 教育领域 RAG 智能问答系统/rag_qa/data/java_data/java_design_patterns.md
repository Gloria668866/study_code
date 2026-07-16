# 常见设计模式在Java中的实现与应用

## 一、设计模式概述

设计模式是一套被反复使用、经过分类编目的代码设计经验总结。使用设计模式的目的是为了提高代码的可重用性、可读性和可靠性。GoF（Gang of Four）设计模式分为三大类共23种：创建型模式（5种）、结构型模式（7种）、行为型模式（11种）。

学习设计模式的关键是理解其**意图**和**适用场景**，而非死记硬背代码实现。同一个模式在不同的业务场景下可能有不同的实现方式。

## 二、创建型模式

创建型模式关注对象的创建机制，使系统独立于对象的创建、组合和表示。

### 2.1 单例模式（Singleton）

确保一个类只有一个实例，并提供全局访问点。

```java
// 1. 饿汉式（线程安全，类加载时创建，可能浪费内存）
public class EagerSingleton {
    private static final EagerSingleton INSTANCE = new EagerSingleton();
    
    private EagerSingleton() {
        // 防止反射破坏
        if (INSTANCE != null) {
            throw new RuntimeException("单例已被创建");
        }
    }
    
    public static EagerSingleton getInstance() {
        return INSTANCE;
    }
    
    // 防止反序列化破坏
    protected Object readResolve() {
        return INSTANCE;
    }
}

// 2. 枚举单例（最安全，自动防反射和反序列化攻击）
public enum EnumSingleton {
    INSTANCE;
    
    public void doSomething() {
        System.out.println("执行操作");
    }
}
// 使用：EnumSingleton.INSTANCE.doSomething();

// 3. 内部类单例（懒加载 + 线程安全，推荐）
public class Singleton {
    private Singleton() {
        if (Holder.INSTANCE != null) {
            throw new RuntimeException("单例已被创建");
        }
    }
    
    private static class Holder {
        private static final Singleton INSTANCE = new Singleton();
    }
    
    public static Singleton getInstance() {
        return Holder.INSTANCE;
    }
}

// 4. 双重检查锁定（DCL，不推荐，代码复杂）
public class DCLSingleton {
    private static volatile DCLSingleton instance;  // volatile防止指令重排序
    
    private DCLSingleton() {}
    
    public static DCLSingleton getInstance() {
        if (instance == null) {
            synchronized (DCLSingleton.class) {
                if (instance == null) {
                    instance = new DCLSingleton();
                }
            }
        }
        return instance;
    }
}
```

**Spring中的单例**：Spring容器默认管理的Bean都是单例（singleton作用域），但Spring的单例是"每个容器一个实例"，与GoF的单例模式略有不同。

### 2.2 工厂方法模式（Factory Method）

定义一个创建对象的接口，让子类决定实例化哪个类。

```java
// 产品接口
public interface PaymentService {
    void pay(BigDecimal amount);
}

// 具体产品
public class AlipayService implements PaymentService {
    @Override
    public void pay(BigDecimal amount) {
        System.out.println("支付宝支付: " + amount);
    }
}

public class WechatPayService implements PaymentService {
    @Override
    public void pay(BigDecimal amount) {
        System.out.println("微信支付: " + amount);
    }
}

// 工厂方法（Spring中常用）
@Configuration
public class PaymentConfig {
    
    @Bean
    @ConditionalOnProperty(name = "payment.type", havingValue = "alipay")
    public PaymentService alipayService() {
        return new AlipayService();
    }
    
    @Bean
    @ConditionalOnProperty(name = "payment.type", havingValue = "wechat")
    public PaymentService wechatPayService() {
        return new WechatPayService();
    }
}

// 策略工厂（动态选择）
@Component
public class PaymentFactory {
    private final Map<String, PaymentService> paymentMap = new HashMap<>();
    
    @Autowired
    public PaymentFactory(List<PaymentService> paymentServices) {
        for (PaymentService service : paymentServices) {
            String type = service.getClass().getAnnotation(PaymentType.class).value();
            paymentMap.put(type, service);
        }
    }
    
    public PaymentService getPaymentService(String type) {
        PaymentService service = paymentMap.get(type);
        if (service == null) {
            throw new IllegalArgumentException("不支持的支付类型: " + type);
        }
        return service;
    }
}
```

### 2.3 抽象工厂模式（Abstract Factory）

提供创建一系列相关或相互依赖对象的接口，而无需指定具体类。

```java
// 抽象产品族：UI组件（按钮、输入框）
public interface Button { void render(); }
public interface InputBox { void render(); }

// 不同风格的具体产品
public class WindowsButton implements Button {
    @Override public void render() { System.out.println("Windows风格按钮"); }
}
public class MacButton implements Button {
    @Override public void render() { System.out.println("Mac风格按钮"); }
}
public class WindowsInputBox implements InputBox {
    @Override public void render() { System.out.println("Windows风格输入框"); }
}
public class MacInputBox implements InputBox {
    @Override public void render() { System.out.println("Mac风格输入框"); }
}

// 抽象工厂
public interface UIFactory {
    Button createButton();
    InputBox createInputBox();
}

// 具体工厂
public class WindowsUIFactory implements UIFactory {
    @Override public Button createButton() { return new WindowsButton(); }
    @Override public InputBox createInputBox() { return new WindowsInputBox(); }
}

public class MacUIFactory implements UIFactory {
    @Override public Button createButton() { return new MacButton(); }
    @Override public InputBox createInputBox() { return new MacInputBox(); }
}
```

### 2.4 建造者模式（Builder）

将复杂对象的构建与表示分离，使同样的构建过程可以创建不同的表示。

```java
// 传统Builder模式
public class HttpRequest {
    private final String url;
    private final String method;
    private final Map<String, String> headers;
    private final String body;
    private final int timeout;
    
    private HttpRequest(Builder builder) {
        this.url = builder.url;
        this.method = builder.method;
        this.headers = Collections.unmodifiableMap(builder.headers);
        this.body = builder.body;
        this.timeout = builder.timeout;
    }
    
    public static class Builder {
        private final String url;        // 必选参数
        private String method = "GET";   // 可选参数（默认值）
        private Map<String, String> headers = new HashMap<>();
        private String body;
        private int timeout = 5000;
        
        public Builder(String url) {
            this.url = url;
        }
        
        public Builder method(String method) {
            this.method = method;
            return this;
        }
        
        public Builder addHeader(String key, String value) {
            this.headers.put(key, value);
            return this;
        }
        
        public Builder body(String body) {
            this.body = body;
            return this;
        }
        
        public Builder timeout(int timeout) {
            this.timeout = timeout;
            return this;
        }
        
        public HttpRequest build() {
            return new HttpRequest(this);
        }
    }
}

// 使用
HttpRequest request = new HttpRequest.Builder("https://api.example.com/users")
    .method("POST")
    .addHeader("Content-Type", "application/json")
    .body("{\"name\":\"张三\"}")
    .timeout(3000)
    .build();

// Lombok @Builder简化实现
@Builder
@Data
public class UserDTO {
    private Long id;
    private String username;
    @Builder.Default  // 使用默认值
    private Integer status = 1;
}
```

### 2.5 原型模式（Prototype）

用原型实例指定创建对象的种类，并通过拷贝这些原型创建新对象。

```java
// 实现Cloneable接口（浅拷贝）
public class Document implements Cloneable {
    private String title;
    private List<String> paragraphs;
    
    @Override
    public Document clone() {
        try {
            Document doc = (Document) super.clone();
            // 深拷贝：复制集合
            doc.paragraphs = new ArrayList<>(this.paragraphs);
            return doc;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
```

## 三、结构型模式

### 3.1 代理模式（Proxy）

为其他对象提供一种代理以控制对这个对象的访问。Spring AOP就是典型的动态代理应用。

```java
// 静态代理
public interface UserService {
    User findById(Long id);
}

public class UserServiceImpl implements UserService {
    @Override
    public User findById(Long id) {
        return new User(id, "张三");
    }
}

// 代理类（添加缓存功能）
public class UserServiceCacheProxy implements UserService {
    private final UserService target;
    private final Map<Long, User> cache = new ConcurrentHashMap<>();
    
    public UserServiceCacheProxy(UserService target) {
        this.target = target;
    }
    
    @Override
    public User findById(Long id) {
        return cache.computeIfAbsent(id, target::findById);
    }
}

// 动态代理（JDK）
public class LoggingProxy implements InvocationHandler {
    private final Object target;
    
    public LoggingProxy(Object target) {
        this.target = target;
    }
    
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.println("调用方法前: " + method.getName());
        long start = System.currentTimeMillis();
        Object result = method.invoke(target, args);
        System.out.println("调用方法后, 耗时: " + (System.currentTimeMillis() - start) + "ms");
        return result;
    }
    
    @SuppressWarnings("unchecked")
    public static <T> T createProxy(T target, Class<T> interfaceType) {
        return (T) Proxy.newProxyInstance(
            interfaceType.getClassLoader(),
            new Class[]{interfaceType},
            new LoggingProxy(target)
        );
    }
}
```

### 3.2 装饰器模式（Decorator）

动态地给对象添加一些额外职责。Java IO是装饰器模式的经典应用。

```java
// 基础组件
public interface DataSource {
    String readData();
}

public class FileDataSource implements DataSource {
    private final String filePath;
    public FileDataSource(String filePath) { this.filePath = filePath; }
    
    @Override
    public String readData() {
        return "从文件读取的数据";
    }
}

// 装饰器基类
public abstract class DataSourceDecorator implements DataSource {
    protected final DataSource wrappee;
    
    public DataSourceDecorator(DataSource source) {
        this.wrappee = source;
    }
}

// 加密装饰器
public class EncryptionDecorator extends DataSourceDecorator {
    public EncryptionDecorator(DataSource source) { super(source); }
    
    @Override
    public String readData() {
        String data = wrappee.readData();
        return decrypt(data);  // 解密
    }
}

// 压缩装饰器
public class CompressionDecorator extends DataSourceDecorator {
    public CompressionDecorator(DataSource source) { super(source); }
    
    @Override
    public String readData() {
        String data = wrappee.readData();
        return decompress(data);  // 解压缩
    }
}

// 使用（可叠加多个装饰器）
DataSource source = new EncryptionDecorator(
    new CompressionDecorator(
        new FileDataSource("data.dat")
    )
);
String data = source.readData();  // 自动完成解压缩和解密
```

### 3.3 适配器模式（Adapter）

将一个类的接口转换成客户端期望的另一个接口。在Spring中常见于日志框架适配。

```java
// 目标接口
public interface LogService {
    void info(String message);
    void error(String message, Throwable ex);
}

// 被适配者（第三方日志库）
public class ThirdPartyLogger {
    public void log(String level, String msg) {
        System.out.println("[" + level + "] " + msg);
    }
}

// 适配器
public class LoggerAdapter implements LogService {
    private final ThirdPartyLogger logger;
    
    public LoggerAdapter(ThirdPartyLogger logger) {
        this.logger = logger;
    }
    
    @Override
    public void info(String message) {
        logger.log("INFO", message);
    }
    
    @Override
    public void error(String message, Throwable ex) {
        logger.log("ERROR", message + " - " + ex.getMessage());
    }
}
```

### 3.4 外观模式（Facade）

为子系统中的一组接口提供一个统一的高层接口。常用于微服务中的API网关。

```java
// 复杂子系统
public class OrderService {
    public void createOrder(Order order) { /* 创建订单 */ }
}
public class InventoryService {
    public void decreaseStock(Long productId, int quantity) { /* 减库存 */ }
}
public class PaymentService {
    public void processPayment(BigDecimal amount, String method) { /* 处理支付 */ }
}
public class NotificationService {
    public void sendOrderConfirmation(String email) { /* 发送确认邮件 */ }
}

// 外观类
@Service
public class CheckoutFacade {
    @Autowired private OrderService orderService;
    @Autowired private InventoryService inventoryService;
    @Autowired private PaymentService paymentService;
    @Autowired private NotificationService notificationService;
    
    @Transactional
    public void checkout(Cart cart, PaymentInfo paymentInfo) {
        // 1. 创建订单
        Order order = orderService.createOrder(cart.toOrder());
        // 2. 扣减库存
        for (CartItem item : cart.getItems()) {
            inventoryService.decreaseStock(item.getProductId(), item.getQuantity());
        }
        // 3. 处理支付
        paymentService.processPayment(cart.getTotalPrice(), paymentInfo.getMethod());
        // 4. 发送通知
        notificationService.sendOrderConfirmation(cart.getUserEmail());
    }
}
```

## 四、行为型模式

### 4.1 策略模式（Strategy）

定义一系列算法，将它们封装起来，使它们可以互相替换。

```java
// 策略接口
public interface DiscountStrategy {
    BigDecimal applyDiscount(BigDecimal originalPrice);
}

// 具体策略
@Component("vip")
public class VipDiscountStrategy implements DiscountStrategy {
    @Override
    public BigDecimal applyDiscount(BigDecimal price) {
        return price.multiply(new BigDecimal("0.8"));  // VIP享8折
    }
}

@Component("super_vip")
public class SuperVipDiscountStrategy implements DiscountStrategy {
    @Override
    public BigDecimal applyDiscount(BigDecimal price) {
        return price.multiply(new BigDecimal("0.7"));  // 超级VIP享7折
    }
}

// 策略上下文（Spring自动注入所有策略实现）
@Service
public class PriceCalculator {
    private final Map<String, DiscountStrategy> strategyMap;
    
    public PriceCalculator(Map<String, DiscountStrategy> strategyMap) {
        this.strategyMap = strategyMap;
    }
    
    public BigDecimal calculate(String userType, BigDecimal originalPrice) {
        DiscountStrategy strategy = strategyMap.getOrDefault(userType,
            price -> price);  // 默认无折扣
        return strategy.applyDiscount(originalPrice);
    }
}
```

### 4.2 观察者模式（Observer）

定义对象间的一对多依赖，当一个对象状态改变时，所有依赖者都会收到通知。Spring事件机制就是观察者模式的实现。

```java
// Spring事件实现（已在前文Spring章节详细介绍）
// 事件类（主题）
public class OrderCreatedEvent extends ApplicationEvent {
    private final Order order;
    public OrderCreatedEvent(Object source, Order order) {
        super(source);
        this.order = order;
    }
    public Order getOrder() { return order; }
}

// 观察者1
@Component
public class SmsListener {
    @EventListener
    @Order(1)
    public void handleOrderCreated(OrderCreatedEvent event) {
        sendSms(event.getOrder().getPhone(), "订单已创建");
    }
}

// 观察者2
@Component
public class LogListener {
    @EventListener
    @Order(2)
    @Async  // 异步执行
    public void handleOrderCreated(OrderCreatedEvent event) {
        log.info("订单创建: {}", event.getOrder().getId());
    }
}
```

### 4.3 模板方法模式（Template Method）

定义算法骨架，将某些步骤延迟到子类实现。

```java
// 抽象模板
public abstract class DataProcessor {
    
    // 模板方法（final防止子类重写）
    public final void process() {
        loadData();
        validate();
        transform();
        save();
        onSuccess();  // 钩子方法
    }
    
    protected abstract void loadData();
    protected abstract void transform();
    
    protected void validate() { /* 默认实现 */ }
    protected void save() { /* 默认实现 */ }
    protected void onSuccess() { /* 钩子方法，子类可选实现 */ }
}

// 具体实现
public class ExcelProcessor extends DataProcessor {
    @Override
    protected void loadData() { System.out.println("从Excel加载数据"); }
    
    @Override
    protected void transform() { System.out.println("转换Excel数据"); }
    
    @Override
    protected void onSuccess() { System.out.println("发送成功通知"); }
}

// Spring中模板方法模式的应用
// JdbcTemplate、RestTemplate、RedisTemplate等
```

### 4.4 责任链模式（Chain of Responsibility）

为请求创建一个接收者链，每个处理器可以处理请求或传给下一个处理器。

```java
// 抽象处理器
public abstract class ApprovalHandler {
    protected ApprovalHandler next;
    
    public ApprovalHandler setNext(ApprovalHandler next) {
        this.next = next;
        return next;
    }
    
    public void handle(ExpenseReport report) {
        if (canHandle(report)) {
            doHandle(report);
        } else if (next != null) {
            next.handle(report);
        } else {
            System.out.println("无人能处理该报销: " + report.getAmount());
        }
    }
    
    protected abstract boolean canHandle(ExpenseReport report);
    protected abstract void doHandle(ExpenseReport report);
}

// 具体处理器
public class ManagerHandler extends ApprovalHandler {
    @Override
    protected boolean canHandle(ExpenseReport report) {
        return report.getAmount().compareTo(new BigDecimal("1000")) <= 0;
    }
    
    @Override
    protected void doHandle(ExpenseReport report) {
        System.out.println("经理审批了: " + report.getAmount());
    }
}

public class DirectorHandler extends ApprovalHandler {
    @Override
    protected boolean canHandle(ExpenseReport report) {
        return report.getAmount().compareTo(new BigDecimal("10000")) <= 0;
    }
    
    @Override
    protected void doHandle(ExpenseReport report) {
        System.out.println("总监审批了: " + report.getAmount());
    }
}

// 使用
ManagerHandler manager = new ManagerHandler();
DirectorHandler director = new DirectorHandler();
manager.setNext(director);  // 构建责任链
manager.handle(new ExpenseReport(new BigDecimal("500")));  // 经理处理
```

**Spring Security中的过滤器链就是典型的责任链模式**。

## 五、设计模式在Spring框架中的应用

| 设计模式 | Spring中的应用 |
|---------|---------------|
| 单例模式 | Bean的singleton作用域 |
| 工厂方法 | BeanFactory、ApplicationContext |
| 代理模式 | AOP代理（JDK/CGLIB） |
| 模板方法 | JdbcTemplate、RestTemplate、KafkaTemplate |
| 观察者模式 | ApplicationEvent事件机制 |
| 策略模式 | Resource接口不同实现、InstantiationStrategy |
| 适配器模式 | HandlerAdapter、MethodArgumentResolver |
| 装饰器模式 | BeanDefinitionDecorator |
| 责任链模式 | FilterChain、InterceptorChain |

## 六、总结

设计模式是解决特定问题场景的最佳实践总结。在Java开发中，Spring框架大量应用了各种设计模式，理解这些模式有助于深入理解Spring的工作原理。实际应用中应遵循"不要过度设计"的原则，在设计模式能真正解决问题时才使用，避免为了模式而模式。同时，JDK 8+引入的Lambda表达式和函数式编程思想，也简化了许多设计模式的实现方式。
