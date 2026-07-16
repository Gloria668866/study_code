# Spring框架核心原理：IoC容器与AOP面向切面编程

## 一、Spring框架概述

Spring是Java企业级开发中最广泛使用的开源框架，由Rod Johnson于2003年创建。其核心目标是简化Java企业级应用开发，提供了一套全面的基础设施支持。Spring框架的核心是**控制反转（IoC）容器**和**面向切面编程（AOP）**，这两个概念构成了Spring生态系统的基石。

Spring框架的主要优势包括：
- **非侵入式设计**：代码可以完全不依赖Spring API
- **模块化架构**：按需选择所需模块，不引入不必要的依赖
- **声明式事务管理**：通过AOP实现，无需手动管理事务
- **与主流框架无缝集成**：支持MyBatis、Hibernate、Struts等
- **活跃的社区生态**：Spring Boot、Spring Cloud、Spring Security等子项目

## 二、IoC容器核心原理

### 2.1 IoC概念理解

控制反转（Inversion of Control）是一种设计思想，将对象的创建和管理权从程序代码转移给外部容器。在传统编程中，对象通过`new`关键字自己创建依赖；在Spring中，由IoC容器负责创建和管理对象及其依赖关系。

**依赖注入（Dependency Injection，DI）** 是IoC的一种实现方式，主要有三种注入方式：

```java
// 1. 构造器注入（推荐方式）
@Component
public class UserService {
    private final UserRepository userRepository;
    
    // 通过构造器注入依赖
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
}

// 2. Setter方法注入
@Component
public class OrderService {
    private PaymentService paymentService;
    
    @Autowired
    public void setPaymentService(PaymentService paymentService) {
        this.paymentService = paymentService;
    }
}

// 3. 字段注入（通过反射，不推荐）
@Component
public class ProductService {
    @Autowired
    private ProductRepository productRepository;  // 难以进行单元测试
}
```

**构造器注入优于字段注入的原因**：
- 依赖不可变（可使用final修饰）
- 保证依赖不为null
- 便于单元测试（可以手动传入mock对象）
- 避免循环依赖问题

### 2.2 Spring Bean容器

Spring容器负责创建、配置和管理Bean的生命周期。核心接口是`BeanFactory`和`ApplicationContext`。

```java
// ApplicationContext初始化
ApplicationContext context = new AnnotationConfigApplicationContext(AppConfig.class);

// 获取Bean
UserService userService = context.getBean(UserService.class);

// 获取所有指定类型的Bean
Map<String, UserService> beans = context.getBeansOfType(UserService.class);
```

**Bean的作用域（Scope）**：

| 作用域 | 说明 |
|--------|------|
| singleton | 默认作用域，整个容器中只有一个实例 |
| prototype | 每次获取都创建新实例 |
| request | 每个HTTP请求创建一个实例（Web环境） |
| session | 每个HTTP会话创建一个实例（Web环境） |
| application | 每个ServletContext创建一个实例（Web环境） |
| websocket | 每个WebSocket创建一个实例 |

```java
@Component
@Scope("prototype")  // 每次获取都是新实例
public class ReportGenerator {
    // 每次调用都产生新对象
}

// 注意：单例Bean注入原型Bean时，原型会失效
// 解决方案：使用@Lookup注解或ApplicationContext.getBean()
```

### 2.3 Bean的生命周期

Spring Bean的完整生命周期包括以下阶段：

1. **实例化**：通过反射创建Bean实例
2. **属性赋值**：填充Bean的依赖属性
3. **BeanNameAware.setBeanName()**：如果实现该接口，设置Bean名称
4. **BeanFactoryAware.setBeanFactory()**：如果实现该接口，设置BeanFactory
5. **ApplicationContextAware.setApplicationContext()**：设置ApplicationContext
6. **BeanPostProcessor.postProcessBeforeInitialization()**：初始化前处理
7. **@PostConstruct标注的方法**：自定义初始化逻辑
8. **InitializingBean.afterPropertiesSet()**：属性设置后回调
9. **BeanPostProcessor.postProcessAfterInitialization()**：初始化后处理（AOP代理在此创建）
10. **Bean就绪**：可以使用
11. **@PreDestroy标注的方法**：销毁前回调
12. **DisposableBean.destroy()**：销毁回调

```java
@Component
public class MyBean implements InitializingBean, DisposableBean {
    
    @PostConstruct
    public void init() {
        System.out.println("自定义初始化");
    }
    
    @Override
    public void afterPropertiesSet() {
        System.out.println("InitializingBean回调");
    }
    
    @PreDestroy
    public void cleanup() {
        System.out.println("自定义清理");
    }
    
    @Override
    public void destroy() {
        System.out.println("DisposableBean回调");
    }
}
```

### 2.4 自动装配与配置

Spring提供多种方式声明Bean：

```java
// 方式1：注解配置（@Component、@Service、@Repository、@Controller）
@Configuration
@ComponentScan("com.example")
public class AppConfig {
    
    // 方式2：@Bean方法声明
    @Bean
    public DataSource dataSource() {
        HikariDataSource ds = new HikariDataSource();
        ds.setJdbcUrl("jdbc:mysql://localhost:3306/mydb");
        ds.setUsername("root");
        ds.setPassword("password");
        return ds;
    }
    
    // 方式3：条件装配
    @Bean
    @ConditionalOnProperty(name = "cache.enabled", havingValue = "true")
    public CacheManager cacheManager() {
        return new RedisCacheManager();
    }
    
    // 方式4：Profile环境配置
    @Bean
    @Profile("dev")
    public MailService devMailService() {
        return new MockMailService();
    }
    
    @Bean
    @Profile("prod")
    public MailService prodMailService() {
        return new SmtpMailService();
    }
}
```

## 三、AOP面向切面编程

### 3.1 AOP核心概念

AOP（Aspect-Oriented Programming）是一种编程范式，通过将横切关注点（如日志、事务、安全）从核心业务逻辑中分离出来，实现代码的解耦和复用。

**核心术语**：
- **切面（Aspect）**：横切关注点的模块化，包含通知和切点
- **连接点（Join Point）**：程序执行的某个特定位置（方法调用、异常抛出等）
- **切点（Pointcut）**：匹配连接点的表达式
- **通知（Advice）**：在切点上执行的动作
- **织入（Weaving）**：将切面应用到目标对象创建代理的过程

### 3.2 AOP通知类型

```java
@Aspect
@Component
public class LoggingAspect {
    
    // 定义切点（重用切点表达式）
    @Pointcut("execution(* com.example.service.*.*(..))")
    public void serviceLayer() {}
    
    // 前置通知：目标方法执行前
    @Before("serviceLayer()")
    public void logBefore(JoinPoint joinPoint) {
        String methodName = joinPoint.getSignature().getName();
        Object[] args = joinPoint.getArgs();
        System.out.println("调用方法: " + methodName + ", 参数: " + Arrays.toString(args));
    }
    
    // 后置通知：目标方法正常返回后（不处理异常）
    @AfterReturning(pointcut = "serviceLayer()", returning = "result")
    public void logAfterReturning(JoinPoint joinPoint, Object result) {
        System.out.println("方法返回: " + result);
    }
    
    // 异常通知：目标方法抛出异常后
    @AfterThrowing(pointcut = "serviceLayer()", throwing = "ex")
    public void logAfterThrowing(JoinPoint joinPoint, Exception ex) {
        System.err.println("方法异常: " + ex.getMessage());
    }
    
    // 最终通知：无论正常返回还是异常都执行
    @After("serviceLayer()")
    public void logAfter(JoinPoint joinPoint) {
        System.out.println("方法执行完毕");
    }
    
    // 环绕通知：功能最强的通知，可控制方法是否执行
    @Around("serviceLayer()")
    public Object logAround(ProceedingJoinPoint joinPoint) throws Throwable {
        long start = System.currentTimeMillis();
        try {
            Object result = joinPoint.proceed();  // 执行目标方法
            long duration = System.currentTimeMillis() - start;
            System.out.println("方法耗时: " + duration + "ms");
            return result;
        } catch (Exception e) {
            System.out.println("方法执行异常，耗时: " + (System.currentTimeMillis() - start) + "ms");
            throw e;
        }
    }
}
```

### 3.3 切点表达式详解

```java
// 常用切点表达式模式
execution(* com.example.service.*.*(..))
// 解释：execution(返回值类型 包名.类名.方法名(参数类型))

// 具体示例：
execution(public * *(..))              // 所有public方法
execution(* com.example..*.*(..))      // com.example包及其子包的所有方法
execution(* com.example.service.UserService.find*(..))  // 所有find开头的方法
execution(* com.example.*.*(java.lang.String, ..))      // 第一个参数为String
execution(* com.example.service.*.*(..)) && @annotation(com.example.Log)  // 带@Log注解

// 切点组合
@Pointcut("execution(* com.example.service.*.*(..))")
public void serviceMethods() {}

@Pointcut("@annotation(org.springframework.transaction.annotation.Transactional)")
public void transactionalMethods() {}

@Pointcut("serviceMethods() && !transactionalMethods()")  // 服务层非事务方法
public void nonTransactionalServiceMethods() {}
```

### 3.4 Spring AOP实现原理

Spring AOP默认使用**JDK动态代理**和**CGLIB代理**两种方式：

- **JDK动态代理**：基于接口的代理，目标类必须实现接口。使用`java.lang.reflect.Proxy`和`InvocationHandler`
- **CGLIB代理**：基于继承的代理，通过生成目标类的子类实现。不能代理final类和方法

```java
// Spring Boot中默认策略（2.x版本后默认使用CGLIB）
// spring.aop.proxy-target-class=true  // 强制使用CGLIB
// spring.aop.proxy-target-class=false // 接口有实现就用JDK代理
```

**AOP应用场景**：
1. 声明式事务管理（@Transactional）
2. 日志记录和性能监控
3. 权限验证和安全控制
4. 缓存处理（@Cacheable）
5. 异常处理和重试机制
6. 请求参数校验和响应格式化

### 3.5 声明式事务管理

事务管理是AOP最经典的应用场景。Spring通过@Transactional注解提供声明式事务管理。

```java
@Service
public class OrderService {
    
    // 默认事务传播：REQUIRED（如果已有事务则加入，没有则新建）
    @Transactional
    public void createOrder(Order order) {
        orderRepository.save(order);
        inventoryService.decreaseStock(order.getProductId(), order.getQuantity());
    }
    
    // 只读事务（优化数据库性能）
    @Transactional(readOnly = true)
    public Order getOrder(Long id) {
        return orderRepository.findById(id).orElse(null);
    }
    
    // 新建独立事务
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void logOperation(String operation) {
        // 该操作在独立事务中执行，不受外部事务影响
    }
    
    // 指定回滚异常
    @Transactional(rollbackFor = {BusinessException.class})
    public void processOrder(Order order) throws BusinessException {
        // 遇到BusinessException时回滚
    }
}
```

**事务传播行为**：

| 传播行为 | 说明 |
|---------|------|
| REQUIRED | 默认；有事务则加入，无则新建 |
| REQUIRES_NEW | 总是新建事务，挂起当前事务 |
| SUPPORTS | 有事务则加入，无则以无事务方式执行 |
| NOT_SUPPORTED | 以非事务方式执行，挂起当前事务 |
| MANDATORY | 必须在事务中，否则抛出异常 |
| NEVER | 不能在事务中，否则抛出异常 |
| NESTED | 嵌套事务，可独立回滚 |

**@Transactional失效场景（常见坑）**：
1. 非public方法上使用（Spring默认只代理public方法）
2. 同类内部方法调用（没有经过代理对象）
3. 异常被catch后未抛出
4. 配置了noRollbackFor的异常
5. 数据库引擎不支持事务（如MyISAM）

```java
// 避免同类调用失效：注入自身代理对象
@Service
public class UserService {
    @Autowired
    private UserService self;  // 注入代理对象
    
    public void outerMethod() {
        self.innerTransactionalMethod();  // 通过代理调用
    }
    
    @Transactional
    public void innerTransactionalMethod() {
        // 事务会生效
    }
}
```

## 四、Spring事件机制

Spring提供了一套应用事件机制，支持组件间的松耦合通信：

```java
// 1. 定义事件
public class UserRegisteredEvent extends ApplicationEvent {
    private final User user;
    
    public UserRegisteredEvent(Object source, User user) {
        super(source);
        this.user = user;
    }
    public User getUser() { return user; }
}

// 2. 发布事件
@Service
public class UserService {
    @Autowired
    private ApplicationEventPublisher eventPublisher;
    
    public void register(User user) {
        // 注册逻辑
        eventPublisher.publishEvent(new UserRegisteredEvent(this, user));
    }
}

// 3. 监听事件（方式一：实现接口）
@Component
public class EmailListener implements ApplicationListener<UserRegisteredEvent> {
    @Override
    public void onApplicationEvent(UserRegisteredEvent event) {
        sendWelcomeEmail(event.getUser());
    }
}

// 4. 监听事件（方式二：注解，推荐）
@Component
public class LogListener {
    @EventListener
    @Async  // 异步执行
    public void handleUserRegistered(UserRegisteredEvent event) {
        log.info("新用户注册: {}", event.getUser().getUsername());
    }
}
```

## 五、总结

Spring框架的IoC容器实现了对象生命周期的统一管理，通过依赖注入降低了组件耦合度。AOP则解决了横切关注点的代码分散问题,使得日志、事务、安全等功能可以集中管理。理解IoC和AOP的工作原理，掌握Bean生命周期、作用域、事务传播机制，是深入使用Spring框架的关键。在实际项目中，合理运用这些核心特性可以显著提高代码的可维护性和可扩展性。
