# 单元测试详解

## 什么是单元测试

单元测试是对软件中最小可测试单元（通常是一个函数或方法）进行验证的测试方法。它确保每个独立组件按照预期工作。

## JUnit框架

JUnit是Java生态最主流的单元测试框架。

### 基本注解

```java
import org.junit.jupiter.api.*;

@Test
void testAddition() {
    Calculator calc = new Calculator();
    assertEquals(5, calc.add(2, 3));
}

@BeforeEach
void setUp() {
    // 每个测试方法执行前调用
}

@AfterEach
void tearDown() {
    // 每个测试方法执行后调用
}

@BeforeAll
static void initAll() {
    // 所有测试方法执行前调用一次
}
```

### 常用断言

```java
assertEquals(expected, actual);      // 相等断言
assertTrue(condition);               // 条件为真
assertThrows(Exception.class, () -> {...}); // 异常断言
assertTimeout(Duration.ofSeconds(1), () -> {...}); // 超时断言
```

## Mockito Mock框架

Mockito用于创建和管理Mock对象，隔离测试目标的外部依赖。

### 创建Mock对象

```java
@Mock
private UserRepository userRepository;

@InjectMocks
private UserService userService;

@Test
void testFindUser() {
    User mockUser = new User("张三", "zhangsan@example.com");
    when(userRepository.findById(1L)).thenReturn(Optional.of(mockUser));
    
    UserDTO result = userService.getUser(1L);
    assertEquals("张三", result.getName());
    verify(userRepository, times(1)).findById(1L);
}
```

### Mock vs Stub

- **Stub**：返回预设值的简单替身，不验证调用行为
- **Mock**：验证交互行为的替身，可验证方法是否被调用、调用次数、调用参数

## TestNG

TestNG是另一个流行的测试框架，提供比JUnit更丰富的特性：

- 测试分组（groups）
- 参数化测试（@DataProvider）
- 依赖测试（dependsOnMethods）
- 并行测试执行

## 代码覆盖率

使用JaCoCo工具生成覆盖率报告：

```xml
<!-- Maven配置 -->
<plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <version>0.8.10</version>
</plugin>
```

### 覆盖率类型

| 类型 | 说明 |
|------|------|
| 行覆盖率 | 被执行过的代码行占比 |
| 分支覆盖率 | 被覆盖的条件分支占比 |
| 方法覆盖率 | 被调用过的方法占比 |
| 类覆盖率 | 被测试过的类占比 |

## 单元测试最佳实践

1. **AAA模式**：Arrange（准备）、Act（执行）、Assert（断言）
2. **FIRST原则**：Fast（快速）、Independent（独立）、Repeatable（可重复）、Self-Validating（自验证）、Timely（及时）
3. **一个测试只验证一个行为**
4. **测试命名清晰**：`methodName_StateUnderTest_ExpectedBehavior`
5. **避免测试中的逻辑**：测试代码应保持简单，不要有if/for等逻辑
