# API接口测试方法与实战

## 一、API测试概述

API（Application Programming Interface）接口测试是验证应用程序接口是否满足功能、性能、安全等方面要求的测试类型。在微服务架构和前后端分离成为主流的今天，API测试的重要性日益凸显。

API测试的核心价值：

- **早期发现缺陷**：API测试不依赖UI，可以在前端开发之前进行
- **测试效率高**：API测试执行速度快，易于自动化
- **覆盖范围广**：可直接验证业务逻辑、数据验证规则、错误处理机制
- **稳定性好**：API接口变更频率远低于UI，测试脚本维护成本低
- **安全性验证**：可直接测试认证授权、数据加密等安全机制

## 二、HTTP协议基础

### 2.1 HTTP请求方法

| 方法 | 功能 | 幂等性 | 安全性 |
|------|------|--------|--------|
| GET | 获取资源 | 是 | 是 |
| POST | 创建资源 | 否 | 否 |
| PUT | 完整更新资源 | 是 | 否 |
| PATCH | 部分更新资源 | 否 | 否 |
| DELETE | 删除资源 | 是 | 否 |
| HEAD | 获取响应头 | 是 | 是 |
| OPTIONS | 获取支持的HTTP方法 | 是 | 是 |

### 2.2 常见HTTP状态码

```text
1xx 信息响应
  100 Continue     - 服务器已接收请求，客户端可以继续发送
  101 Switching     - 服务器将切换协议

2xx 成功响应
  200 OK            - 请求成功
  201 Created       - 资源创建成功
  204 No Content    - 成功但不返回响应体

3xx 重定向
  301 Moved Permanently  - 永久重定向
  302 Found              - 临时重定向
  304 Not Modified       - 资源未修改，使用缓存

4xx 客户端错误
  400 Bad Request   - 请求参数错误
  401 Unauthorized  - 未认证
  403 Forbidden     - 无权限
  404 Not Found     - 资源不存在
  405 Method Not Allowed - 请求方法不允许
  409 Conflict      - 资源冲突
  422 Unprocessable Entity - 无法处理的实体
  429 Too Many Requests - 请求频率超限

5xx 服务器错误
  500 Internal Server Error - 服务器内部错误
  502 Bad Gateway    - 网关错误
  503 Service Unavailable - 服务暂不可用
  504 Gateway Timeout - 网关超时
```

### 2.3 RESTful API设计规范

```text
资源命名规范：
GET     /api/v1/users         获取用户列表
GET     /api/v1/users/{id}    获取特定用户
POST    /api/v1/users         创建新用户
PUT     /api/v1/users/{id}    完整更新用户
PATCH   /api/v1/users/{id}    部分更新用户
DELETE  /api/v1/users/{id}    删除用户

GET     /api/v1/users/{id}/orders  获取用户的订单列表
POST    /api/v1/orders              创建订单

常用查询参数：
?page=1&size=20           分页查询
?sort=created_at,desc     排序
?status=active&role=admin 多条件过滤
?fields=id,name,email     字段选择
?q=keyword                全文搜索
```

## 三、Postman 接口测试实战

### 3.1 Postman 核心功能

Postman是国内使用最广泛的API开发和测试工具，提供图形化界面进行接口调试、自动化测试和文档生成。

```javascript
// Postman Pre-request Script（前置脚本）
// 在请求发送前自动生成动态参数
const timestamp = Date.now();
const nonce = Math.random().toString(36).substring(2);

// 生成签名验证（示例使用HMAC-SHA256）
const secret = pm.environment.get("api_secret");
const message = pm.request.url.toString() + timestamp + nonce;
const signature = CryptoJS.HmacSHA256(message, secret).toString();

// 将动态值设置到请求头和环境变量
pm.request.headers.add({
    key: "X-Timestamp",
    value: timestamp.toString()
});
pm.request.headers.add({
    key: "X-Nonce",
    value: nonce
});
pm.request.headers.add({
    key: "X-Signature",
    value: signature
});
```

```javascript
// Postman Tests Script（测试脚本）
// 验证HTTP状态码
pm.test("状态码为200", function () {
    pm.response.to.have.status(200);
});

// 验证响应体结构
pm.test("响应包含必要字段", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property("code");
    pm.expect(jsonData).to.have.property("message");
    pm.expect(jsonData).to.have.property("data");
});

// 验证业务逻辑
pm.test("返回的用户数据正确", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData.code).to.eql(200);
    pm.expect(jsonData.data).to.be.an("object");
    pm.expect(jsonData.data.name).to.be.a("string").and.not.empty;
    pm.expect(jsonData.data.email).to.match(
        /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
    );
});

// 验证响应时间
pm.test("响应时间小于500ms", function () {
    pm.expect(pm.response.responseTime).to.be.below(500);
});

// 链式调用：从响应中提取token用于下一个请求
pm.test("提取并保存认证token", function () {
    const jsonData = pm.response.json();
    pm.expect(jsonData.data.token).to.exist;
    pm.environment.set("auth_token", jsonData.data.token);
});

// JSON Schema验证
const schema = {
    "type": "object",
    "required": ["id", "name", "email", "createdAt"],
    "properties": {
        "id": { "type": "integer" },
        "name": { "type": "string", "minLength": 1 },
        "email": { "type": "string", "format": "email" },
        "createdAt": { "type": "string", "format": "date-time" }
    }
};

pm.test("响应数据结构符合Schema定义", function () {
    pm.response.to.have.jsonSchema(schema);
});
```

### 3.2 Postman集合与变量管理

```text
Postman变量层级（优先级从高到低）：
1. 全局变量（Global）
2. 集合变量（Collection）
3. 环境变量（Environment）
4. 数据变量（Data - 来自CSV/JSON文件）
5. 局部变量（Local - 脚本中临时创建）

环境变量示例：
开发环境：
  base_url: http://localhost:3000
  db_host: localhost
  api_key: dev_key_xxx

测试环境：
  base_url: https://test-api.example.com
  db_host: test-db.internal
  api_key: test_key_xxx

生产环境：
  base_url: https://api.example.com
  db_host: prod-db.internal
  api_key: prod_key_xxx
```

### 3.3 Newman命令行运行

```bash
# 安装Newman（Postman的命令行运行器）
npm install -g newman

# 运行单个集合
newman run user_api_tests.postman_collection.json \
  --environment test_env.postman_environment.json

# 运行集合并生成报告
newman run collection.json \
  -e environment.json \
  --reporters cli,htmlextra \
  --reporter-htmlextra-export report.html \
  --iteration-count 3 \
  --delay-request 100

# 数据驱动测试（CSV文件）
newman run collection.json \
  -d test_data.csv \
  --iteration-count 0  # 0表示使用CSV的行数作为迭代次数
```

## 四、REST Assured 框架详解

### 4.1 REST Assured 核心API

REST Assured是Java生态中最流行的API测试框架，采用Given-When-Then的BDD风格编写测试用例。

```java
import io.restassured.RestAssured;
import io.restassured.http.ContentType;
import io.restassured.response.Response;
import static io.restassured.RestAssured.*;
import static org.hamcrest.Matchers.*;
import org.junit.jupiter.api.*;

public class UserApiTest {
    
    @BeforeAll
    public static void setup() {
        // 全局基础配置
        RestAssured.baseURI = "https://api.example.com";
        RestAssured.basePath = "/api/v1";
        RestAssured.port = 443;
        // 启用请求和响应的日志记录（调试用）
        RestAssured.enableLoggingOfRequestAndResponseIfValidationFails();
    }
    
    @Test
    public void testGetAllUsers() {
        // Given-When-Then 风格的API测试
        given()
            .header("Authorization", "Bearer " + getAuthToken())
            .queryParam("page", 1)
            .queryParam("size", 10)
        .when()
            .get("/users")
        .then()
            .statusCode(200)
            .contentType(ContentType.JSON)
            .body("code", equalTo(200))
            .body("data.content.size()", greaterThan(0))
            .body("data.content[0].id", notNullValue())
            .body("data.content[0].name", not(emptyString()))
            .time(lessThan(2000L));  // 响应时间验证
    }
    
    @Test
    public void testCreateUser() {
        // 构建请求体
        String requestBody = """
            {
                "name": "张三",
                "email": "zhangsan@example.com",
                "age": 28,
                "role": "admin"
            }
        """;
        
        // 发送POST请求并提取响应
        Response response = given()
            .header("Authorization", "Bearer " + getAuthToken())
            .contentType(ContentType.JSON)
            .body(requestBody)
        .when()
            .post("/users")
        .then()
            .statusCode(201)
            .body("data.name", equalTo("张三"))
            .body("data.email", equalTo("zhangsan@example.com"))
            .extract().response();
        
        // 提取创建的用户ID，用于后续测试
        int userId = response.path("data.id");
        System.out.println("创建的用户ID: " + userId);
    }
    
    @Test
    public void testUpdateUser() {
        // 使用JSON路径提取和验证
        given()
            .header("Authorization", "Bearer " + getAuthToken())
            .contentType(ContentType.JSON)
            .body("{\"name\": \"李四\"}")
        .when()
            .put("/users/1")
        .then()
            .statusCode(200)
            .body("data.name", equalTo("李四"));
    }
    
    @Test
    public void testDeleteUser() {
        given()
            .header("Authorization", "Bearer " + getAuthToken())
        .when()
            .delete("/users/100")
        .then()
            .statusCode(204);
    }
    
    @Test
    public void testValidationErrors() {
        // 测试参数校验 - 缺少必填字段
        String invalidBody = "{\"email\": \"invalid\"}";
        
        given()
            .header("Authorization", "Bearer " + getAuthToken())
            .contentType(ContentType.JSON)
            .body(invalidBody)
        .when()
            .post("/users")
        .then()
            .statusCode(400)
            .body("message", containsString("参数错误"))
            .body("errors", hasSize(greaterThan(0)));
    }
    
    @Test
    public void testAuthenticationFailure() {
        // 测试认证失败场景
        given()
            .header("Authorization", "Bearer invalid_token")
            .contentType(ContentType.JSON)
            .body("{}")
        .when()
            .post("/users")
        .then()
            .statusCode(401)
            .body("message", containsString("认证失败"));
    }
    
    private String getAuthToken() {
        // 获取认证token的方法
        return given()
            .contentType(ContentType.JSON)
            .body("{\"username\":\"admin\",\"password\":\"admin123\"}")
        .when()
            .post("/auth/login")
        .then()
            .statusCode(200)
            .extract().path("data.token");
    }
}
```

### 4.2 请求和响应的序列化与反序列化

```java
// 使用POJO类进行请求序列化
public class CreateUserRequest {
    private String name;
    private String email;
    private int age;
    private String role;
    
    // 构造器、Getter和Setter省略
}

// 使用POJO类进行响应反序列化
public class ApiResponse<T> {
    private int code;
    private String message;
    private T data;
    
    // Getter和Setter省略
}

public class User {
    private int id;
    private String name;
    private String email;
    private int age;
    private String role;
    private String createdAt;
    
    // Getter和Setter省略
}

@Test
public void testCreateUserWithPojo() {
    CreateUserRequest request = new CreateUserRequest();
    request.setName("王五");
    request.setEmail("wangwu@example.com");
    request.setAge(30);
    request.setRole("user");
    
    // 使用泛型提取响应
    ApiResponse<User> response = given()
        .header("Authorization", "Bearer " + getAuthToken())
        .contentType(ContentType.JSON)
        .body(request)
    .when()
        .post("/users")
    .then()
        .statusCode(201)
        .extract()
        .as(new TypeRef<ApiResponse<User>>() {});
    
    assertThat(response.getCode(), equalTo(201));
    assertThat(response.getData().getName(), equalTo("王五"));
    assertThat(response.getData().getEmail(), equalTo("wangwu@example.com"));
}
```

## 五、接口自动化测试设计

### 5.1 测试数据准备与清理策略

接口自动化测试中，测试数据的管理直接影响测试的可靠性和可重复性。常用的数据策略包括：

1. **测试前准备**：在测试执行前，通过API创建测试所需的数据
2. **测试后清理**：在测试执行后，通过API删除测试过程中产生的数据
3. **使用隔离数据**：为每个测试用例分配独立的测试数据，避免数据冲突
4. **数据工厂模式**：将测试数据的创建逻辑封装在工厂类中

```java
// 测试数据工厂示例
public class TestDataFactory {
    
    public static User createTestUser(RestAssuredConfig config) {
        String uniqueEmail = "test_" + System.currentTimeMillis() + "@example.com";
        
        return given()
            .spec(config.getRequestSpec())
            .body(Map.of(
                "name", "测试用户_" + System.currentTimeMillis(),
                "email", uniqueEmail,
                "age", 25,
                "role", "tester"
            ))
        .when()
            .post("/users")
        .then()
            .statusCode(201)
            .extract().as(User.class);
    }
    
    public static void cleanupTestUser(RestAssuredConfig config, int userId) {
        given()
            .spec(config.getRequestSpec())
        .when()
            .delete("/users/" + userId)
        .then()
            .statusCode(204);
    }
}
```

### 5.2 接口测试覆盖维度

```text
API接口测试覆盖维度：
├── 功能验证
│   ├── 正常场景：合法的请求参数，预期成功返回
│   ├── 异常场景：非法的请求参数，预期错误提示
│   ├── 边界场景：参数的边界值验证
│   └── 业务流程：多接口顺序调用的端到端场景
├── 数据验证
│   ├── 必填字段验证
│   ├── 字段类型验证
│   ├── 字段长度/范围验证
│   ├── 数据格式验证（邮箱、手机号等）
│   └── 业务规则验证
├── 安全验证
│   ├── 认证鉴权：无token、过期token、错误token
│   ├── 权限控制：越权访问、角色校验
│   ├── 参数注入：SQL注入、XSS攻击测试
│   └── 敏感信息：响应中是否暴露敏感数据
├── 性能验证
│   ├── 响应时间基线
│   ├── 高并发场景
│   └── 大负载数据量
└── 异常处理
    ├── 网络异常：超时、连接中断
    ├── 服务异常：下游服务不可用
    └── 数据异常：脏数据、重复数据
```

### 5.3 接口自动化测试框架设计

```text
推荐的接口自动化测试框架分层结构：
┌──────────────────────────────────────────┐
│          测试用例层 (Test Cases)          │
│  编写具体的测试场景和断言验证             │
├──────────────────────────────────────────┤
│          业务逻辑层 (Service Layer)        │
│  封装业务API调用，组合多个接口操作         │
├──────────────────────────────────────────┤
│          请求封装层 (Request Layer)        │
│  封装HTTP请求的发送、重试、日志记录        │
├──────────────────────────────────────────┤
│          配置管理层 (Config Layer)         │
│  环境配置、认证信息、全局参数管理           │
├──────────────────────────────────────────┤
│          工具支撑层 (Utility Layer)        │
│  数据驱动、报告生成、断言增强、DB操作      │
└──────────────────────────────────────────┘
```

### 5.4 持续集成中的API测试

```yaml
# GitLab CI API测试流水线配置
api-test:
  stage: test
  image: maven:3.8-openjdk-17
  services:
    - name: postgres:15
      alias: test-db
  variables:
    DB_HOST: test-db
    DB_USER: test
    DB_PASSWORD: test_password
    TEST_ENV: ci
  script:
    # 启动被测服务
    - java -jar target/app.jar &
    - sleep 30  # 等待服务启动
    
    # 健康检查
    - curl --retry 10 --retry-delay 5 --retry-connrefused http://localhost:8080/actuator/health
    
    # 运行API自动化测试
    - mvn test -Dtest="*ApiTest" -Dgroups="api"
  artifacts:
    when: always
    reports:
      junit: target/surefire-reports/TEST-*.xml
    paths:
      - target/surefire-reports/
```

## 六、API文档测试

API文档测试确保接口文档与实际实现保持一致。对于使用OpenAPI（Swagger）规范的项目，可进行自动化文档验证：

- 验证文档生成是否正确
- 验证实际响应是否符合文档定义的Schema
- 验证文档中描述的HTTP状态码是否与实际一致
- 通过文档自动生成测试用例（如使用Swagger Codegen）
