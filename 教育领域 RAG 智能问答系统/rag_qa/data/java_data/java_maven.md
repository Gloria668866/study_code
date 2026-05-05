# Maven/Gradle构建工具使用与最佳实践

## 一、构建工具概述

Java构建工具经历了从Ant到Maven再到Gradle的演进过程。Maven是目前企业中最广泛使用的构建工具，Gradle在Android和新项目中逐渐流行。

**为什么需要构建工具？**
- 依赖管理：自动下载和管理第三方库及其传递依赖
- 构建自动化：编译、测试、打包、部署一体化
- 标准化项目结构：统一目录布局，降低项目交接成本
- 生命周期管理：清晰的构建阶段和插件机制

## 二、Maven详解

### 2.1 Maven核心概念

Maven的核心配置文件是`pom.xml`（Project Object Model），定义了项目的基本信息、依赖、插件和构建配置。

**Maven项目标准目录结构**：

```
project/
├── pom.xml                    # Maven项目配置文件
├── src/
│   ├── main/
│   │   ├── java/              # Java源代码
│   │   ├── resources/         # 资源文件（配置文件等）
│   │   └── webapp/            # Web应用资源
│   └── test/
│       ├── java/              # 测试代码
│       └── resources/         # 测试资源文件
└── target/                    # 构建输出目录
```

### 2.2 pom.xml配置详解

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    
    <!-- 项目坐标（GAV） -->
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>        <!-- 组织/公司标识 -->
    <artifactId>demo-project</artifactId>  <!-- 项目名称 -->
    <version>1.0.0-SNAPSHOT</version>     <!-- 版本 -->
    <packaging>jar</packaging>             <!-- 打包方式：jar/war/pom -->
    
    <!-- 属性定义 -->
    <properties>
        <java.version>17</java.version>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <spring-boot.version>3.2.0</spring-boot.version>
        <maven.compiler.source>${java.version}</maven.compiler.source>
        <maven.compiler.target>${java.version}</maven.compiler.target>
    </properties>
    
    <!-- 依赖管理（BOM - Bill of Materials） -->
    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-dependencies</artifactId>
                <version>${spring-boot.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>
    
    <!-- 项目依赖 -->
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <!-- 版本由dependencyManagement统一管理 -->
        </dependency>
        
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>8.0.33</version>
            <scope>runtime</scope>  <!-- 运行时依赖，编译时不需要 -->
        </dependency>
        
        <!-- 测试依赖 -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <scope>test</scope>
        </dependency>
        
        <!-- 排除传递依赖 -->
        <dependency>
            <groupId>com.example</groupId>
            <artifactId>some-lib</artifactId>
            <exclusions>
                <exclusion>
                    <groupId>commons-logging</groupId>
                    <artifactId>commons-logging</artifactId>
                </exclusion>
            </exclusions>
        </dependency>
    </dependencies>
    
    <!-- 多模块管理 -->
    <modules>
        <module>module-common</module>
        <module>module-service</module>
        <module>module-web</module>
    </modules>
    
    <!-- 构建配置 -->
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
            
            <!-- Maven Compiler Plugin -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
                <configuration>
                    <source>17</source>
                    <target>17</target>
                </configuration>
            </plugin>
            
            <!-- 打包时排除某些文件 -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <configuration>
                    <excludes>
                        <exclude>**/application-local.yml</exclude>
                    </excludes>
                </configuration>
            </plugin>
        </plugins>
        
        <!-- 资源过滤 -->
        <resources>
            <resource>
                <directory>src/main/resources</directory>
                <filtering>true</filtering>  <!-- 开启变量替换 -->
            </resource>
        </resources>
    </build>
    
    <!-- Maven仓库配置 -->
    <repositories>
        <repository>
            <id>aliyun</id>
            <url>https://maven.aliyun.com/repository/public</url>
        </repository>
    </repositories>
    
    <!-- Profile多环境配置 -->
    <profiles>
        <profile>
            <id>dev</id>
            <activation>
                <activeByDefault>true</activeByDefault>
            </activation>
            <properties>
                <env>development</env>
            </properties>
        </profile>
        <profile>
            <id>prod</id>
            <properties>
                <env>production</env>
            </properties>
        </profile>
    </profiles>
</project>
```

### 2.3 Maven依赖作用域

| Scope | 说明 | 示例 |
|-------|------|------|
| compile（默认） | 编译、测试、运行都有效 | spring-core |
| provided | 编译和测试有效，运行时由容器提供 | servlet-api |
| runtime | 测试和运行有效，编译不需要 | JDBC驱动 |
| test | 仅测试有效 | JUnit |
| system | 类似provided，需指定本地jar路径 | - |
| import | 引入依赖管理POM | spring-boot-dependencies |

### 2.4 Maven生命周期与命令

Maven有三套独立的生命周期：Clean、Default（Build）、Site。

**常用命令**：

```bash
# 清理
mvn clean                              # 删除target目录

# 编译
mvn compile                            # 编译源代码
mvn test-compile                       # 编译测试代码

# 测试
mvn test                               # 运行单元测试
mvn test -Dtest=UserServiceTest        # 运行指定测试类
mvn test -Dmaven.test.skip=true        # 跳过测试

# 打包
mvn package                            # 打包（jar/war）
mvn package -DskipTests                # 打包但跳过测试

# 安装到本地仓库
mvn install                            # 安装到本地Maven仓库

# 部署到远程仓库
mvn deploy                             # 部署到远程仓库

# 依赖分析
mvn dependency:tree                    # 查看依赖树
mvn dependency:analyze                 # 分析未使用的依赖
mvn versions:display-dependency-updates # 查看依赖更新

# 多线程构建
mvn clean install -T 4C                # 4个CPU核心并行构建

# 离线构建
mvn clean install -o                   # 离线模式（不下载依赖）

# 跳过测试
mvn clean install -DskipTests          # 跳过测试执行但编译测试
mvn clean install -Dmaven.test.skip=true # 跳过测试编译和执行

# 指定Profile
mvn clean package -P prod              # 使用prod profile
```

### 2.5 Maven依赖冲突解决

```bash
# 查看依赖冲突
mvn dependency:tree -Dincludes=com.google.guava

# 依赖传递规则：
# 1. 最短路径优先：A -> B -> C(1.0), A -> D -> E -> C(2.0)
#    最终使用C(1.0)，因为路径更短
# 2. 先声明优先：路径长度相同时，谁先在pom中声明用谁
# 3. 手动排除：
<dependency>
    <groupId>com.example</groupId>
    <artifactId>lib-a</artifactId>
    <exclusions>
        <exclusion>
            <groupId>com.google.guava</groupId>
            <artifactId>guava</artifactId>
        </exclusion>
    </exclusions>
</dependency>
<dependency>
    <groupId>com.google.guava</groupId>
    <artifactId>guava</artifactId>
    <version>32.0.0-jre</version>
</dependency>
```

### 2.6 私有仓库

使用Nexus或Artifactory搭建私有Maven仓库，管理内部依赖和代理中央仓库。

```xml
<!-- settings.xml中配置镜像和认证 -->
<settings>
    <mirrors>
        <mirror>
            <id>nexus</id>
            <mirrorOf>*</mirrorOf>
            <url>http://nexus.example.com/repository/maven-public/</url>
        </mirror>
    </mirrors>
    
    <servers>
        <server>
            <id>nexus-releases</id>
            <username>deployer</username>
            <password>${nexus.password}</password>
        </server>
    </servers>
</settings>
```

## 三、Gradle详解

### 3.1 Gradle核心概念

Gradle使用基于Groovy（或Kotlin）的DSL定义构建脚本，比Maven的XML更灵活、简洁。

**Gradle项目结构**：

```
project/
├── build.gradle                  # 构建脚本（Groovy DSL）
├── build.gradle.kts              # 构建脚本（Kotlin DSL）
├── settings.gradle               # 项目设置（包含子模块信息）
├── gradle/
│   └── wrapper/                  # Gradle Wrapper
├── gradlew                       # Unix Wrapper脚本
├── gradlew.bat                   # Windows Wrapper脚本
└── src/
    ├── main/java/
    └── test/java/
```

### 3.2 build.gradle核心配置

```groovy
// 插件
plugins {
    id 'java'
    id 'org.springframework.boot' version '3.2.0'
    id 'io.spring.dependency-management' version '1.1.4'
}

// 项目信息
group = 'com.example'
version = '1.0.0-SNAPSHOT'
sourceCompatibility = '17'

// 仓库
repositories {
    maven { url 'https://maven.aliyun.com/repository/public' }
    mavenCentral()
}

// 依赖
dependencies {
    // Spring Boot Starter
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
    
    // 数据库驱动
    runtimeOnly 'mysql:mysql-connector-java:8.0.33'
    
    // 工具库
    implementation 'com.google.guava:guava:32.0.0-jre'
    compileOnly 'org.projectlombok:lombok'
    annotationProcessor 'org.projectlombok:lombok'
    
    // 测试
    testImplementation 'org.springframework.boot:spring-boot-starter-test'
    testImplementation 'org.junit.jupiter:junit-jupiter'
    
    // 多模块依赖
    implementation project(':module-common')
}

// 配置
configurations {
    all {
        // 全局排除传递依赖
        exclude group: 'commons-logging', module: 'commons-logging'
    }
    compileOnly {
        extendsFrom annotationProcessor
    }
}

// 测试配置
test {
    useJUnitPlatform()
    testLogging {
        events "passed", "skipped", "failed"
    }
}

// 多环境配置
bootRun {
    systemProperty 'spring.profiles.active', System.getProperty('spring.profiles.active', 'dev')
}
```

### 3.3 Gradle依赖配置类型

| 配置 | 说明 | Maven对应 |
|------|------|-----------|
| implementation | 编译和运行时依赖，不暴露给消费者 | compile（推荐） |
| api | 编译和运行时依赖，暴露给消费者 | compile |
| compileOnly | 仅编译时依赖 | provided |
| runtimeOnly | 仅运行时依赖 | runtime |
| testImplementation | 测试编译和运行时依赖 | test |
| annotationProcessor | 注解处理器 | - |

### 3.4 Gradle常用命令

```bash
# 基础命令
./gradlew build                       # 构建项目（编译+测试+打包）
./gradlew clean                       # 清理构建目录
./gradlew test                        # 运行测试
./gradlew jar                         # 打包jar

# 跳过测试
./gradlew build -x test               # 排除test任务

# 并行构建
./gradlew build --parallel

# 守护进程（加速构建）
./gradlew build --daemon

# 刷新依赖
./gradlew build --refresh-dependencies

# 依赖分析
./gradlew dependencies                # 查看依赖树
./gradlew dependencyInsight --dependency guava  # 指定依赖分析

# 生成Wrapper
gradle wrapper --gradle-version 8.5
```

### 3.5 Gradle Wrapper

Gradle Wrapper确保所有开发者使用相同版本的Gradle，无需预装Gradle。

```properties
# gradle/wrapper/gradle-wrapper.properties
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-8.5-bin.zip
networkTimeout=10000
validateDistributionUrl=true
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
```

## 四、构建工具最佳实践

### 4.1 版本管理策略

```groovy
// 方式1：统一属性变量（Maven/Gradle通用）
ext {
    springVersion = '3.2.0'
    guavaVersion = '32.0.0-jre'
}

// 方式2：使用BOM（Bill of Materials）统一管理
dependencyManagement {
    imports {
        mavenBom "org.springframework.boot:spring-boot-dependencies:${springVersion}"
    }
}

// 方式3：Gradle Version Catalog（推荐现代项目）
// gradle/libs.versions.toml
[versions]
spring-boot = "3.2.0"
guava = "32.0.0-jre"

[libraries]
spring-boot-starter-web = { group = "org.springframework.boot", name = "spring-boot-starter-web", version.ref = "spring-boot" }
guava = { group = "com.google.guava", name = "guava", version.ref = "guava" }

[bundles]
spring-web = ["spring-boot-starter-web", "spring-boot-starter-validation"]
```

### 4.2 多模块项目结构

```
parent-project/
├── pom.xml (packaging=pom)
├── common/                        # 公共模块
│   └── pom.xml
├── service/                       # 业务服务模块
│   └── pom.xml (依赖common)
├── web/                           # Web层模块
│   └── pom.xml (依赖service)
└── assembly/                      # 打包模块
    └── pom.xml (依赖所有模块)
```

### 4.3 构建加速技巧

1. **Maven多线程构建**：`mvn -T 4C clean install`
2. **Maven增量构建**：`mvn -pl module-name -am`（仅构建指定模块及其依赖模块）
3. **Gradle守护进程**：配置`org.gradle.daemon=true`
4. **Gradle并行构建**：配置`org.gradle.parallel=true`
5. **Gradle构建缓存**：配置`org.gradle.caching=true`
6. **本地仓库**：搭建Nexus代理，避免重复下载
7. **合理排除测试**：开发阶段使用`-DskipTests`

### 4.4 CI/CD中的构建配置

```yaml
# Jenkinsfile 或 GitHub Actions 中的构建步骤
stages {
    stage('Build') {
        steps {
            sh 'mvn clean compile -DskipTests'
            // 或
            sh './gradlew compileJava'
        }
    }
    stage('Test') {
        steps {
            sh 'mvn test'
            // 或
            sh './gradlew test'
        }
    }
    stage('Package') {
        steps {
            sh 'mvn package -DskipTests'
            // 或
            sh './gradlew bootJar'
        }
    }
}
```

## 五、总结

Maven和Gradle是Java生态中不可替代的构建工具。Maven以约定大于配置的理念，提供了标准化的构建流程；Gradle则以其灵活性和性能优势受到青睐。在实际项目中，需要掌握依赖管理、多模块构建、版本统一管理、构建优化等核心技能。对于新建项目，推荐使用Gradle；对于维护现有项目，保持与团队一致的工具选择。
