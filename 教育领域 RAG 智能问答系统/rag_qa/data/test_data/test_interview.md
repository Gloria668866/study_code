# 软件测试面试高频考点

## 测试基础概念

### Q: 什么是软件测试？为什么需要软件测试？
软件测试是验证软件产品是否满足预期需求、发现缺陷并评估软件质量的过程。需要软件测试的原因包括：发现并修复Bug降低修复成本、确保产品满足用户需求、降低项目风险、提升用户满意度和产品竞争力。

### Q: 黑盒测试与白盒测试的区别？
- **黑盒测试**：不考虑内部结构，只关注输入和输出。方法包括等价类划分、边界值分析、决策表测试等。
- **白盒测试**：基于代码内部结构的测试。方法包括语句覆盖、分支覆盖、路径覆盖、条件覆盖等。
- **灰盒测试**：介于两者之间，了解部分内部结构。

### Q: 什么是回归测试？
回归测试是在代码修改后，重新执行已有的测试用例，确保修改没有引入新的Bug或影响原有功能。在敏捷开发中通常通过自动化回归测试来提高效率。

### Q: 冒烟测试和健全性测试的区别？
- **冒烟测试（Smoke Testing）**：验证系统的核心功能是否正常工作，通常在每次构建后进行。是一种广度优先的测试。
- **健全性测试（Sanity Testing）**：在冒烟测试通过后，验证特定功能在修复或变更后是否正常工作。是一种深度优先的测试。

## 自动化测试

### Q: 什么时候不适合做自动化测试？
以下情况不适合自动化：
- 需求频繁变化的模块
- 只执行一次的测试
- 需要大量人工判断的测试（如UI美观性）
- 开发周期短、预算有限的项目
- 测试环境不稳定的情况

### Q: Selenium中如何定位元素？
常用的8种定位方式：
1. `driver.findElement(By.id("elementId"))`
2. `driver.findElement(By.name("elementName"))`
3. `driver.findElement(By.className("className"))`
4. `driver.findElement(By.tagName("tag"))`
5. `driver.findElement(By.linkText("link text"))`
6. `driver.findElement(By.cssSelector("css"))`
7. `driver.findElement(By.xpath("//div[@id='test']"))`
8. `driver.findElement(By.partialLinkText("partial"))`

推荐优先级：id > name > cssSelector > xpath

## 性能测试

### Q: TPS和QPS的区别？
- **TPS（Transactions Per Second）**：每秒处理的事务数，一个事务可能包含多个请求
- **QPS（Queries Per Second）**：每秒查询数，常用于衡量单个接口的处理能力
- **RT（Response Time）**：响应时间，从发起请求到收到响应的时间
- **并发用户数**：同一时刻与系统交互的用户数量

### Q: 如何分析性能瓶颈？
1. 确定性能指标基线（如期望TPS > 1000）
2. 使用监控工具（Prometheus + Grafana）
3. 分析CPU、内存、磁盘I/O、网络I/O
4. 使用APM工具（SkyWalking、Pinpoint）追踪慢请求
5. 数据库慢查询分析，索引优化

## 测试工程师职业发展

### 技术路线
初级测试工程师 -> 中级测试工程师 -> 高级测试工程师 -> 测试架构师

关键技能发展：
- 自动化测试框架设计能力
- 性能测试与调优能力
- CI/CD流水线设计
- 测试平台与工具开发（测试开发方向）
- 安全测试专项能力
