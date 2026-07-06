# 安全测试基础

## 安全测试概述

安全测试是验证软件系统是否能够保护数据和维护功能的安全性的测试过程。目标是发现系统中的安全漏洞和弱点，防止未经授权的访问、数据泄露和恶意攻击。

## OWASP Top 10 安全漏洞

OWASP（Open Web Application Security Project）定期发布最关键的Web应用安全风险：

### 1. 注入攻击（Injection）
最常见的是SQL注入，攻击者在输入中嵌入恶意代码。

**防护措施：**
- 使用参数化查询（PreparedStatement）
- ORM框架使用（Hibernate、MyBatis参数绑定）
- 输入验证和过滤
- 最小权限原则

### 2. 跨站脚本攻击（XSS）
攻击者向Web页面注入恶意脚本，在用户浏览器中执行。

**防护措施：**
- 输出编码（HTML实体编码）
- CSP（Content Security Policy）头设置
- HttpOnly Cookie标记
- 输入验证和净化

### 3. 跨站请求伪造（CSRF）
诱导用户在已认证的Web应用中执行非预期的操作。

**防护措施：**
- CSRF Token验证
- SameSite Cookie属性
- 验证Referer/Origin头
- 关键操作二次确认

### 4. 认证与授权缺陷
- 弱密码策略
- 会话管理不当
- 权限控制缺失
- JWT Token安全问题

### 5. 敏感数据泄露
- 明文存储密码（应使用bcrypt/scrypt哈希）
- 数据传输未加密
- 日志中包含敏感信息
- 错误信息暴露系统细节

## 渗透测试入门

### 信息收集阶段
- 端口扫描（Nmap）
- 子域名枚举
- 目录扫描（DirBuster、gobuster）
- 技术栈识别（Wappalyzer）

### 漏洞扫描
- 自动化扫描工具：Burp Suite、OWASP ZAP、Nessus
- SQL注入测试：sqlmap
- XSS测试：XSSer

### 常用测试Payload

XSS测试：
```
<script>alert('XSS')</script>
<img src=x onerror=alert('XSS')>
```

SQL注入测试：
```
' OR '1'='1
1' UNION SELECT NULL--
```

## 安全测试集成到CI/CD

1. SAST（静态应用安全测试）：SonarQube、Checkmarx
2. DAST（动态应用安全测试）：OWASP ZAP集成
3. 依赖项扫描：OWASP Dependency-Check、Snyk
4. 容器镜像扫描：Trivy、Clair

安全测试应在CI/CD流水线的每个阶段执行，及早发现并修复漏洞。
