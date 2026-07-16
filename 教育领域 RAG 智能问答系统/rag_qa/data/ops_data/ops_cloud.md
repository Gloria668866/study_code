# 云计算基础

## 云计算概述

云计算（Cloud Computing）是一种通过互联网按需提供计算资源（服务器、存储、数据库、网络、软件等）的服务模式。用户无需自行购买和维护物理硬件，只需为实际使用的资源付费。

## 服务模型

### IaaS（基础设施即服务）
提供虚拟化的计算资源，如虚拟机、存储和网络。用户可以完全控制操作系统和运行环境，无需管理底层硬件。

代表服务：AWS EC2、阿里云ECS、Azure Virtual Machines

### PaaS（平台即服务）
提供应用开发和部署平台，用户只需关注应用代码，无需管理底层基础设施。

代表服务：AWS Elastic Beanstalk、阿里云ESS、Heroku、Google App Engine

### SaaS（软件即服务）
通过互联网直接提供软件应用，用户通过浏览器访问，无需安装和管理。

代表服务：Google Workspace、Office 365、Salesforce

## AWS核心服务

### 计算服务

**EC2（Elastic Compute Cloud）**
可扩展的虚拟服务器，支持多种实例类型：
- 通用型（t3/m5）：均衡的计算、内存和网络资源
- 计算优化型（c5）：高性能CPU
- 内存优化型（r5）：大内存应用

**Lambda**
无服务器计算服务，按执行时间和请求次数计费，无需管理服务器。

### 存储服务

**S3（Simple Storage Service）**
对象存储服务，提供99.999999999%（11个9）的数据持久性：
- Standard：频繁访问的热数据
- Infrequent Access：不常访问的数据
- Glacier：归档存储，成本最低

**EBS（Elastic Block Store）**
块存储卷，挂载到EC2实例使用。

**RDS（Relational Database Service）**
托管的关系型数据库，支持MySQL、PostgreSQL、Oracle、SQL Server。

### 网络与内容分发

**VPC（Virtual Private Cloud）**
隔离的虚拟网络环境，可定义IP地址范围、子网、路由表和网关。

**CloudFront**
全球内容分发网络（CDN），加速静态和动态内容的分发。

## 阿里云核心服务

### ECS（Elastic Compute Service）
弹性计算服务，类似AWS EC2：
- 实例规格族：通用型g7、计算型c7、内存型r7
- 支持按量付费和包年包月

### OSS（Object Storage Service）
对象存储服务，类似AWS S3：
- 标准存储、低频访问存储、归档存储
- 图片处理、视频转码等增值服务

### 其他核心服务
- **SLB**：负载均衡服务
- **CDN**：内容分发网络
- **ACK**：阿里云容器服务Kubernetes版
- **Function Compute**：无服务器计算

## 云原生技术

### 容器化
使用Docker将应用及其依赖打包成标准化容器镜像。

### Kubernetes编排
自动化容器部署、扩缩和管理。

### 微服务架构
将单体应用拆分为独立部署的小型服务。

### DevOps实践
CI/CD流水线 + 基础设施即代码（Terraform/CloudFormation）

## 云安全

- **身份与访问管理（IAM）**：控制谁可以访问什么资源
- **安全组**：虚拟防火墙，控制入站和出站流量
- **加密**：传输加密（TLS/SSL）和存储加密（KMS）
- **合规认证**：ISO 27001、SOC 1/2/3、PCI DSS
