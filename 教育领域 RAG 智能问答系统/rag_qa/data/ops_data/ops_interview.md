# 运维/DevOps/SRE面试高频考点

## Linux基础

### Q: 如何查看Linux系统负载？
使用`top`、`uptime`、`w`命令查看系统负载。load average的三个数值分别表示1分钟、5分钟、15分钟的平均负载。通常负载值不超过CPU核心数为正常。

### Q: 如何查找占用CPU最高的进程？
```bash
ps aux --sort=-%cpu | head -10
top -o %CPU
```

### Q: 硬链接和软链接的区别？
- **硬链接**：指向同一个inode，删除原文件不影响硬链接，不能跨文件系统
- **软链接（符号链接）**：类似于Windows快捷方式，存储目标路径，原文件删除后失效

## Docker

### Q: Docker镜像、容器、仓库的区别？
- **镜像（Image）**：只读模板，包含运行应用所需的文件系统
- **容器（Container）**：镜像的运行实例，拥有可写层
- **仓库（Registry）**：存储和分发镜像的地方（如Docker Hub）

### Q: Dockerfile中COPY和ADD的区别？
COPY只复制本地文件到镜像。ADD除了复制功能外，还支持URL下载和tar自动解压。推荐优先使用COPY，需要解压时再使用ADD。

### Q: 如何减小Docker镜像体积？
- 使用alpine等精简基础镜像
- 多阶段构建（multi-stage build）
- 合并RUN指令减少镜像层
- 清理缓存和临时文件
- 使用.dockerignore排除不需要的文件

## Kubernetes

### Q: Pod、Service、Deployment分别是什么？
- **Pod**：K8s最小部署单元，包含一个或多个容器
- **Service**：为Pod提供稳定的网络访问入口和负载均衡
- **Deployment**：声明式管理Pod的创建、更新和扩缩

### Q: 如何排查Pod启动失败？
```bash
kubectl describe pod <pod-name>  # 查看详细信息
kubectl logs <pod-name>           # 查看日志
kubectl get events --sort-by='.lastTimestamp'  # 查看集群事件
```

### Q: ConfigMap和Secret的区别？
ConfigMap存储非敏感配置数据（明文），Secret存储敏感数据（如密码、Token），数据以base64编码存储。

## CI/CD

### Q: 持续集成、持续交付、持续部署的区别？
- **持续集成（CI）**：频繁合并代码到主干，自动构建和测试
- **持续交付（CD）**：代码随时可以部署到生产环境，但需要手动触发部署
- **持续部署（CD）**：代码通过自动化测试后自动部署到生产环境

### Q: 蓝绿部署和金丝雀部署是什么？
- **蓝绿部署**：维护两套完全相同的环境（蓝和绿），切换流量实现零停机部署
- **金丝雀部署**：逐步将新版本流量从小比例扩大，验证无问题后全量切换

## 监控与故障排除

### Q: 一个Web服务响应慢，如何排查？
1. 查看系统资源：CPU、内存、磁盘I/O、网络
2. 检查应用日志：错误日志、慢请求日志
3. 数据库排查：慢查询、连接数、锁等待
4. 网络排查：ping、traceroute、curl测试
5. APM工具追踪：分析调用链确定瓶颈位置

### Q: 什么是SRE？
SRE（Site Reliability Engineering）是Google提出的运维理念，用软件工程方法解决运维问题。核心概念包括：
- **SLO/SLI/SLA**：服务水平目标/指标/协议
- **错误预算**：允许的最大不可用时间
- **减少重复劳动**：自动化运维任务
- **事后分析**：无责的事后复盘文化
