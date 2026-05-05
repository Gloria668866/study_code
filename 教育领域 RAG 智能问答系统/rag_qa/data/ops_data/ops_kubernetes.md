# Kubernetes 容器编排平台

## 一、Kubernetes 简介

Kubernetes（简称 K8s）是 Google 开源的容器编排平台，用于自动化部署、扩展和管理容器化应用。它提供了服务发现、负载均衡、存储编排、自动伸缩、自愈等能力。

### 核心特性

- **服务发现与负载均衡**：自动为 Pod 分配 IP 和 DNS 名
- **自动装箱**：根据资源需求自动调度容器
- **自愈**：自动重启失败容器、替换和重新调度
- **水平伸缩**：基于 CPU/内存指标自动扩缩容
- **滚动更新与回滚**：零停机部署

## 二、K8s 架构

```
                    ┌──────────────────────┐
                    │      kubectl          │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    API Server        │ <── 集群统一入口
                    └──────────┬───────────┘
           ┌───────────────────┼───────────────────┐
           │                   │                   │
    ┌──────▼──────┐   ┌────────▼──────┐   ┌───────▼───────┐
    │  Scheduler  │   │  Controller   │   │     etcd      │
    │  (调度器)    │   │  Manager      │   │   (键值存储)   │
    └─────────────┘   └───────────────┘   └───────────────┘

                    ┌──────────────────────┐
                    │     Worker Node       │
                    │  ┌─────────────────┐  │
                    │  │     kubelet     │  │  <── 节点代理
                    │  │   kube-proxy    │  │  <── 网络代理
                    │  │ Container       │  │  <── 容器运行时
                    │  │ Runtime(CRI-O)  │  │
                    │  └─────────────────┘  │
                    └──────────────────────┘
```

### 核心组件

| 组件 | 角色 | 说明 |
|------|------|------|
| API Server | Master | 集群统一入口，RESTful API |
| etcd | Master | 分布式键值存储，保存集群所有状态 |
| Scheduler | Master | 负责 Pod 调度到合适节点 |
| Controller Manager | Master | 管理控制器（Deployment、ReplicaSet 等） |
| kubelet | Node | 节点代理，管理 Pod 生命周期 |
| kube-proxy | Node | 网络代理，实现 Service 规则 |

## 三、核心资源对象

### 3.1 Pod

Pod 是 K8s 最小调度单位，包含一个或多个共享网络和存储的容器。

```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx:alpine
    ports:
    - containerPort: 80
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
    livenessProbe:
      httpGet:
        path: /health
        port: 80
      initialDelaySeconds: 5
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /ready
        port: 80
      initialDelaySeconds: 3
      periodSeconds: 5
```

### 3.2 Deployment

Deployment 为 Pod 和 ReplicaSet 提供声明式更新，支持滚动更新和回滚。

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: myapp:v2.0
        ports:
        - containerPort: 8080
        env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: db_host
```

### 3.3 Service

Service 为一组 Pod 提供稳定的网络访问入口。

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  type: ClusterIP          # ClusterIP | NodePort | LoadBalancer
  selector:
    app: myapp
  ports:
  - protocol: TCP
    port: 80               # Service 端口
    targetPort: 8080       # Pod 端口
```

Service 类型对比：

| 类型 | 说明 | 使用场景 |
|------|------|----------|
| ClusterIP | 集群内部 IP | 内部服务通信（默认） |
| NodePort | 每个节点开放端口 | 开发/测试环境 |
| LoadBalancer | 云厂商负载均衡器 | 生产环境对外暴露 |
| ExternalName | DNS CNAME 解析 | 外部服务映射 |

### 3.4 ConfigMap 与 Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  db_host: postgres-service
  db_port: "5432"
  log_level: info
  app.properties: |
    server.tomcat.max-threads=200
    spring.profiles.active=production

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
data:
  db_password: cGFzc3dvcmQxMjM=    # Base64编码
  api_key: Z2hwX2tleV9oZXJl
```

## 四、常用 kubectl 命令

### 4.1 资源查看与管理

```bash
# 查看资源
kubectl get pods                           # 查看Pod列表
kubectl get pods -o wide                   # 查看Pod详情（含IP、节点）
kubectl get pods --all-namespaces          # 所有命名空间的Pod
kubectl get deployments                    # 查看部署
kubectl get services                       # 查看服务
kubectl get configmap                      # 查看配置
kubectl get nodes                          # 查看节点状态

# 查看详细信息
kubectl describe pod <pod-name>            # Pod 详细描述（事件、状态）
kubectl describe node <node-name>          # 节点资源信息

# 查看日志
kubectl logs <pod-name>                    # 查看Pod日志
kubectl logs -f <pod-name>                 # 实时跟踪日志
kubectl logs <pod-name> -c <container>     # 多容器Pod指定容器
kubectl logs --tail=100 <pod-name>         # 最近100行
kubectl logs --since=1h <pod-name>         # 最近1小时

# 进入容器
kubectl exec -it <pod-name> -- /bin/bash
kubectl exec <pod-name> -- ls /app
```

### 4.2 部署与更新

```bash
# 创建资源
kubectl apply -f deployment.yaml
kubectl create deployment nginx --image=nginx:alpine
kubectl create configmap app-config --from-file=config.properties
kubectl create secret generic db-secret --from-literal=password=xyz123

# 更新操作
kubectl set image deployment/myapp app=myapp:v3.0   # 更新镜像
kubectl rollout status deployment/myapp             # 查看更新状态
kubectl rollout history deployment/myapp            # 查看更新历史
kubectl rollout undo deployment/myapp               # 回滚到上一版本
kubectl rollout undo deployment/myapp --to-revision=2  # 回滚到指定版本

# 扩缩容
kubectl scale deployment myapp --replicas=5         # 手动扩容
kubectl autoscale deployment myapp --min=2 --max=10 --cpu-percent=80  # 自动伸缩

# 删除资源
kubectl delete -f deployment.yaml
kubectl delete pod <pod-name>
kubectl delete deployment <name>
```

### 4.3 故障排查

```bash
# Pod 状态诊断
kubectl describe pod <pod-name>          # 查看事件（ImagePullBackOff, CrashLoopBackOff等）
kubectl get events --sort-by=.metadata.creationTimestamp | tail -20

# 资源使用
kubectl top pods                         # Pod 资源使用（需要 metrics-server）
kubectl top nodes                        # 节点资源使用

# 端口转发（本地调试）
kubectl port-forward pod/<pod-name> 8080:80  # Pod端口转发到本地
kubectl port-forward service/<svc-name> 8080:80  # Service端口转发

# 执行临时Pod用于调试
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
```

## 五、部署策略

| 策略 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| 滚动更新 | 逐步替换旧 Pod | 零停机 | 短时间内新旧版本共存 |
| 蓝绿部署 | 新旧两套环境切换 | 快速回滚 | 双倍资源 |
| 金丝雀发布 | 少部分流量切到新版本 | 风险低 | 实现复杂（需Istio等） |

## 六、实践要点

1. **资源限制**：始终设置 `requests` 和 `limits`，避免资源争抢
2. **健康检查**：配置 `livenessProbe`（存活检测）和 `readinessProbe`（就绪检测）
3. **命名空间隔离**：不同环境（dev/staging/prod）使用不同 namespace
4. **标签规范**：统一标签命名（`app`, `version`, `environment`, `tier`）
5. **配置外部化**：使用 ConfigMap/Secret 管理配置，不要硬编码

掌握 Kubernetes 的核心概念和 kubectl 命令，是云原生时代运维工程师的必备技能。
