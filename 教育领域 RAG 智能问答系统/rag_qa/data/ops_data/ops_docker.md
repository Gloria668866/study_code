# Docker 容器技术详解

## 一、容器技术概述

Docker 是目前最主流的容器化平台，它通过操作系统级虚拟化实现了应用的快速打包、分发和运行。与虚拟机不同，Docker 容器共享宿主机内核，启动速度快（秒级），资源开销小。

### 容器 vs 虚拟机对比

| 特性 | Docker 容器 | 传统虚拟机 |
|------|------------|-----------|
| 启动速度 | 秒级 | 分钟级 |
| 资源隔离 | 进程级（Namespace） | 硬件级（Hypervisor） |
| 磁盘占用 | MB 级别 | GB 级别 |
| 性能损耗 | 约 2-3% | 约 10-20% |
| 镜像管理 | 分层存储，增量更新 | 完整快照 |

## 二、核心概念：镜像、容器、仓库

### 2.1 镜像（Image）

镜像是一个只读模板，包含运行应用所需的操作系统、运行时、代码和依赖。镜像采用分层存储的 Union FS（联合文件系统），每一层代表一个操作。

```bash
# 拉取镜像
docker pull nginx:1.25-alpine
docker pull python:3.11-slim

# 查看镜像
docker images
docker image ls --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# 查看镜像分层历史
docker history nginx:latest

# 删除镜像
docker rmi <image_id>
docker image prune -a   # 删除所有未使用的镜像
```

### 2.2 容器（Container）

容器是镜像的运行实例，可以在其上附加一个可写层来存储运行时数据。

```bash
# 运行容器
docker run -d --name web -p 8080:80 nginx:latest
docker run -it --rm ubuntu:22.04 /bin/bash   # 交互式，退出后自动删除

# 容器生命周期管理
docker ps -a                  # 查看所有容器（包括停止的）
docker start web              # 启动已停止的容器
docker stop web               # 优雅停止（SIGTERM + 10s等待 + SIGKILL）
docker restart web            # 重启
docker rm web                 # 删除容器
docker rm -f web              # 强制删除运行中的容器

# 进入容器
docker exec -it web /bin/bash # 新建终端进入
docker attach web             # 连接主进程

# 查看容器信息
docker logs -f --tail 100 web # 查看日志
docker inspect web            # 查看容器详细信息（JSON格式）
docker stats                  # 实时查看容器资源使用
docker top web                # 查看容器内进程
```

### 2.3 仓库（Registry）

仓库用于存储和分发 Docker 镜像，Docker Hub 是默认的公共仓库。

```bash
# 登录仓库
docker login registry.example.com -u username

# 推送/拉取镜像
docker tag myapp:latest registry.example.com/myapp:v1.0
docker push registry.example.com/myapp:v1.0
docker pull registry.example.com/myapp:v1.0

# 搭建私有仓库
docker run -d -p 5000:5000 --name registry registry:2
```

## 三、Dockerfile 编写

Dockerfile 是构建镜像的蓝图，下面是一个多阶段构建的 Python 应用示例：

```dockerfile
# ---- 第一阶段：构建阶段 ----
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --target=/install

# ---- 第二阶段：运行阶段 ----
FROM python:3.11-slim

# 创建非root用户
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY app.py .

# 声明端口
EXPOSE 8080

# 切换到非root用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# 启动命令
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
```

### Dockerfile 常用指令

| 指令 | 说明 | 最佳实践 |
|------|------|----------|
| `FROM` | 基础镜像 | 优先使用 alpine/slim 版本减小体积 |
| `WORKDIR` | 工作目录 | 使用绝对路径，避免 `cd` |
| `COPY` vs `ADD` | 复制文件 | 优先使用 COPY（更透明），ADD 用于 tar 解压 |
| `RUN` | 执行命令 | 合并多个命令减少层数 |
| `ENV` | 环境变量 | 敏感数据不要写在这里 |
| `ARG` | 构建参数 | 用于传递构建时参数 |
| `USER` | 运行时用户 | 不要以 root 运行 |
| `CMD` vs `ENTRYPOINT` | 启动命令 | CMD 可被覆盖，ENTRYPOINT 作为固定入口 |

### 构建优化技巧

```bash
# 构建镜像
docker build -t myapp:v1.0 -f Dockerfile .

# 使用 .dockerignore 排除文件
# 文件示例：
# .git
# __pycache__
# *.pyc
# .env
# node_modules

# 利用缓存（先复制依赖文件再复制源码，依赖未变时可复用缓存）
```

## 四、容器数据管理

### 4.1 数据卷（Volume）

```bash
# 创建和管理卷
docker volume create app_data
docker volume ls
docker volume inspect app_data

# 挂载卷运行容器
docker run -d -v app_data:/data --name db mysql:8
docker run -d -v /host/path:/container/path nginx  # 绑定挂载
```

### 4.2 数据卷 vs 绑定挂载

| 特性 | Volume | Bind Mount |
|------|--------|------------|
| 管理方式 | Docker 管理 | 宿主机路径 |
| 可移植性 | 高 | 低，依赖宿主机路径 |
| 性能 | 较好（Linux原生） | 一般 |

## 五、Docker Compose 多容器编排

Compose 通过 YAML 文件定义和运行多个容器。

```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: postgres:15-alpine
    container_name: postgres_db
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pg_data:/var/lib/postgresql/data
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d myapp"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: redis_cache
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  app:
    build:
      context: .
      dockerfile: Dockerfile
    image: myapp:latest
    container_name: myapp_server
    ports:
      - "8080:8080"
    environment:
      DB_HOST: db
      REDIS_HOST: redis
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - backend
      - frontend
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: nginx_proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    networks:
      - frontend

volumes:
  pg_data:

networks:
  backend:
    driver: bridge
  frontend:
    driver: bridge
```

### Compose 常用命令

```bash
docker compose up -d          # 启动所有服务（后台运行）
docker compose down           # 停止并删除
docker compose down -v        # 同时删除数据卷
docker compose ps             # 查看服务状态
docker compose logs -f app    # 查看指定服务日志
docker compose exec app bash  # 进入服务容器
docker compose restart nginx  # 重启指定服务
```

## 六、容器网络模式

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| bridge | 默认模式，独立网络命名空间，通过 docker0 网桥通信 | 单机多容器通信 |
| host | 直接使用宿主机网络栈 | 高性能网络需求 |
| none | 无网络 | 离线/安全场景 |
| overlay | 跨主机容器网络（Swarm/K8s） | 集群部署 |
| macvlan | 分配物理MAC地址 | 需要直接访问物理网络 |

```bash
# 自定义网络
docker network create --driver bridge --subnet 172.20.0.0/16 mynet
docker run -d --network mynet --name app1 nginx
docker network inspect mynet
```

## 七、Docker 安全最佳实践

1. **最小权限原则**：容器不要以 root 运行，设置 `USER` 指令
2. **镜像扫描**：使用 `docker scan` 或 Trivy 扫描镜像漏洞
3. **只读文件系统**：`docker run --read-only` 防止意外修改
4. **资源限制**：设置 CPU 和内存限制防止资源耗尽
   ```bash
   docker run -d --cpus="1.5" --memory="512m" --memory-swap="1g" nginx
   ```
5. **密钥管理**：不要将密码硬编码在镜像中，使用 Docker Secrets 或环境变量注入
6. **定期更新**：基础镜像和依赖保持最新版本

Docker 的核心价值在于"一次构建，处处运行"，通过标准化的打包方式解决了环境不一致问题，是现代 DevOps 的基石技术。
