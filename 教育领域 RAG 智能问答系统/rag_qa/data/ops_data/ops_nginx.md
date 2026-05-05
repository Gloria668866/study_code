# Nginx反向代理与负载均衡

## Nginx概述

Nginx是一个高性能的HTTP和反向代理服务器，也是IMAP/POP3/SMTP代理服务器。由俄罗斯程序员Igor Sysoev开发。Nginx以其高并发处理能力（单机可达5万+并发连接）、低内存消耗和丰富的模块生态而闻名。

## 反向代理配置

### 基本反向代理

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### WebSocket代理

```nginx
location /ws/ {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400;
}
```

## 负载均衡策略

Nginx upstream模块支持多种负载均衡算法：

### 1. 轮询（Round Robin）- 默认策略
```nginx
upstream backend {
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
}
```

### 2. 权重（Weight）
```nginx
upstream backend {
    server 192.168.1.10:8080 weight=3;
    server 192.168.1.11:8080 weight=1;
}
```

### 3. IP Hash
```nginx
upstream backend {
    ip_hash;
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
}
```
同一IP的请求始终路由到同一台服务器，适合需要会话保持的场景。

### 4. 最少连接（Least Connections）
```nginx
upstream backend {
    least_conn;
    server 192.168.1.10:8080;
    server 192.168.1.11:8080;
}
```

### 5. Fair（第三方模块，按响应时间分配）

## HTTPS配置

```nginx
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /etc/nginx/ssl/example.com.pem;
    ssl_certificate_key /etc/nginx/ssl/example.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://backend;
    }
}

# HTTP强制跳转HTTPS
server {
    listen 80;
    server_name example.com;
    return 301 https://$server_name$request_uri;
}
```

## 性能优化

### 工作进程配置
```nginx
worker_processes auto;  # 自动匹配CPU核心数
worker_connections 10240;
worker_rlimit_nofile 65535;
```

### 缓存配置
```nginx
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=my_cache:10m max_size=10g inactive=60m;
proxy_cache my_cache;
proxy_cache_valid 200 302 10m;
proxy_cache_valid 404 1m;
```

### Gzip压缩
```nginx
gzip on;
gzip_comp_level 6;
gzip_types text/plain text/css application/json application/javascript text/xml;
gzip_min_length 1000;
```

### 静态文件优化
```nginx
location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
    expires 30d;
    add_header Cache-Control "public, immutable";
}
```

## 常用运维命令

```bash
nginx -t              # 测试配置文件语法
nginx -s reload       # 热重载配置（不停机）
nginx -s stop         # 快速停止
nginx -s quit         # 优雅停止（处理完当前请求）
```
