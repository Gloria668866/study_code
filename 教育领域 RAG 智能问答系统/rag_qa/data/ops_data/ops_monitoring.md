# 监控告警体系

## 一、监控体系概述

### 1.1 监控的四个黄金信号

Google SRE 提出的四个核心监控指标：

| 指标 | 说明 | 示例 |
|------|------|------|
| 延迟（Latency） | 请求处理时间 | P50/P90/P99 响应时间 |
| 流量（Traffic） | 系统请求量 | QPS、网络IO、数据库连接数 |
| 错误（Errors） | 失败请求比例 | HTTP 5xx 比例、异常率 |
| 饱和度（Saturation） | 资源饱和程度 | CPU 使用率、内存使用率、队列长度 |

### 1.2 监控体系层次

```
┌──────────────────────────────────────┐
│         业务监控（订单量/转化率）       │  ← 最上层
├──────────────────────────────────────┤
│       应用监控（QPS/错误率/延迟）       │
├──────────────────────────────────────┤
│    中间件监控（MySQL/Redis/MQ/ES）      │
├──────────────────────────────────────┤
│   基础设施监控（CPU/内存/磁盘/网络）     │  ← 最底层
└──────────────────────────────────────┘
```

## 二、Prometheus 指标采集

### 2.1 Prometheus 架构

Prometheus 是云原生监控的事实标准，采用 Pull 模型定时抓取目标的指标数据。

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│ Exporter │   │ Exporter │   │ Short-lived │
│ (Node)   │   │ (MySQL)  │   │   Jobs       │
└────┬─────┘   └────┬─────┘   └──────┬───────┘
     │              │                │
     ▼              ▼                ▼
┌────────────┐  ┌──────────────────────────┐
│ Prometheus │──│   Alertmanager  ←告警路由  │
│   Server   │  └──────────────────────────┘
└─────┬──────┘
      │
      ▼
┌──────────┐   ┌──────────┐
│ Grafana  │   │  PromQL  │
│ (可视化)  │   │  (查询)   │
└──────────┘   └──────────┘
```

### 2.2 Prometheus 配置文件

```yaml
# prometheus.yml
global:
  scrape_interval: 15s      # 采集间隔
  evaluation_interval: 15s  # 告警规则评估间隔
  external_labels:
    cluster: 'prod-cluster-01'
    datacenter: 'cn-east-1'

# 告警规则
rule_files:
  - 'alerts/*.yml'

# 采集目标
scrape_configs:
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

  - job_name: 'node-exporter'
    static_configs:
      - targets:
        - '10.0.1.10:9100'
        - '10.0.1.11:9100'
        - '10.0.1.12:9100'
```

### 2.3 常用 Exporter

| Exporter | 端口 | 监控对象 |
|----------|------|----------|
| node_exporter | 9100 | Linux 主机（CPU/内存/磁盘/网络） |
| mysql_exporter | 9104 | MySQL 数据库 |
| redis_exporter | 9121 | Redis 缓存 |
| nginx-prometheus-exporter | 9113 | Nginx 指标 |
| kube-state-metrics | 8080 | K8s 资源对象状态 |
| blackbox_exporter | 9115 | HTTP/DNS/TCP 探针 |

### 2.4 PromQL 查询语言

```promql
# 基础查询
node_cpu_seconds_total{mode="idle"}             # CPU空闲时间
node_memory_MemAvailable_bytes / 1024 / 1024    # 可用内存(MB)

# 聚合查询
sum(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (instance)
# 每台主机的非空闲CPU使用率

# 函数查询
rate(http_requests_total{status="500"}[5m])     # 5分钟内500错误速率
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
# P99 响应时间

# 预测
predict_linear(node_filesystem_free_bytes{mountpoint="/"}[1h], 4 * 3600)
# 预测4小时后磁盘空间剩余
```

### 2.5 告警规则

```yaml
# alerts/cpu.yml
groups:
- name: node-alerts
  rules:
  - alert: HighCPUUsage
    expr: 100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "节点 {{ $labels.instance }} CPU使用率超过80%"
      description: "CPU使用率 {{ $value | humanize }}%，已持续5分钟"

  - alert: DiskAlmostFull
    expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 10
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "磁盘空间不足 ({{ $labels.instance }} /)"

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "服务 {{ $labels.job }} 在 {{ $labels.instance }} 上已停止"
```

## 三、Grafana 可视化看板

### 3.1 核心概念

| 概念 | 说明 |
|------|------|
| Data Source | 数据源（Prometheus、Elasticsearch、MySQL 等） |
| Dashboard | 看板，包含多个 Panel |
| Panel | 面板，每个面板展示一个图表 |
| Variable | 变量，用于动态筛选（如选择主机、服务） |
| Alert | Grafana 也支持内置告警 |

### 3.2 看板设计原则

1. **分层组织**：按照基础设施 → 中间件 → 应用 → 业务的层次组织看板
2. **RED 方法**：每个服务的看板应包括 Rate（QPS）、Errors（错误率）、Duration（延迟）
3. **合理聚合**：重要图表（概览）→ 详细信息图表（钻取），避免一张图包含过多指标
4. **阈值标注**：在图表上标注告警阈值，直观判断是否正常

### 3.3 常用 Dashboard 模板

```
# Grafana 社区模板 ID
Node Exporter Full       : 1860   # 主机监控
K8s Cluster Overview     : 7249   # K8s集群概览
Nginx Ingress Controller : 9614   # Nginx Ingress监控
Spring Boot Statistics   : 6756   # Java应用监控
MySQL Overview           : 7362   # MySQL数据库监控
```

## 四、ELK 日志收集

### 4.1 ELK 架构

ELK 由 Elasticsearch、Logstash、Kibana 三个组件组成（现在加入 Filebeat 称为 ELK Stack）。

```
┌──────────┐     ┌───────────┐     ┌──────────────┐     ┌──────────┐
│ Filebeat │────▶│ Logstash  │────▶│ Elasticsearch│◀────│ Kibana   │
│ (采集)   │     │ (处理过滤) │     │  (存储索引)   │     │ (可视化) │
└──────────┘     └───────────┘     └──────────────┘     └──────────┘

Filebeat 替代方案: Fluentd / Fluent Bit (K8s 中更常用)
```

### 4.2 Filebeat 配置

```yaml
# filebeat.yml
filebeat.inputs:
- type: filestream
  id: app-logs
  enabled: true
  paths:
    - /var/log/app/*.log
  fields:
    app: myapp
    environment: production
  fields_under_root: true

- type: filestream
  id: nginx-access
  paths:
    - /var/log/nginx/access.log
  parsers:
    - ndjson:
        keys_under_root: true

output.elasticsearch:
  hosts: ["es01:9200", "es02:9200", "es03:9200"]
  index: "logs-%{[agent.version]}-%{+yyyy.MM.dd}"
  # 索引生命周期管理
  ilm:
    enabled: true
    policy_name: logs-policy
    rollover_alias: logs
```

### 4.3 Logstash Pipeline 配置

```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  # 解析Nginx日志
  if [fields][app] == "nginx" {
    grok {
      match => {
        "message" => '%{IPORHOST:client_ip} - %{DATA:remote_user} \[%{HTTPDATE:timestamp}\] "%{WORD:method} %{DATA:request} HTTP/%{NUMBER:http_version}" %{NUMBER:status} %{NUMBER:bytes} "%{DATA:referrer}" "%{DATA:user_agent}"'
      }
      remove_field => ["message"]
    }
    geoip {
      source => "client_ip"
      target => "geoip"
    }
  }
  
  # 解析JSON日志
  if [fields][app] == "myapp" {
    json {
      source => "message"
    }
  }
}

output {
  elasticsearch {
    hosts => ["es01:9200", "es02:9200"]
    index => "logs-%{[@metadata][beat]}-%{+YYYY.MM.dd}"
  }
}
```

### 4.4 Kibana 常用操作

- **Discover**：原始日志搜索和浏览
- **Visualize**：创建可视化图表（柱状图、饼图、折线图等）
- **Dashboard**：组合多个图表
- **Dev Tools**：直接操作 ES API

```json
// Dev Tools - 查询最近15分钟的错误日志
GET /logs-*/_search
{
  "query": {
    "bool": {
      "must": [
        {"match": {"level": "ERROR"}},
        {"range": {"@timestamp": {"gte": "now-15m"}}}
      ]
    }
  },
  "sort": [{"@timestamp": {"order": "desc"}}],
  "size": 100
}
```

## 五、告警体系设计

### 5.1 告警分级

| 级别 | 说明 | 响应时间 | 通知方式 |
|------|------|----------|----------|
| Critical | 生产服务不可用 | 5分钟 | 电话 + 短信 + 即时通讯 |
| Warning | 资源使用率高、部分故障 | 30分钟 | 即时通讯（钉钉/飞书/Slack） |
| Info | 可关注但无需立即处理 | 次日 | 邮件 |

### 5.2 告警收敛

避免告警风暴的关键策略：
- **告警分组**：Prometheus Alertmanager 的 `group_by` 机制
- **告警抑制**：当 A 告警发生时，抑制相关的 B 告警
- **告警静默**：计划维护期间手动静默
- **重复间隔**：设置 `repeat_interval` 避免频繁重复通知

### 5.3 Alertmanager 配置

```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'cluster']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  routes:
  - match:
      severity: critical
    receiver: 'ops-oncall'
    continue: true
  - match:
      severity: warning
    receiver: 'ops-channel'

receivers:
- name: 'ops-oncall'
  webhook_configs:
  - url: 'https://hooks.slack.com/services/xxx/yyy/zzz'
    send_resolved: true
- name: 'ops-channel'
  webhook_configs:
  - url: 'https://oapi.dingtalk.com/robot/send?access_token=xxx'
```

## 六、实践总结

1. **先监控后开发**：新服务上线前先接入监控和告警
2. **指标标准化**：统一使用 Prometheus metrics 格式，定义统一的 label 规范
3. **日志结构化**：日志输出使用 JSON 格式，包含 trace_id 便于链路追踪
4. **告警有效性**：定期回顾告警，清除无效告警，避免"狼来了"效应
5. **可观测性三支柱**：Metrics（指标）+ Logging（日志）+ Tracing（链路追踪）三管齐下

监控体系的最终目标不是收集海量数据，而是让团队能够快速发现、定位和解决问题，保障服务稳定运行。
