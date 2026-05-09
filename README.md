# AI 赋能的智能简历分析系统

一个面向招聘筛选场景的智能简历分析 MVP，支持 PDF 简历上传、文本解析、结构化信息提取、岗位需求关键词分析与匹配评分，并支持可选缓存机制。

## 一、项目背景

在招聘流程中，HR 需要快速筛选大量简历。手工阅读简历、提取关键信息、与岗位需求做比对耗时较高。本项目实现了一个可在阿里云 Serverless 场景下部署的简历分析服务，帮助招聘者快速完成初筛。

## 二、实现目标

### 必选能力
- 上传单个 PDF 格式简历
- 解析多页 PDF 文本
- 清洗与结构化处理简历文本
- 提取关键信息：姓名、电话、邮箱、地址
- 接收岗位描述文本
- 提取岗位关键词并进行匹配评分
- 以 JSON 格式返回结果
- 提供简洁可用的交互页面

### 加分能力
- 支持 Redis 缓存，避免重复解析与评分
- 预留 AI 模型接入接口，便于后续替换成在线大模型
- 提供部署说明与可公开访问前端方案

## 三、技术选型

- **后端框架**：FastAPI
- **开发语言**：Python 3.10+
- **PDF 解析**：pypdf
- **缓存**：Redis（可选）
- **前端**：原生 HTML / CSS / JavaScript
- **部署目标**：阿里云 Serverless（函数计算 FC） + GitHub Pages

### 选型原因

1. **FastAPI** 适合 RESTful API 开发，性能好，文档自动生成。
2. **pypdf** 适合处理多页 PDF 文本提取，依赖轻，适合 Serverless 环境。
3. **Redis** 可作为缓存层，降低重复解析成本。
4. **原生前端** 不依赖复杂构建链，适合快速交付与 GitHub Pages 部署。

## 四、系统架构

### 架构图

```text
[用户浏览器]
    |
    | 上传 PDF / 输入岗位描述
    v
[前端页面 GitHub Pages]
    |
    | REST API 调用
    v
[FastAPI 后端服务]
    |
    |-- PDF 解析
    |-- 文本清洗
    |-- 关键信息提取
    |-- 岗位关键词分析
    |-- 匹配评分
    |-- JSON 返回
    |
    |-- 可选 Redis 缓存
    v
[结构化结果]
```

### 核心流程

1. 用户上传 PDF 简历并输入岗位描述。
2. 后端计算文件哈希值，先查询缓存。
3. 若无缓存，则解析 PDF 多页文本并清洗。
4. 提取姓名、电话、邮箱、地址等信息。
5. 对岗位描述进行关键词分析。
6. 计算关键词匹配率与经验相关性评分。
7. 返回结构化 JSON。

## 五、接口说明

### 1. 健康检查

`GET /health`

返回示例：

```json
{
  "status": "ok",
  "service": "AI赋能的智能简历分析系统"
}
```

### 2. 系统元信息

`GET /api/meta`

返回示例：

```json
{
  "service": "AI赋能的智能简历分析系统",
  "version": "1.0.0",
  "features": [
    "resume-upload",
    "pdf-parsing",
    "information-extraction",
    "job-matching",
    "json-response",
    "redis-cache-optional"
  ]
}
```

### 3. 简历分析

`POST /api/resume/analyze`

#### 表单参数
- `file`：PDF 简历文件，必填
- `job_description`：岗位描述，选填

#### 返回字段
- `file_name`：文件名
- `file_hash`：文件哈希
- `pages`：PDF 页数
- `raw_text`：原始文本
- `cleaned_text`：清洗后的文本
- `summary`：结构化关键信息
- `match`：匹配评分结果
- `cached`：是否命中缓存

#### 返回示例

```json
{
  "file_name": "resume.pdf",
  "file_hash": "a1b2c3...",
  "pages": 2,
  "raw_text": "...",
  "cleaned_text": "...",
  "summary": {
    "name": "张三",
    "phone": "13800000000",
    "email": "zhangsan@example.com",
    "address": "北京市海淀区",
    "job_intent": "Python后端开发",
    "expected_salary": "15K-20K",
    "work_years": "3",
    "education": "本科",
    "projects": ["简历分析系统"]
  },
  "match": {
    "keyword_score": 84.5,
    "experience_score": 60,
    "overall_score": 77.15,
    "matched_keywords": ["python", "fastapi"],
    "missing_keywords": ["redis"],
    "analysis": "岗位关键词共 3 个，匹配 2 个。候选人工作年限约 3 年。"
  },
  "cached": false
}
```

### 4. 仅分析岗位描述

`POST /api/job/match`

#### 表单参数
- `job_description`：岗位描述，必填

#### 返回示例

```json
{
  "job_description": "Python 后端开发，熟悉 FastAPI 和 Redis",
  "keywords": ["python", "后端开发", "fastapi", "redis"],
  "keyword_count": 4
}
```

## 六、本地运行

### 1. 启动后端

```bash
cd smart_resume_ai/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. 打开前端

直接打开 `smart_resume_ai/frontend/index.html`，或部署到 GitHub Pages。

### 3. 访问接口文档

启动后端后访问：

- Swagger UI：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

## 七、部署方式

### 1. 阿里云 Serverless / 函数计算 FC

建议将 FastAPI 作为 HTTP 入口部署到函数计算环境，使用 Python 运行时。

部署要点：
- 将 `backend/app/main.py` 作为服务入口
- 配置依赖包
- 如需缓存，设置 `REDIS_URL`
- 保持无状态设计，缓存与文件存储外置

### 2. GitHub Pages 前端

部署 `frontend/index.html` 到 GitHub Pages：

- 新建公开仓库
- 将 `frontend/index.html` 放到仓库根目录或 `docs` 目录
- 在 GitHub Pages 中选择对应分支/目录发布
- 修改前端中的 API 地址为线上后端地址

## 八、缓存设计

- 使用文件内容 SHA-256 + 岗位描述 SHA-256 作为缓存键
- 命中缓存时直接返回结果
- 未配置 Redis 时自动降级为直通模式，不影响主流程

## 九、AI 选型与扩展说明

本次交付版本采用“规则抽取 + 评分”的稳定 MVP 方案，以保证 24 小时内可交付、可演示、可验收。

同时在项目中预留了 AI 扩展位：
- 可替换为阿里云兼容大模型 API
- 可将结构化抽取和匹配评分交给 LLM 做增强
- 可加入更细粒度的岗位画像、技能图谱和语义相似度评分

## 十、项目结构

```text
smart_resume_ai/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   └── requirements.txt
├── frontend/
│   └── index.html
├── .env.example
└── README.md
```

## 十一、验收建议

评审时建议按以下顺序演示：
1. 打开前端页面
2. 上传 PDF 简历
3. 输入岗位描述
4. 展示 JSON 结果
5. 展示匹配分和结构化信息
6. 展示缓存命中效果

## 十二、说明

这是一个面向笔试提交的可交付 MVP，核心目标是快速、稳定、结构清晰地完成招聘简历筛选场景的基础能力实现。