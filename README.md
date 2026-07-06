# Study Code - AI Agent 开发学习代码

本仓库包含两个独立的 AI Agent 学习项目：

## 📁 项目结构

```
study_code/
├── agent_learn/       # 渐进式 AI Agent 开发学习路径
│   ├── function_call/    基础函数调用（4个等级）
│   ├── mcp_base/         MCP 协议（3种传输方式）
│   ├── agent_types/      Agent 设计模式（5种）
│   ├── A2A/              A2A 协议核心概念
│   ├── A2A_base/         A2A 串行编排
│   └── a2a_case/         A2A 实战案例
│
└── smart_voyage/      # 基于 A2A 协议的旅行智能助手
    ├── a2a_server/        A2A Agent 服务端
    ├── mcp_server/        MCP 工具服务端
    ├── utils/             工具函数（天气爬虫等）
    ├── sql/               数据库建表/数据
    ├── app.py             Streamlit Web 界面
    └── main.py            命令行交互界面
```

## 🔑 配置说明

使用前请配置 API Key：

1. **agent_learn**: 编辑 `agent_learn/agent_learn/config.py`
2. **smart_voyage**: 编辑 `smart_voyage/config.py`
3. **天气爬虫**: 编辑 `smart_voyage/utils/spider_weather.py`

需要自行申请：
- LLM API Key（DeepSeek / SiliconFlow / OpenAI）
- 和风天气 API Key（可选，仅天气爬虫需要）
- MySQL 数据库

## 🚀 快速开始

```bash
# 安装依赖
pip install python-a2a langchain-openai langchain-mcp-adapters mcp mysql-connector-python streamlit pytz schedule

# 配置 API Key 后运行 agent_learn 示例
cd agent_learn
python agent_learn/function_call/C01_define_tool.py

# 或运行 smart_voyage 旅行助手
cd smart_voyage
python main.py
```

详情见各子项目内的 README。
