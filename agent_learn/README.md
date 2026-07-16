# Agent Learning - AI Agent 开发渐进式学习代码

本仓库包含从零到一的 AI Agent 开发学习路径，覆盖 **Function Call → MCP 协议 → Agent 设计模式 → A2A 协议** 完整技术栈。

## 📚 学习路径

### Phase 1: Function Call（函数调用）
| 文件 | 内容 | 难度 |
|------|------|------|
| `C01_define_tool.py` | 原始 JSON Schema 定义工具 | ⭐ |
| `C02_by_annotation.py` | `@tool` 装饰器定义工具 | ⭐ |
| `C03_pydantic.py` | Pydantic 模型定义工具 | ⭐⭐ |
| `C04_by_agent.py` | LangChain Agent 自动调用工具 | ⭐⭐ |

### Phase 2: MCP 协议（Model Context Protocol）
| 传输方式 | Server | Client |
|----------|--------|--------|
| **stdio** | `server_stdio.py` | `client_raw.py` / `client_agent.py` |
| **SSE** | `server_sse.py` | `client_raw.py` / `client_agent.py` |
| **Streamable HTTP** | `server_streamable.py` | `client_raw.py` / `client_agent.py` |

### Phase 3: Agent 设计模式
| 文件 | 模式 | 说明 |
|------|------|------|
| `C01_ToolUsePattern.py` | 🛠️ Tool Use | Agent 调用工具回答问题 |
| `C02_ReActPattern.py` | 🤔 ReAct | 推理+行动循环 |
| `C03_ReflectionPattern.py` | 🔄 Reflection | 自我反思修正 |
| `C04_PlanningPattern.py` | 📋 Planning | 拆解任务分步执行 |
| `C05_MultiAgent.py` | 👥 Multi-Agent | 多专家协作工作流 |

### Phase 4: A2A 协议（Agent-to-Agent）
| 文件 | 内容 |
|------|------|
| `A2A/agent_card.py` | Agent 卡片定义 |
| `A2A/agent_skill.py` | Agent 技能定义 |
| `A2A/task.py` | 任务消息结构 |
| `A2A/tast_state.py` | 任务状态枚举 |
| `A2A/a2a_server_artifacts.py` | A2A Server 示例 |
| `A2A/a2a_client_artifacts.py` | A2A Client 示例 |
| `A2A/agentrouter.py` | Agent 路由 |
| `A2A/agentnetwork.py` | Agent 网络管理 |

### Phase 5: A2A 实战案例
| 文件 | 说明 |
|------|------|
| `a2a_case/weather_agent.py` | 天气查询 A2A Agent Server |
| `a2a_case/ticket_agent.py` | 票务预订 A2A Agent Server |
| `a2a_case/router_A2Aagent_Server.py` | LLM 路由器 A2A Server |
| `a2a_case/main.py` | 主控调用示例 |
| `A2A_base/a2a_serial/weather_agent.py` | 串行 A2A 天气 Agent |
| `A2A_base/a2a_serial/ticket_agent.py` | 串行 A2A 票务 Agent |
| `A2A_base/a2a_serial/main_orchestrator.py` | 串行编排主控 |

## 🚀 快速开始

```bash
# 安装依赖
pip install python-a2a langchain-openai langchain-mcp-adapters mcp

# 配置 API Key
# 编辑 config.py 填入你的 API Key 和 base_url

# Phase 1: 运行 Function Call 示例
python function_call/C01_define_tool.py

# Phase 2: 启动 MCP Server
python mcp_base/streamable/server_streamable.py

# Phase 3: 运行 Agent 模式
python agent_types/C01_ToolUsePattern.py

# Phase 4: 启动 A2A Agent
python a2a_case/weather_agent.py
python a2a_case/ticket_agent.py
python a2a_case/router_A2Aagent_Server.py
python a2a_case/main.py
```

## 📖 学习顺序建议

```
1. function_call/C01 → C04   理解 LLM 如何调用工具
2. mcp_base/*                 理解 MCP 协议的三种传输方式
3. agent_types/C01 → C05      掌握 5 种 Agent 设计模式
4. A2A/*                      理解 A2A 协议核心概念
5. A2A_base/a2a_serial/*      体验串行多 Agent 编排
6. a2a_case/*                 完成 A2A 实战案例
```
