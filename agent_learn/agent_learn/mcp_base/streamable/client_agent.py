#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: client_agent.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
import json
import logging
import asyncio
from langchain_openai import ChatOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from agent_learn.config import Config

conf = Config()

# 创建模型
llm = ChatOpenAI(base_url=conf.base_url,
                 api_key=conf.api_key,
                 model=conf.model_name,
                 temperature=0.1)

# MCP 服务器的 Streamable-HTTP 连接地址
server_url = "http://127.0.0.1:8001/mcp"

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 提高日志级别以捕获更多信息
    format='[客户端] %(asctime)s - %(levelname)s - %(message)s'
)

# 定义mcp客户端
mcp_client = None

async def run_agent():
    global mcp_client
    logging.info(f"准备连接到 Streamable-HTTP 服务器: {server_url}")
    # 启动 MCP server，通过streamable建立连接
    async with streamablehttp_client(server_url) as (read, write, _):
        logging.info("连接已成功建立！")
        # 使用读写通道创建 MCP 会话
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
                logging.info("会话初始化成功，可以开始加载工具。")
                # 动态创建一个临时类 MCPClientHolder，把 session 放进去。这样就可以在函数外部通过 mcp_client.session 调用 MCP 工具
                mcp_client = type("MCPClientHolder", (), {"session": session})()

                # 从 session 自动获取 MCP server 提供的工具列表。
                tools = await load_mcp_tools(session)
                # print(f"tools-->{tools}")

                # 创建 agent 的提示模板
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个乐于助人的助手，能够调用工具回答用户问题。"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])

                # 构建工具调用代理
                agent = create_tool_calling_agent(llm, tools, prompt)

                # 创建代理执行器
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                # 代理调用
                print("MCP客户端启动，输入'quit'退出")
                while True:
                    query = input("\nQuery: ").strip()
                    if query.lower() == "quit":
                        break
                    # 发送用户查询到 agent 并打印格式化响应
                    logging.info(f"处理用户查询: {query}")
                    try:
                        response = await agent_executor.ainvoke({"input": query})
                        print(f"response-->{response}")
                    except Exception:
                        print("解析有问题")
            except Exception as e:
                logging.error(f"会话初始化或工具调用时发生错误: {e}", exc_info=True)
                raise

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except Exception as e:
        logging.error(f"客户端运行失败: {e}", exc_info=True)