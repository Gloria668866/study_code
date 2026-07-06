#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: client_raw.py
作者: ZZS
项目: LlmProject
创建日期: 2026/1/31
描述: 
"""
import asyncio
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 配置mcp服务器脚本路径
server_script = r".\server_stdio.py"

# 配置mcp服务器启动参数
server_params = StdioServerParameters(
    command="python" if server_script.endswith(".py") else "node",
    args=[server_script],
)

# 定义mcp客户端
mcp_client = None

# 主要的异步函数run_agent
async def run():
    global mcp_client
    # 启动 MCP server，并通过标准输入输出建立异步连接。
    async with stdio_client(server_params) as (read, write):
        # 使用读写通道创建 MCP 会话。
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 动态创建一个临时类 MCPClientHolder，把 session 放进去。这样就可以在函数外部通过 mcp_client.session 调用 MCP 工具
            mcp_client = type("MCPClientHolder", (), {"session": session})()

            # 从 session 自动获取 MCP server 提供的工具列表。
            tools = await load_mcp_tools(session)
            # print(f"tools-->{tools}")

            # 调用 MCP server 的 get_weather 工具
            response=await session.call_tool("get_weather", arguments={})
            print(f"response-->{response}")
    return

# 启动运行
if __name__ == "__main__":
    asyncio.run(run())


# mcp_client = type("MCPClientHolder", (), {"session": session})()
# class MCPClientHolder:
#     """
#     MCP客户端持有者类，用于存储和访问MCP会话对象
#     """
#     def __init__(self, session):
#         """
#         初始化方法，接收session参数并保存为实例属性
#
#         Args:
#             session: MCP工具的会话对象
#         """
#         # 将传入的session保存为实例属性
#         self.session = session