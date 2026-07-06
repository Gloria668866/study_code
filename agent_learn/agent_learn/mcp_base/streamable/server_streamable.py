#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: server_streamable.py
作者: ZZS
项目: LlmProject
创建日期: 2026/2/1
描述: 
"""
from mcp.server.fastmcp import FastMCP

# 创建 MCP 实例，指定服务名称、日志级别、主机和端口
mcp = FastMCP("sdg", log_level="ERROR", host="127.0.0.1", port=8001)

@mcp.tool(
    name="query_high_frequency_question",
    description="从知识库中检索常见问题解答（FAQ）,返回包含问题和答案的结构化JSON数据。",
)
async def query_high_frequency_question() -> str:
    """
    高频问题查询
    """
    try:
        print("调用查询高频问题的tool成功！！")
        return "高频问题是: 恐龙是怎么灭绝的？"
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise

@mcp.tool(
    name="get_weather",
    description="查询天气"
)
async def get_weather() -> str:
    """
    查询天气的tools
    """
    try:
        print("调用查询天气的tools")
        return "北京的天气是多云"
    except Exception as e:
        print(f"Unexpected error in question retrieval: {str(e)}")
        raise


def main():
    """
    启动 Streamable HTTP 服务器。
    """
    print("正在启动MCP Streamable服务器...")
    print("服务器将在 http://localhost:8001 上运行")
    print("按 Ctrl+C 停止服务器")
    try:
        mcp.run(transport="streamable-http")  # 使用 streamable-http 传输方式
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e}")


if __name__ == "__main__":
    main()