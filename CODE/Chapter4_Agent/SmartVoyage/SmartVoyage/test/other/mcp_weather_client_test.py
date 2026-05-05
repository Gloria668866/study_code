# mcp_client_test.py
import asyncio
import json
import logging
from python_a2a.mcp import MCPClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_weather_mcp():
    port = 6001
    client = MCPClient(f"http://localhost:{port}")
    try:
        # 获取工具列表
        tools = await client.get_tools()
        logger.info("天气 MCP 可用工具：")
        for tool in tools:
            logger.info(f"- {tool.get('name', '未知')}: {tool.get('description', '无描述')}")

        # 测试1: 查询指定日期天气
        sql = "SELECT * FROM weather_data WHERE city = '北京' AND fx_date = '2025-07-30'"
        result = await client.call_tool("query_weather", sql=sql)
        result_data = json.loads(result) if isinstance(result, str) else result
        logger.info(f"指定日期天气结果：{result_data}")

        # 测试2: 查询未来5天天气
        sql_range = "SELECT * FROM weather_data WHERE city = '北京' AND fx_date BETWEEN '2025-07-31' AND '2025-08-04'"
        result_range = await client.call_tool("query_weather", sql=sql_range)
        result_range_data = json.loads(result_range) if isinstance(result_range, str) else result_range
        logger.info(f"天气范围查询结果：{result_range_data}")
    except Exception as e:
        logger.error(f"天气 MCP 测试出错：{str(e)}", exc_info=True)
    finally:
        await client.close()


async def main():
    await test_weather_mcp()

if __name__ == "__main__":
    asyncio.run(main())