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

async def test_ticket_mcp():
    port = 6002
    client = MCPClient(f"http://localhost:{port}")
    try:
        # 获取工具列表
        tools = await client.get_tools()
        logger.info("票务 MCP 可用工具：")
        for tool in tools:
            logger.info(f"- {tool.get('name', '未知')}: {tool.get('description', '无描述')}")

        # 测试1: 查询机票
        sql_flights = "SELECT * FROM flight_tickets WHERE departure_city = '上海' AND arrival_city = '北京' AND DATE(departure_time) = '2025-09-21' AND cabin_type = '公务舱'"
        result_flights = await client.call_tool("query_tickets", sql=sql_flights)
        result_flights_data = json.loads(result_flights) if isinstance(result_flights, str) else result_flights
        logger.info(f"机票查询结果：{result_flights_data}")

        # 测试2: 查询火车票
        sql_trains = "SELECT * FROM train_tickets WHERE departure_city = '北京' AND arrival_city = '上海' AND DATE(departure_time) = '2025-08-26' AND seat_type = '二等座'"
        result_trains = await client.call_tool("query_tickets", sql=sql_trains)
        result_trains_data = json.loads(result_trains) if isinstance(result_trains, str) else result_trains
        logger.info(f"火车票查询结果：{result_trains_data}")

        # 测试3: 查询演唱会票
        sql_concerts = "SELECT * FROM concert_tickets WHERE city = '北京' AND artist = '刀郎' AND DATE(start_time) = '2025-08-23' AND ticket_type = '看台'"
        result_concerts = await client.call_tool("query_tickets", sql=sql_concerts)
        result_concerts_data = json.loads(result_concerts) if isinstance(result_concerts, str) else result_concerts
        logger.info(f"演唱会票查询结果：{result_concerts_data}")

    except Exception as e:
        logger.error(f"票务 MCP 测试出错：{str(e)}", exc_info=True)
    finally:
        await client.close()

async def main():
    # await test_weather_mcp()
    await test_ticket_mcp()

if __name__ == "__main__":
    asyncio.run(main())