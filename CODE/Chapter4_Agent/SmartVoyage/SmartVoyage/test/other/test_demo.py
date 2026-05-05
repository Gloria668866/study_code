#!/usr/bin/env_log.log python
"""
A2A 客户端测试代码，用于测试 ticket_server.py 和 weather_server.py
"""
import asyncio
import json
import aiohttp
import logging
from datetime import datetime, timedelta
import pytz

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class A2AClient:
    def __init__(self, server_url):
        self.server_url = server_url

    async def send_query(self, session, query_text):
        """发送查询并处理响应"""
        payload = {
            "message": {
                "content": {"text": query_text},
                "role": "user"
            }
        }
        try:
            async with session.post(
                f"{self.server_url}/task",
                json=payload,
                timeout=10
            ) as response:
                if response.status != 200:
                    logger.error(f"请求失败，状态码: {response.status}")
                    return {"status": "error", "message": f"HTTP {response.status}"}

                result = await response.json()
                task_status = result.get("status", {})
                state = task_status.get("state", "")
                logger.info(f"任务状态: {state}")

                if state == "INPUT_REQUIRED":
                    message = task_status.get("message", {}).get("content", {}).get("text", "需要更多信息")
                    return {"status": "input_required", "message": message}
                elif state == "COMPLETED":
                    artifacts = result.get("artifacts", [])
                    if artifacts and "parts" in artifacts[0]:
                        text = artifacts[0]["parts"][0].get("text", "无结果")
                        return {"status": "completed", "message": text}
                return {"status": "error", "message": "未知响应状态"}
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            return {"status": "error", "message": str(e)}

async def test_ticket_server():
    """测试票务服务器"""
    client = A2AClient("http://localhost:5006")
    async with aiohttp.ClientSession() as session:
        # 测试用例 1: 火车票查询
        logger.info("测试火车票查询: 北京到上海 明天 硬卧")
        tomorrow = (datetime.now(pytz.timezone('Asia/Shanghai')) + timedelta(days=1)).strftime('%Y-%m-%d')
        query = f"火车票 北京 上海 {tomorrow} 硬卧"
        result = await client.send_query(session, query)
        logger.info(f"结果: {result['message']}")

        # 测试用例 2: 机票查询
        logger.info("测试机票查询: 上海到广州 明天 经济舱")
        query = f"机票 上海 广州 {tomorrow} 经济舱"
        result = await client.send_query(session, query)
        logger.info(f"结果: {result['message']}")

        # 测试用例 3: 演唱会查询
        logger.info("测试演唱会查询: 北京 刀郎 2025-08-23 看台")
        query = "演唱会 北京 刀郎 2025-08-23 看台"
        result = await client.send_query(session, query)
        logger.info(f"结果: {result['message']}")

        # 测试用例 4: 模糊查询（缺少信息）
        logger.info("测试模糊查询: 火车票")
        query = "火车票"
        result = await client.send_query(session, query)
        logger.info(f"结果: {result['message']}")

        # 测试用例 5: 追问逻辑
        logger.info("测试追问: 北京到上海")
        query = "北京到上海"
        result = await client.send_query(session, query)
        logger.info(f"结果: {result['message']}")
        if result["status"] == "input_required":
            logger.info("追问: 明天 硬卧")
            query = f"明天 硬卧"
            result = await client.send_query(session, query)
            logger.info(f"结果: {result['message']}")

async def test_weather_server():
    """测试天气服务器"""
    client = A2AClient("http://localhost:5005")
    async with aiohttp.ClientSession() as session:
        # 测试用例 1: 单日天气
        logger.info("测试单日天气: 北京 明天")
        tomorrow = (datetime.now(pytz.timezone('Asia/Shanghai')) + timedelta(days=1)).strftime('%Y-%m-%d')
        query = f"北京 {tomorrow}"
        result = await client.send_query(session, query)
        logger.info(f"结果: {result['message']}")

        # 测试用例 2: 多日天气
        logger.info("测试多日天气: 上海未来3天")
        query = "上海未来3天"
        result = await client.send_query(session, query)
        logger.info(f"结果: {result['message']}")

        # 测试用例 3: 模糊查询（缺少信息）
        logger.info("测试模糊查询: 今天")
        query = "今天"
        result = await client.send_query(session, query)
        logger.info(f"结果: {result['message']}")

        # 测试用例 4: 追问逻辑
        logger.info("测试追问: 北京明天")
        query = "北京明天"
        result = await client.send_query(session, query)
        logger.info(f"结果: {result['message']}")
        if result["status"] == "input_required":
            logger.info("追问: 后天")
            query = "后天"
            result = await client.send_query(session, query)
            logger.info(f"结果: {result['message']}")

async def main():
    logger.info("开始测试票务服务器...")
    await test_ticket_server()
    logger.info("\n开始测试天气服务器...")
    await test_weather_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")