#!/usr/bin/env_log.log python
"""
测试 ticket_server.py 的客户端代码
参考 douban_A2Aagent_client_noStreaming.py 的实现：
- 使用彩色日志记录过程
- 获取助手信息
- 解析 JSON 响应，生成用户友好总结
- 测试查询匹配数据库数据
"""

import asyncio
import json
import colorlog
import logging
from python_a2a import A2AClient
from datetime import datetime, timedelta
import pytz
import requests  # 添加 requests 以直接调用 API，绕过 ask 方法问题

# 设置彩色日志
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={'INFO': 'green', 'ERROR': 'red'}
))
logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(colorlog.INFO)


def generate_summary(query, response):
    """生成总结，参考豆瓣的解说员风格"""
    try:
        if isinstance(response, str):
            return f"**票务总结**:\n查询 '{query}' 返回字符串响应（可能服务器错误）：{response}。请检查服务器或稍后重试！"

        # 解析 JSON 响应
        data = response["parts"][0]["data"] if response.get("parts") else response
        if data.get("status") == "no_data":
            return f"**票务总结**:\n抱歉，查询 '{query}' 未找到结果。如果需要其他日期或类型，请补充更多细节！"
        elif data.get("status") == "error":
            return f"**票务总结**:\n查询 '{query}' 失败：{data['message']}。请检查输入或稍后重试！"
        elif data.get("status") == "success":
            result_text = []
            for item in data.get("data", []):
                if "train_number" in item:
                    result_text.append(
                        f"{item['departure_city']} 到 {item['arrival_city']} {item['departure_time']}: "
                        f"车次 {item['train_number']}，{item['seat_type']}，票价 {item['price']}元，剩余 {item['remaining_seats']} 张"
                    )
                elif "flight_number" in item:
                    result_text.append(
                        f"{item['departure_city']} 到 {item['arrival_city']} {item['departure_time']}: "
                        f"航班 {item['flight_number']}，{item['cabin_type']}，票价 {item['price']}元，剩余 {item['remaining_seats']} 张"
                    )
                elif "artist" in item:
                    result_text.append(
                        f"{item['city']} {item['start_time']}: {item['artist']} 演唱会，{item['ticket_type']}，"
                        f"场地 {item['venue']}，票价 {item['price']}元，剩余 {item['remaining_seats']} 张"
                    )
            result_text = "\n".join(result_text) if result_text else "无结果"
            return f"**票务总结**:\n欢迎来到票务查询助手！根据您的查询 '{query}'，为您找到：\n{result_text}\n继续探索更多票务信息吧！"
        else:
            return f"**票务总结**:\n查询 '{query}' 返回未知格式，请稍后重试！"
    except Exception as e:
        return f"**票务总结**:\n解析响应失败：{str(e)}。请检查查询 '{query}' 或稍后重试！"


async def test_ticket_server():
    """测试票务服务器"""
    client = A2AClient("http://localhost:5006")

    # 获取代理信息
    try:
        logger.info("获取票务助手信息")
        logger.info(f"名称: {client.agent_card.name}")
        logger.info(f"描述: {client.agent_card.description}")
        logger.info(f"版本: {client.agent_card.version}")
        if client.agent_card.skills:
            logger.info("支持技能:")
            for skill in client.agent_card.skills:
                logger.info(f"- {skill.name}: {skill.description}")
                if skill.examples:
                    logger.info(f"  示例: {', '.join(skill.examples)}")
    except Exception as e:
        logger.error(f"无法获取票务助手信息: {str(e)}")

    # 测试用例（匹配数据库数据）
    queries = [
        "火车票 北京 上海 2025-08-26 二等座",  # 匹配 mcp_weather_client_test.py 数据
        "机票 上海 北京 2025-09-21 公务舱",  # 匹配 mcp_weather_client_test.py 数据
        "演唱会 北京 刀郎 2025-08-23 看台",  # 已验证成功
        "火车票",  # 模糊查询
        "不存在的票务 2025-07-31",  # 异常查询
    ]

    url = "http://localhost:5006"  # 使用服务器 URL 直接调用 API

    for query in queries:
        logger.info(f"测试票务查询: {query}")
        try:
            # 构建任务并发送（绕过 ask 方法，直接获取完整响应）
            task = {"message": {"content": {"text": query}}}
            r = requests.post(f"{url}/tasks/send", json=task)
            if r.status_code != 200:
                raise Exception(f"发送任务失败: {r.text}")

            task_resp = r.json()
            state = task_resp["status"]["state"]

            if state == "completed":
                # 提取 artifacts
                if "artifacts" in task_resp and task_resp["artifacts"]:
                    response = task_resp["artifacts"][0]
                else:
                    response = {"parts": [{"data": {"status": "error", "message": "无 artifacts"}}]}
            elif state == "input_required":
                # 提取 message 并包装为类似 JSON 格式
                message_text = task_resp["status"]["message"]["content"]["text"]
                response = {"parts": [{"data": {"status": "input_required", "message": message_text}}]}
            else:
                response = {"parts": [{"data": {"status": "error", "message": f"未知状态: {state}"}}]}

            logger.info(f"原始响应: {response}")
            summary = generate_summary(query, response)
            logger.info(summary)
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            summary = f"**票务总结**:\n查询 '{query}' 失败：{str(e)}。请检查输入或稍后重试！"
            logger.info(summary)


async def main():
    logger.info("开始测试票务服务器...")
    await test_ticket_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")