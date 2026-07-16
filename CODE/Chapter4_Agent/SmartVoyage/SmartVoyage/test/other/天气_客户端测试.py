#!/usr/bin/env_log.log python
"""
测试 weather_server.py 的客户端代码
参考 douban_A2Aagent_client_noStreaming.py 的实现：
- 使用彩色日志记录过程
- 获取助手信息
- 解析响应，生成用户友好总结
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
    try:
        if isinstance(response, str):
            return f"**天气总结**:\n查询 '{query}' 返回字符串响应（可能服务器错误）：{response}。请检查服务器或稍后重试！"

        # 解析响应
        if response.get("parts"):
            data = response["parts"][0]
            if data.get("type") == "text":
                return f"**天气总结**:\n欢迎来到天气查询助手！根据您的查询 '{query}'，为您找到：\n{data['text']}\n继续探索更多天气信息吧！"
        elif response.get("status") == "input_required":
            return f"**天气总结**:\n查询 '{query}' 需要更多信息：{response['message']}。请补充细节！"
        else:
            return f"**天气总结**:\n查询 '{query}' 返回未知格式，请稍后重试！"
    except Exception as e:
        return f"**天气总结**:\n解析响应失败：{str(e)}。请检查查询 '{query}' 或稍后重试！"


async def test_weather_server():
    """测试天气服务器"""
    client = A2AClient("http://localhost:5005")

    # 获取代理信息
    try:
        logger.info("获取天气助手信息")
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
        logger.error(f"无法获取天气助手信息: {str(e)}")

    # 测试用例（匹配数据库数据和提示示例）
    queries = [
        "北京 2025-07-30",  # 具体日期查询
        "上海未来3天",  # 多天查询
        "北京明天",  # 相对日期查询
        "你好",  # 追问示例
        "今天有什么好吃的",  # 非相关查询，追问
        "不存在的城市 2025-07-31",  # 异常查询，可能无数据
    ]

    url = "http://localhost:5005"  # 使用服务器 URL 直接调用 API

    for query in queries:
        logger.info(f"测试天气查询: {query}")
        try:
            # 构建任务并发送（绕过 ask 方法，直接获取完整响应）
            task = {"message": {"content": {"text": query}}}
            r = requests.post(f"{url}/tasks/send", json=task)
            if r.status_code != 200:
                raise Exception(f"发送任务失败: {r.text}")

            task_resp = r.json()
            state = task_resp["status"]["state"].lower()  # 忽略大小写

            if state == "completed":
                # 提取 artifacts
                if "artifacts" in task_resp and task_resp["artifacts"]:
                    response = task_resp["artifacts"][0]
                else:
                    response = {"parts": [{"type": "text", "text": "无结果。如果需要其他日期，请补充。"}]}
            elif state == "input_required":
                # 提取 message 并包装为类似 JSON 格式
                message_text = task_resp["status"]["message"]["content"]["text"]
                response = {"status": "input_required", "message": message_text}
            else:
                response = {"parts": [{"type": "text", "text": f"未知状态: {task_resp['status']['state']}。请重试。"}]}

            logger.info(f"原始响应: {response}")
            summary = generate_summary(query, response)
            logger.info(summary)
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            summary = f"**天气总结**:\n查询 '{query}' 失败：{str(e)}。请检查输入或稍后重试！"
            logger.info(summary)


async def main():
    logger.info("开始测试天气服务器...")
    await test_weather_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")