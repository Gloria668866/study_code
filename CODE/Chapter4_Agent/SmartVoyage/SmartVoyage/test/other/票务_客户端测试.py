#!/usr/bin/env_log.log python
"""
测试 ticket_server.py 和 weather_server.py 的客户端代码
参考 douban_A2Aagent_client_noStreaming.py 的实现思路：
- 使用彩色日志记录过程
- 获取助手信息
- 支持测试查询，并模拟生成总结（使用静态提示，假设无LLM依赖，或可选集成LLM）
- 处理正常查询、模糊查询和异常情况
- 使用异步方式确保高效交互
"""

import asyncio
import colorlog
import logging
from python_a2a import A2AClient
from datetime import datetime, timedelta
import pytz

# 设置彩色日志，参考豆瓣客户端
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={'INFO': 'green', 'ERROR': 'red'}
))
logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(colorlog.INFO)


def generate_summary(query, response, server_type):
    """生成总结"""
    if "no_data" in response or "无结果" in response:
        return f"**{server_type}总结**:\n抱歉，查询 '{query}' 未找到结果。如果需要其他日期或类型，请补充更多细节！"
    else:
        return f"**{server_type}总结**:\n欢迎来到{server_type}查询助手！根据您的查询 '{query}'，这里是关键信息：\n{response}\n如果需要更多查询，请继续！"


async def test_server(client, server_name, queries):
    """通用测试函数"""
    # 获取代理信息
    try:
        logger.info(f"获取{server_name}助手信息")
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
        logger.error(f"无法获取{server_name}助手信息: {str(e)}")

    # 测试查询
    for query in queries:
        logger.info(f"测试{server_name}查询: {query}")
        try:
            # 发送查询，参考豆瓣的ask方法
            response = client.ask(query)
            logger.info(f"原始响应: {response}")

            # 生成总结，参考豆瓣的解说员总结
            summary = generate_summary(query, response, server_name)
            logger.info(summary)

        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            summary = generate_summary(query, f"查询失败: {str(e)}", server_name)
            logger.info(summary)


async def main():
    # 初始化客户端
    ticket_client = A2AClient("http://localhost:5006")
    weather_client = A2AClient("http://localhost:5005")

    # 票务服务器测试用例
    tomorrow = (datetime.now(pytz.timezone('Asia/Shanghai')) + timedelta(days=1)).strftime('%Y-%m-%d')
    ticket_queries = [
        f"火车票 北京 上海 2025-08-02 硬卧",  # 正常查询：火车票
        f"机票 上海 广州 2025年09月11日 头等舱",  # 正常查询：机票
        "演唱会 北京 刀郎 2025-08-23 看台",  # 正常查询：演唱会
    ]

    # 天气服务器测试用例
    weather_queries = [
        # f"北京 {tomorrow}",  # 正常查询：单日天气
        # "上海未来3天",  # 正常查询：多日天气
        # "今天",  # 模糊查询
        # "不存在的城市 2025-07-31",  # 异常查询：无结果
    ]

    # 运行测试
    await test_server(ticket_client, "票务", ticket_queries)
    await test_server(weather_client, "天气", weather_queries)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")