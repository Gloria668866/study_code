# mcp_weather_server.py
# 功能：提供天气数据库查询接口，仅支持SELECT查询，返回JSON结果。鲁棒性：异常捕获，JSON格式化。

import mysql.connector
import logging
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from python_a2a.mcp import FastMCP
import uvicorn
from python_a2a.mcp import create_fastapi_app

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 自定义JSON编码器
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(obj, datetime) else obj.strftime('%Y-%m-%d')
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


# 天气服务类
class WeatherService:
    def __init__(self):
        # 连接数据库
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",  # 替换为您的数据库用户名
            password="YOUR_MYSQL_PASSWORD",  # 替换为您的数据库密码
            database="travel_rag"
        )

    def execute_query(self, sql: str) -> str:
        try:
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            # 格式化结果中的日期和数值
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (date, datetime, timedelta, Decimal)):
                        result[key] = self.default_encoder(value)
            return json.dumps({"status": "success", "data": results} if results else {"status": "no_data", "message": "未找到天气数据，请确认城市和日期。"}, cls=DateEncoder, ensure_ascii=False)
        except Exception as e:
            logger.error(f"天气查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    def default_encoder(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, Decimal):
            return float(obj)
        return obj


if __name__ == '__main__':

    service = WeatherService()
    print(service.conn.is_connected())
    service.conn.close()