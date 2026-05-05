# mcp_ticket_server.py
# 功能：提供票务数据库查询接口，仅支持SELECT查询，返回JSON结果。鲁棒性：异常捕获，JSON格式化，去掉预订功能。

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

if __name__ == '__main__':
    encoder = DateEncoder()
    print(encoder.default(datetime(2025, 8, 11, 8, 0)))
    print(encoder.default(date(2025, 8, 11)))
    print(encoder.default(timedelta(days=1)))
    print(encoder.default(Decimal('123.45')))