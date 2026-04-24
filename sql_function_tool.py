import json
import pymysql

# 定义数据库模式
database_schema_string1 = """
CREATE TABLE `emp` (
  `empno` int DEFAULT NULL, --员工编号
  `ename` varchar(50) DEFAULT NULL, --员工姓名
  `job` varchar(50) DEFAULT NULL,--员工工作
  `mgr` int DEFAULT NULL,--员工领导
  `hiredate` date DEFAULT NULL,--员工入职日期
  `sal` int DEFAULT NULL,--员工的月薪
  `comm` int DEFAULT NULL,--员工年终奖
  `deptno` int DEFAULT NULL --员工部门编号
);

CREATE TABLE `DEPT` (
  `DEPTNO` int NOT NULL, -- 部门编码
  `DNAME` varchar(14) DEFAULT NULL,--部门名称
  `LOC` varchar(13) DEFAULT NULL,--地点
  PRIMARY KEY (`DEPTNO`)
);
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "使用此函数回答业务问题，输入是一个标准的 SQL 查询语句",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"用于提取信息回答问题的SQL语句。数据库架构：{database_schema_string1}"
                    }
                },
                "required": ["query"],
            },
        }
    }
]


def ask_database(query):
    """执行SQL查询并返回结果"""
    try:
        conn = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='password',  # 请确保密码正确
            database='it_heima',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor  # 推荐返回字典格式，大模型更容易理解
        )
        with conn.cursor() as cursor:
            print(f"[执行SQL]: {query}")
            cursor.execute(query)
            result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        return f"数据库查询出错: {str(e)}"


def parse_response(response_message):
    """解析模型回复并执行对应的函数"""
    if not response_message.tool_calls:
        return None

    tool_call = response_message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    if function_name == "ask_database":
        # 执行数据库查询
        return ask_database(query=function_args.get("query"))
    return None