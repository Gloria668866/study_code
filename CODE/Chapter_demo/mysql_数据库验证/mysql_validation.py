import mysql.connector
from mysql.connector import errorcode

# MySQL 配置信息
MYSQL_HOST = "82.156.249.211"
MYSQL_USER = "root"
MYSQL_PASSWORD = "YOUR_MYSQL_PASSWORD"
MYSQL_DB = "resume_rag_db"  # 目标数据库名称


def create_database(cursor):
    """
    检查并创建指定的数据库。
    """
    try:
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB} DEFAULT CHARACTER SET 'utf8mb4'")
        print(f"数据库 '{MYSQL_DB}' 创建成功或已存在。")
    except mysql.connector.Error as err:
        print(f"创建数据库失败: {err}")
        exit(1)


# --- 主程序逻辑 ---
try:
    # 尝试连接到 MySQL 服务器，注意这里没有指定 database 参数
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        port=3306
    )

    if conn.is_connected():
        print("成功连接到 MySQL 服务器!")

        # 创建一个游标对象
        cursor = conn.cursor()

        # 验证并创建数据库
        create_database(cursor)

        # 重新连接到新创建的（或已存在的）数据库
        conn.close()
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )

        if conn.is_connected():
            print(f"成功切换到数据库 '{MYSQL_DB}'")
            cursor = conn.cursor()

            # 验证数据库版本
            cursor.execute("SELECT VERSION()")
            db_version = cursor.fetchone()
            print(f"MySQL 数据库版本: {db_version[0]}")

except mysql.connector.Error as err:
    # 处理连接失败的情况
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("用户名或密码错误。")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("数据库不存在。")
    else:
        print(f"连接失败: {err}")

finally:
    # 确保连接被关闭
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL 连接已关闭。")