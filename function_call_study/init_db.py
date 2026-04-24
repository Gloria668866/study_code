import pymysql


def init_database():
    print("正在连接数据库...")
    # 请确保这里的账号密码和你的本地配置一致
    conn = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='password',
        database='it_heima',
        charset='utf8mb4'
    )
    cursor = conn.cursor()

    try:
        # 1. 清理可能存在的旧表 (避免重复运行报错)
        print("正在清理旧表...")
        cursor.execute("DROP TABLE IF EXISTS `emp`;")
        cursor.execute("DROP TABLE IF EXISTS `DEPT`;")

        # 2. 创建 DEPT 部门表
        print("正在创建 DEPT 表...")
        cursor.execute("""
        CREATE TABLE `DEPT` (
          `DEPTNO` int NOT NULL, 
          `DNAME` varchar(14) DEFAULT NULL,
          `LOC` varchar(13) DEFAULT NULL,
          PRIMARY KEY (`DEPTNO`)
        );
        """)

        # 3. 创建 emp 员工表
        print("正在创建 emp 表...")
        cursor.execute("""
        CREATE TABLE `emp` (
          `empno` int DEFAULT NULL, 
          `ename` varchar(50) DEFAULT NULL, 
          `job` varchar(50) DEFAULT NULL,
          `mgr` int DEFAULT NULL,
          `hiredate` date DEFAULT NULL,
          `sal` int DEFAULT NULL,
          `comm` int DEFAULT NULL,
          `deptno` int DEFAULT NULL
        );
        """)

        # 4. 插入测试数据
        print("正在插入测试数据...")
        # 插入部门数据
        cursor.execute(
            "INSERT INTO `DEPT` VALUES (10, '研发部', '北京'), (20, '销售部', '上海'), (30, '财务部', '广州');")

        # 插入员工数据
        cursor.execute("INSERT INTO `emp` VALUES (7369, '史密斯', '前台', 7902, '2020-12-17', 4000, NULL, 20);")
        cursor.execute("INSERT INTO `emp` VALUES (7499, '马云', '销售', 7698, '2021-02-20', 8000, 3000, 20);")
        cursor.execute("INSERT INTO `emp` VALUES (7839, '马化腾', '董事长', NULL, '2019-11-17', 50000, 100000, 10);")
        cursor.execute("INSERT INTO `emp` VALUES (7566, '雷军', '研发经理', 7839, '2020-04-02', 30000, NULL, 10);")

        # 提交事务
        conn.commit()
        print("✅ 数据库初始化完成！表已建好，数据已插入。")

    except Exception as e:
        # 发生错误时回滚
        conn.rollback()
        print(f"❌ 发生错误: {e}")

    finally:
        # 关闭连接
        cursor.close()
        conn.close()


if __name__ == '__main__':
    init_database()