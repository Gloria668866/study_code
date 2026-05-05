
#定义配置文件
class Config(object):
    def __init__(self):
        #大模型信息
        self.api_key="YOUR_DEEPSEEK_API_KEY_HERE"
        self.api_url="https://gateway.ai.cloudflare.com/v1/d2cbfe461e343906da9615cbceab35c6/test1/deepseek"
        self.model_name="deepseek-chat"

        #数据库信息
        self.db_host="localhost"
        self.db_port=3306
        self.db_user="root"
        self.db_password="YOUR_MYSQL_PASSWORD"
        self.db_database="douban_movies"
        self.db_insurance="insurance_db"


