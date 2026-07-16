"""Quick import QA data to MySQL."""
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from mysql_qa.db.mysql_client import MySQLClient

mysql = MySQLClient()
mysql.create_table()
csv_path = os.path.join(project_root, "rag_qa", "data", "all_subjects_qa.csv")
print(f"Importing from: {csv_path}")
mysql.insert_data(csv_path)
print("MySQL import done!")
mysql.close()
