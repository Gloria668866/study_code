"""应用层读写库连接（用户/会话/知识库）。

与 app/db.py 区分：
- app/db.py     = 只读分析库（DATABASE_URL），供 Text2SQL 查 dim_*/fact_*，不写入。
- app/database.py = 读写应用库（APP_DATABASE_URL），存用户/会话/消息/知识库元数据。
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .config import APP_DATABASE_URL
from .models import Base

# SQLite 在多线程（uvicorn）下需要关掉 same-thread 检查；其它库无需此参数
_connect_args = {"check_same_thread": False} if APP_DATABASE_URL.startswith("sqlite") else {}
app_engine = create_engine(APP_DATABASE_URL, future=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=app_engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """建表（幂等）。生产用 schema.sql / 迁移工具时可不调用。"""
    Base.metadata.create_all(app_engine)


def get_db():
    """FastAPI 依赖：每请求一个会话，结束自动关闭。"""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
