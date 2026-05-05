from pydantic import BaseModel, Field
from typing import List

# 这就是我们的“设计蓝图”
class Actor(BaseModel):
    """定义我们期望的、完美的JSON结构"""
    name: str = Field(description="演员的姓名")
    height: int = Field(description="演员的身高（厘米），必须是一个纯数字")
    films: List[str] = Field(description="该演员出演过的电影名称列表")


#实例化
actor = Actor(name="张三", height=180, films=["电影1", "电影2"])
print(actor.name)
print(actor.height)