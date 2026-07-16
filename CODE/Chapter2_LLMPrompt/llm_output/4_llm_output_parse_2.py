# 首先，请确保已安装必要的库:
# pip install langchain langchain-openai pydantic
import json
import os
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from typing import List

# --- 步骤 1: 配置你的大语言模型 ---
# 这是执行“修复”任务的“工人”。
print("--- 步骤 1: 配置LLM ---")
llm = ChatOpenAI(
    # 这里使用DeepSeek模型作为示例
    model="deepseek-chat",
    # 请替换为您的API密钥，或设置为环境变量
    api_key = "YOUR_DEEPSEEK_API_KEY_HERE",
    base_url="https://api.deepseek.com/v1",
    # 使用低温，让“修复”任务的结果更稳定、更可预测
    temperature=0.0
)
print("LLM配置完成。")

# --- 步骤 2: 使用Pydantic定义“完美的报告蓝图” ---
# 我们告诉LangChain，我们最终想要的数据长什么样。
print("\n--- 步骤 2: 定义Pydantic数据模型 (我们的目标) ---")


class Actor(BaseModel):
    """这个类就是我们期望的、完美的JSON结构定义"""
    # Field中的description会帮助LangChain生成更精确的指令
    name: str = Field(description="演员的姓名")
    height: int = Field(description="演员的身高（厘米），必须是一个纯数字")
    films: List[str] = Field(description="该演员出演过的电影名称列表")


print("数据模型 `Actor` 已定义。")

# --- 步骤 3: 准备一份“有问题的报告” ---
# 这是我们的LLM助理交上来的、格式不完美的原始文本。
print("\n--- 步骤 3: 准备一份格式错误的LLM输出 ---")
# bad_llm_output = """
# 当然，这是您要的演员信息：
# {
#   "name": "基努·里维斯",
#   "height": "186cm",
#   "films": ["黑客帝国", "疾速追杀"]
# }
# 祝您有美好的一天！
# """

# print(json.loads(bad_llm_output))
bad_llm_output="我爱中国"

# print(f"收到的“坏”数据:\n{bad_llm_output}")

# --- 步骤 4: 创建并使用 OutputFixingParser ---
# 这就是我们的“智能项目经理”。
print("\n--- 步骤 4: 创建并使用OutputFixingParser ---")

# .from_llm() 是最关键的构造函数，它需要两样东西：
# 1. llm: 用哪个LLM去执行修复任务。
# 2. parser: 我们的“设计蓝图”是什么。这里传入一个基于Actor模型创建的标准解析器。
output_fixing_parser = OutputFixingParser.from_llm(
    parser=PydanticOutputParser(pydantic_object=Actor),
    llm=llm
)

try:
    # 我们直接调用修复解析器的 .parse() 方法
    # 它在后台会自动完成“尝试解析 -> 捕获错误 -> 调用LLM修复 -> 再次解析”的全部流程
    print("\n正在调用修复解析器... (如果原始数据有问题，这里会发起一次API调用来修复)")
    fixed_result = output_fixing_parser.parse(bad_llm_output)

    # 如果代码能走到这里，说明修复和解析都已成功！
    print("\n √ [结论]: OutputFixingParser 成功修复并解析！")
    print("   修复后的结果是一个干净的Pydantic对象:")
    print(f"   - 姓名 (name): {fixed_result.name}")
    print(f"   - 身高 (height): {fixed_result.height} (数据类型: {type(fixed_result.height)})")
    print(f"   - 电影 (films): {fixed_result.films}")

except Exception as e:
    # 正常情况下很难失败，但我们也做好准备
    print(f"\n[结论]: 连OutputFixingParser都失败了！")
    print(f"   错误原因: {e}")

print("\n--- 演示结束 ---")