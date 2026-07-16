import os
import json
from langchain_openai import ChatOpenAI
from typing import List, Dict

# --- 配置 ---
API_KEY ="YOUR_DEEPSEEK_API_KEY_HERE" # 替换为你的密钥
API_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# --- 知识库 (结构化数据) ---
KNOWLEDGE_BASE = [
    {"name": "AI 大模型开发工程师", "price": 3000, "duration": 12, "target": ["程序员", "软件工程师"]},
    {"name": "AI 大模型数据分析工程师", "price": 2500, "duration": 10, "target": ["数据分析师"]},
    {"name": "AI 大模型运维工程师", "price": 2000, "duration": 8, "target": ["运维工程师"]},
    {"name": "AI 大模型 Java 开发工程师", "price": 3500, "duration": 15, "target": ["Java程序员", "程序员"]},
]


# --- 定义可被调用的外部工具 (Python 函数) ---
def search_courses(profession: str, max_budget: int) -> List[Dict]:
    """根据profession职业和预算上限，搜索完全匹配的课程。"""

    print(f"\n[--- Python Tool Executing: search_courses(profession='{profession}', max_budget={max_budget}) ---]")
    results = [course for course in KNOWLEDGE_BASE if profession in course["target"] and course["price"] <= max_budget]
    return results


def find_closest_course(profession: str, budget: int) -> Dict:
    """当找不到完全匹配的课程时，为特定职业寻找价格最接近的课程。"""
    print(f"\n[--- Python Tool Executing: find_closest_course(profession='{profession}', budget={budget}) ---]")
    programmer_courses = [course for course in KNOWLEDGE_BASE if profession in course["target"]]
    if not programmer_courses:
        return {}
    return min(programmer_courses, key=lambda c: abs(c['price'] - budget))


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 初始化 LLM
    llm = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=API_URL,
        temperature=0.1,
        max_tokens=500
    )

    # 2. 定义用户问题和 ReAct 指令
    user_query = "我想学习Java程序员，预算2000贝，预算时间6个月"

    # 这个 Prompt 就是 ReAct 模式的“驱动程序”
    prompt_template = """
    你是一个专业的课程推荐助手。你必须严格遵循以下的决策流程，通过思考-行动循环来得出最终答案。

    # 明确的决策流程 ---
    **决策流程:**
    1.  **精确匹配**: 首先，必须使用 `search_courses` 工具尝试找到完全符合用户职业和预算的课程。
    2.  **近似匹配**: 如果上一步没有找到任何结果 (即 `search_courses` 返回了空列表 `[]`), 则接着使用 `find_closest_course` 工具，为用户寻找职业相同但价格最接近的课程。
    3.  **最终方案 (兜底策略)**: 如果 `find_closest_course` 工具依然没有找到任何课程 (例如，因为用户的职业不存在于我们的课程库中，工具返回了空结果), 你必须停止使用工具。此时，你的最终行动 (`action`) 必须是 `Finish`，并在最终答案中礼貌地告知用户暂时没有匹配的课程，同时引导他们访问我们的官方网站 `www.itcast.cn` 查看所有可用课程。

    **可用工具:**
    # --- 对工具的描述 ---
    - `search_courses(profession: str, max_budget: int)`: 根据职业和预算上限，搜索完全匹配的课程。如果找不到，会返回一个空列表 `[]`。
    - `find_closest_course(profession: str, budget: int)`: 当找不到完全匹配的课程时，为特定职业寻找价格最接近的课程。如果该职业没有任何课程，会返回一个空结果。

    **输出格式:**
    你的回应必须是严格的 JSON 格式，包含 'thought' 和 'action' 两个字段。
    - 'thought': 你的思考过程，必须体现你正在遵循上述的某一个决策步骤。
    - 'action': 你选择的行动，格式为 `{{"tool_name": "工具名", "tool_input": {{...}}}}` 或 `{{"tool_name": "Finish", "tool_input": "最终答案"}}`。

    **历史记录 (草稿纸):**
    {scratchpad}

    **当前任务:**
    {user_query}

    请生成你的下一步 JSON 回应:
    """

    # 3. 开始 ReAct 循环
    scratchpad = ""  # “草稿纸”，记录历史
    max_turns = 5

    for i in range(max_turns):
        print(f"\n{'=' * 20} ReAct 循环 第 {i + 1} 轮 {'=' * 20}")

        # 构造带有历史记录的 Prompt
        current_prompt = prompt_template.format(scratchpad=scratchpad, user_query=user_query)
        print(f"[--- Current_Prompt ---]\n{current_prompt}")

        # 调用 LLM 进行“思考”和“行动”决策
        response = llm.invoke(current_prompt)
        llm_output = response.content.strip()

        try:
            decision = json.loads(llm_output)
            thought = decision.get("thought")
            action = decision.get("action")

            # --- 这是 ReAct 的核心展示 ---
            print(f"\n**Reason (LLM 的思考):**\n{thought}")
            print(f"\n**Act (LLM 的行动决策):**\n{json.dumps(action, indent=2, ensure_ascii=False)}")

            # 更新草稿纸
            scratchpad += f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}\n"

            # --- 代码(调度员)部分开始工作 ---
            if action["tool_name"] == "Finish":
                print(f"\n{'=' * 20} 流程结束 {'=' * 20}")
                print(f"\n**最终答案:**\n{action['tool_input']}")
                break

            # 执行工具调用
            observation = None
            if action["tool_name"] == "search_courses":
                observation = search_courses(**action["tool_input"])
            elif action["tool_name"] == "find_closest_course":
                observation = find_closest_course(**action["tool_input"])
            else:
                observation = f"未知工具: {action['tool_name']}"

            print(f"\n**Observation (工具执行结果):**\n{observation}")

            # 将观察结果更新到草稿纸上，为下一轮做准备
            scratchpad += f"Observation: {observation}\n"

        except Exception as e:
            print(f"\n发生错误: {e}")
            print(f"LLM的原始输出: {llm_output}")
            break
    else:
        print("\n已达到最大循环次数，流程终止。")

        #observation