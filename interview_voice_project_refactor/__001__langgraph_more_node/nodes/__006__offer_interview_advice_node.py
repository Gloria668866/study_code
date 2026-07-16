from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from __001__langgraph_more_node.agent_state import AgentState
from __003__fastapi.update_mysql import update_mysql
from common.llm import my_llm
from __002__db_helper_parse.db_helper import my_db_helper


# 定义 Pydantic schema
class InterviewAdvice(BaseModel):
    overall_comment: str = Field(description="对整体面试表现的点评")
    overall_score: float = Field(description="面试整体评分，满分10分")
    strengths: List[str] = Field(description="面试者的优势点")
    weaknesses: List[str] = Field(description="面试者的不足之处")
    suggestions: List[str] = Field(description="改进建议")


parser = JsonOutputParser(pydantic_object=InterviewAdvice)


async def offer_interview_advice_node(state: AgentState):
    await update_mysql("开始提供面试建议", record_id=state["record_id"], processing_status=1)
    voice_arrange_text = state["voice_arrange_text"]

    parser = JsonOutputParser(pydantic_object=InterviewAdvice)
    format_instructions = parser.get_format_instructions()

    # ⚠️ system 里必须直接给出 parser 的 format 指令
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位专业的面试辅导专家。\n"
         "以下是一次面试的语音转文字逐字稿，因此可能包含口头禅、停顿词和语气词。\n"
         "请结合这一点进行分析，但重点放在内容本身，而不是转写的噪音。\n"
         "评分标准：整体评分为0到10分，10分代表表现极佳，0分代表非常糟糕。"
         "请严格输出符合以下 JSON schema 的内容，不要多余解释：\n\n"
         "{format_instructions}"),
        ("human",
         "以下是一次完整的面试逐字稿（语音转文字）：\n\n{interview_text}\n\n"
         "请你根据 schema 生成 JSON 格式的面试反馈。")
    ]).partial(format_instructions=format_instructions)

    # 只能对 LLM 部分 stream，parser 不支持流式解析
    llm_chain = prompt | my_llm

    print("\n=== 流式输出开始 ===\n")
    chunks = []
    for chunk in llm_chain.stream({"interview_text": voice_arrange_text}):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        print(content, end="", flush=True)
        chunks.append(content)
    print("\n\n=== 流式输出结束 ===\n")

    full_output = "".join(chunks).strip()

    advice_dict = parser.parse(full_output)
    print(advice_dict)

    state["interview_advice"] = advice_dict
    await update_mysql("完成提供面试建议", record_id=state["record_id"])
    overall_comments = advice_dict.get("overall_comment", "")
    overall_score = advice_dict.get("overall_score", 0.0)
    strengths = str(advice_dict.get("strengths", []))
    weaknesses = str(advice_dict.get("weaknesses", []))
    improvement_suggestions = str(advice_dict.get("suggestions", []))
    my_db_helper.update_interview_record(state["record_id"], {"overall_comments": overall_comments,
                                                              "interview_score": overall_score,
                                                              "strengths": strengths,
                                                              "weaknesses": weaknesses,
                                                              "improvement_suggestions": improvement_suggestions})
    return state


if __name__ == '__main__':
    import asyncio

    asyncio.run(
        offer_interview_advice_node({"record_id": 3,
                                     "voice_arrange_text": ""}))