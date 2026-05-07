from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from __001__langgraph_more_node.agent_state import AgentState
from __003__fastapi.update_mysql import update_mysql
from common.llm import my_llm
from __002__db_helper_parse.db_helper import my_db_helper


# 定义 Pydantic 模型，用于标准化 JSON 输出
class Analysis(BaseModel):
    exam_point: str = Field(description="考点分析：这道题主要考察什么能力、知识点或技能")
    answer_approach: str = Field(description="答题思路：理想的答题思路和结构应该是什么")
    answer_evaluation: str = Field(description="回答评价：面试者的回答质量如何，有哪些优点和不足")
    score: float = Field(description="本题得分，满分10分，可保留1位小数")


class InterviewAnalysis(BaseModel):
    analysis: Analysis = Field(description="面试题分析")
    sample_answer: str = Field(description="参考答案：更专业、更简洁的参考答案")


# 使用 JsonOutputParser 确保输出为 JSON
parser = JsonOutputParser(pydantic_object=InterviewAnalysis)

# 定义提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业的面试官和职业指导师，需要分析面试题和面试者回答，并给出参考答案和评分。

要求：
- 分析要客观、专业、具体。
- 评分标准：满分10分，5分为合格，10分为极佳，0分为完全不符合要求。
- 评分时综合考虑表达逻辑、专业性、完整性和沟通能力。
- 输出必须严格符合以下JSON格式，不要添加任何其他文字。

{format_instructions}"""),
    ("human", "面试题：{question}\n面试者回答：{user_answer}\n请严格按照JSON格式进行分析、评分并提供参考答案：")
]).partial(format_instructions=parser.get_format_instructions())


async def process_single_qa(qa: dict, chain) -> dict:
    """
    处理单个问答对，生成分析和参考答案（JSON格式）
    
    Args:
        qa: 包含question和user_answer的字典
        chain: 统一的LLM处理链
        
    Returns:
        包含analysis和sample_answer的完整问答字典
    """
    # await put_think_msg_and_update_mysql(f"\n问题: {qa['question']}", record_id=state["record_id"])
    # await put_think_msg_and_update_mysql(f"用户回答: {qa['user_answer']}", record_id=state["record_id"])

    print(f"\n问题: {qa['question']}")
    print(f"用户回答: {qa['user_answer']}")
    print("分析和参考答案: ", end="", flush=True)

    # 先流式输出原始文本
    collected_text = ""
    for chunk in chain.stream({
        "question": qa["question"],
        "user_answer": qa["user_answer"]
    }):
        token = chunk.content
        print(token, end="", flush=True)  # 实时打印
        collected_text += token
    print("\n--- 流式结束 ---")

    # 再用 parser 解析完整文本
    result = parser.parse(collected_text)

    qa_with_llm = {
        **qa,
        "analysis": result['analysis'],
        "sample_answer": result['sample_answer']
    }

    return qa_with_llm


async def offer_sample_answer_node(state: AgentState):
    """
    使用大模型生成面试题分析和参考答案，流式打印，再解析为 dict
    """
    await update_mysql("开始提供参考答案和分析", record_id=state["record_id"], processing_status=1)
    interview_topic_list = state["interview_topic_list"]

    # 创建 chain
    chain = prompt | my_llm

    new_interview_topic_list = []
    for i, qa in enumerate(interview_topic_list):
        await update_mysql(f"正在处理第{i + 1}/{len(interview_topic_list)}个问题的回答", record_id=state["record_id"])
        qa_with_llm = await process_single_qa(qa, chain)
        new_interview_topic_list.append(qa_with_llm)

    state["interview_topic_list"] = new_interview_topic_list
    print(state["interview_topic_list"])
    await update_mysql("完成提供参考答案和分析", record_id=state["record_id"])
    # def add_interview_analysis_detail(self, interview_record_analysis_id, interview_question=None,
    #                                   interviewee_answer=None,
    #                                   reference_answer=None, point_analysis=None, answer_thoughts=None,
    #                                   answer_evaluation=None, answer_score=None):
    for qa in state["interview_topic_list"]:
        my_db_helper.add_interview_analysis_detail(interview_record_analysis_id=state['record_id'],
                                                   interview_question=qa.get('question', ''),
                                                   interviewee_answer=qa.get('user_answer',''),
                                                   reference_answer=qa.get('sample_answer', ''),
                                                   point_analysis=qa.get('analysis', {}).get('exam_point', ''),
                                                   answer_thoughts=qa.get('analysis', {}).get('answer_approach', ''),
                                                   answer_evaluation=qa.get('analysis', {}).get("answer_evaluation", ''),
                                                   answer_score=qa.get('analysis', {}).get("score", 0.0))
    return state


if __name__ == '__main__':
    import asyncio

    asyncio.run(offer_sample_answer_node({
        "interview_topic_list": [
            {
                "question": "请先自我介绍一下，并打开摄像头。",
                "user_answer": "面试官您好，我叫小明，毕业于广东外贸大学南国商学院物联网工程专业。在校期间通过自学AR知识，包括NRP领域的深度学习、机器学习以及大模型的前沿技术方向。毕业后从事基础数据处理工作，近期参与电商RAG智能客服项目和运动医学领域知识图谱构建项目。"
            },
            {
                "question": "毕业后是否从事了一年的Agent开发工作？",
                "user_answer": "是的，但负责的工作较为基础，主要包括数据清洗、处理以及模型框架的基本构建。部署和上线环节不属于我的工作范畴。"
            },
            {
                "question": "一年内参与了几个项目？主要参与的是哪个？",
                "user_answer": "共参与三个项目，其中第一个电商RAG项目是我近期掌握较好的主要项目。"
            },
            {
                "question": "该项目团队规模如何？你主要负责哪些部分？",
                "user_answer": "团队共有6人。我主要负责数据清洗、SQL模块和RAG集成，并协助意图识别模块的工作。"
            },
            {
                "question": "电商RAG智能客服具体解决什么问题？",
                "user_answer": "该系统针对电商平台客服在商品价格、促销活动和售后政策时效性方面的不足，通过构建外挂知识库（集成近两个月更新的商品信息和政策），结合上下文识别技术输入大模型，提升回答的准确性、时效性和召回率。"
            },
            {
                "question": "该系统面向ToC还是ToB用户？",
                "user_answer": "同时面向ToC和ToB用户：消费者可查询商品信息，客服团队可获取商品访问量、销量及用户反馈等数据。"
            },
            {
                "question": "如何实现用户间的安全隔离？",
                "user_answer": "通过环境隔离和分库机制，分别存储客户端和客服团队的数据。使用Docker容器化技术保障轻量级部署和资源效率。"
            },
            {
                "question": "数据抓取是否合法？公司商业模式是什么？",
                "user_answer": "抓取前会参考网站的Robots协议，确保合规。公司主要提供数据服务，并为中小商户提供定向外包开发。"
            },
            {
                "question": "运动医学知识图谱项目是否涉及语音对话？",
                "user_answer": "不涉及语音对话，采用机器人文本问答形式。原始数据为Word或PDF格式。"
            },
            {
                "question": "如何对运动医学数据进行标注和清洗？请举例说明问答流程。",
                "user_answer": "以用户输入'体重较大如何运动减脂'为例：先进行意图识别，若属运动医学范畴则抽取实体（如'减脂'），匹配知识图谱中存储的向量化数据（如跑步动作、热身事项），生成Cypher查询语句检索结果后送入大模型生成答案。"
            },
            {
                "question": "该流程与直接使用DeepSeek相比有何优势？",
                "user_answer": "知识图谱可动态更新最新规则和损伤类型等信息，避免大模型因训练数据截止时间导致的幻觉问题，提升准确性。"
            },
            {
                "question": "运动问答系统当前用户量和应用场景如何？",
                "user_answer": "用户量约3万，峰值达10万。系统集成在健身APP、运动康复平台等ToB场景，目前服务约三至四家企业。"
            },
            {
                "question": "用户反馈的主要问题是什么？数据来源有哪些？",
                "user_answer": "用户反馈集中在营养素与补剂成分关系识别精度不足。数据主要来自国内外运动医学网站。"
            },
            {
                "question": "您对本公司岗位的业务需求有何疑问？",
                "user_answer": "请问该岗位具体业务方向是什么？机器人的交互方式是否支持语音识别？团队规模如何？"
            }

        ]}))
