from __001__langgraph_more_node.agent_state import AgentState
from __002__db_helper_parse.db_helper import my_db_helper
from __003__fastapi.update_mysql import update_mysql


async def generate_markdown_node(state: AgentState):
    await update_mysql("开始生成markdown结果", record_id=state["record_id"], processing_status=1)
    interview_topic_list = state['interview_topic_list']
    interview_advice = state["interview_advice"]
    interview_date_str = state["interview_info_dict"]["interview_date_str"]
    name = state["interview_info_dict"]["name"]

    # 生成 Markdown 文档
    md_parts = []

    # 标题
    md_parts.append(f"# {name}同学面试复盘报告\n")
    md_parts.append(f"### 面试时间:{interview_date_str}")

    # 第一部分：整体建议
    md_parts.append("## 一、面试表现分析与建议\n")

    md_parts.append(f"### 1、整体点评：\n")
    md_parts.append(f"\n{interview_advice['overall_comment']}\n")

    md_parts.append(f"### 2、整体评分：\n")
    md_parts.append(f"\n{interview_advice['overall_score']}\n")

    md_parts.append("### 3、优势:\n")
    for strength in interview_advice["strengths"]:
        md_parts.append(f"- {strength}")

    md_parts.append("### 4、不足: \n")
    for weakness in interview_advice["weaknesses"]:
        md_parts.append(f"- {weakness}")

    md_parts.append("### 5、改进建议：\n")
    for suggestion in interview_advice["suggestions"]:
        md_parts.append(f"- {suggestion}")

    # 第二部分：逐题问答
    md_parts.append("## 二、面试问答记录\n")
    for idx, qa in enumerate(interview_topic_list, 1):
        md_parts.append(f"### 问题 {idx}: {qa['question']}\n")
        md_parts.append(f"- **面试者回答**：\n\n{qa['user_answer']}\n")

        # 参考答案
        if 'sample_answer' in qa and qa['sample_answer']:
            md_parts.append(f"- **参考答案**：\n\n{qa['sample_answer']}\n")

        # 面试题分析
        if 'analysis' in qa and qa['analysis']:
            analysis = qa['analysis']
            md_parts.append(f"- **面试题分析**：\n")
            md_parts.append(f"  - **考点分析**：{analysis.get('exam_point', '')}\n")
            md_parts.append(f"  - **答题思路**：{analysis.get('answer_approach', '')}\n")
            md_parts.append(f"  - **回答评价**：{analysis.get('answer_evaluation', '')}\n")
            md_parts.append(f"  - **回答评分**：{analysis.get('score', '')}\n")

    # 拼接 markdown
    markdown_doc = "\n".join(md_parts)

    # 存入 state
    state["interview_markdown_text"] = markdown_doc
    print(state["interview_markdown_text"])
    my_db_helper.update_interview_record(state["record_id"], {"markdown_text": markdown_doc})
    await update_mysql("完成生成markdown结果", record_id=state["record_id"], processing_status=2)
    return state


if __name__ == '__main__':
    import asyncio

    asyncio.run(generate_markdown_node({"interview_topic_list": [{'question': '请先自我介绍一下，并打开摄像头。',
                                                                  'user_answer': '面试官您好，我叫小明，毕业于广东外贸大学南国商学院物联网工程专业。在校期间通过自学AR知识，包括NRP领域的深度学习、机器学习以及大模型的前沿技术方向。毕业后从事基础数据处理工作，近期参与电商RAG智能客服项目和运动医学领域知识图谱构建项目。',
                                                                  'analysis': {
                                                                      'exam_point': '这道题主要考察面试者的沟通表达能力、自我认知能力、职业发展路径清晰度以及技术背景与岗位匹配度。通过自我介绍了解候选人的教育背景、技术专长、项目经验和职业规划。',
                                                                      'answer_approach': '理想的答题思路应该包括：1) 基本信息简要介绍；2) 教育背景与专业特长；3) 技术能力和知识体系；4) 相关项目经验及个人贡献；5) 职业发展方向与目标。结构要清晰，重点突出与应聘岗位相关的技能和经验。',
                                                                      'answer_evaluation': '回答质量较好，优点：教育背景和专业明确，技术方向（AR、深度学习、机器学习、大模型）符合当前技术趋势，项目经验具体且有相关性。不足：缺乏个人在项目中的具体角色和贡献描述，技术深度展示不够，职业规划和发展目标不清晰，表达略显笼统。'},
                                                                  'sample_answer': '面试官您好，我是小明，广东外贸大学南国商学院物联网工程专业毕业。专注于AR技术领域，系统学习过深度学习、机器学习和大模型等前沿技术。在数据处理方面有扎实基础，近期主导了电商RAG智能客服系统的算法优化，提升了客服响应准确率15%；参与运动医学知识图谱项目，负责实体关系抽取模块开发。希望将我的技术能力应用于贵公司的AI产品研发，为业务创造实际价值。'},
                                                                 {'question': '毕业后是否从事了一年的Agent开发工作？',
                                                                  'user_answer': '是的，但负责的工作较为基础，主要包括数据清洗、处理以及模型框架的基本构建。部署和上线环节不属于我的工作范畴。',
                                                                  'analysis': {
                                                                      'exam_point': '这道题主要考察面试者的工作经历真实性、技术能力边界以及职业发展阶段的定位。重点评估其是否具备Agent开发的实际经验，以及在团队中的角色定位和技术贡献度。',
                                                                      'answer_approach': '理想的答题思路应包含：明确确认工作经历，具体说明工作内容和技术栈，客观描述职责范围，并体现对完整开发流程的理解。结构上应先肯定回答，再详细说明工作内容，最后补充职责边界。',
                                                                      'answer_evaluation': '面试者的回答质量较好，优点在于：明确确认了工作经历，具体说明了数据清洗、模型构建等基础工作内容，诚实地界定了职责边界。不足之处是缺乏具体技术栈描述和工作成果量化，未能体现对Agent开发完整流程的理解深度。'},
                                                                  'sample_answer': '是的，毕业后我在XX公司从事了为期一年的Agent开发工作。主要负责数据处理、特征工程和基础模型构建，使用Python、PyTorch等技术栈完成了多个项目的预处理模块。虽然未参与部署上线环节，但对Agent开发的完整流程有清晰认知，并在基础工作中积累了扎实的工程实践经验。'},
                                                                 {'question': '一年内参与了几个项目？主要参与的是哪个？',
                                                                  'user_answer': '共参与三个项目，其中第一个电商RAG项目是我近期掌握较好的主要项目。',
                                                                  'analysis': {
                                                                      'exam_point': '考察项目经验广度与深度、重点项目的参与程度和贡献度，以及项目描述的清晰度和专业性',
                                                                      'answer_approach': '应清晰列出项目数量，明确主要项目并简要说明其业务背景、个人角色和关键技术点，体现项目经验和专业能力',
                                                                      'answer_evaluation': '回答简洁但过于笼统，仅提及项目数量和主要项目名称，缺乏具体业务背景、个人职责和技术细节的描述，未能充分展示项目经验和专业能力'},
                                                                  'sample_answer': '一年内参与三个项目，主要参与电商RAG项目，负责构建基于检索增强生成的智能客服系统，使用LangChain框架实现知识库检索和LLM生成回答，提升了客服响应准确率30%。'}],
                                        "interview_advice": {
                                            'overall_comment': '面试者罗培京在面试中能够清晰地介绍自己的教育背景和工作经历，对参与的项目有基本的了解。但在回答技术细节和项目深度问题时表现不够自信，存在逻辑不够清晰、表达不够精准的问题。整体表现中等，技术深度和沟通能力有待提升。',
                                            'strengths': ['对参与的项目有基本了解，能够描述项目背景和主要功能',
                                                          '对RAG、知识图谱等技术概念有一定认知',
                                                          '能够主动询问公司业务需求和团队情况，表现出对岗位的兴趣'],
                                            'weaknesses': ['技术细节回答不够深入，对项目架构和实现原理理解较浅',
                                                           '表达逻辑不够清晰，存在较多口头禅和重复表述',
                                                           '对数据抓取合法性等敏感问题回答不够专业',
                                                           '对用户量、商业模式等业务问题回答含糊不清',
                                                           '缺乏对技术方案优劣的深入分析和对比思考'],
                                            'suggestions': [
                                                '加强技术深度学习，特别是对参与项目的架构设计和实现细节要有更深入理解',
                                                '提升表达能力和逻辑思维，减少口头禅，提高回答的精准度和条理性',
                                                '加强对业务模式、数据合规等非技术问题的思考和准备',
                                                '练习用更结构化的方式描述技术流程和方案对比',
                                                '增强对技术方案优劣的分析能力，能够清晰阐述不同方案的适用场景']},
                                        "interview_date_str": "2024年1月15日",
                                        "name": "小明"}))
