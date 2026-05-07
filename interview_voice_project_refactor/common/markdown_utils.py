import ast


def literal_eval(str_data):
    try:
        return ast.literal_eval(str_data)
    except:
        return []


def generate_markdown(records, detail_list):
    if len(records) <= 0:
        return ""
    name = records[0].get("name", "")
    interview_date_str = str(records[0].get("interview_time", ""))
    overall_comment = records[0].get("overall_comments", "")
    overall_score = records[0].get("interview_score", 0.0)
    strengths = literal_eval(records[0].get("strengths", ""))
    weaknesses = literal_eval(records[0].get("weaknesses", ""))
    improvements_suggestions = literal_eval(records[0].get("improvement_suggestions", ""))

    # 生成 Markdown 文档
    md_parts = []

    # 标题
    md_parts.append(f"# {name}同学面试复盘报告\n")
    md_parts.append(f"### 面试时间:{interview_date_str}")

    # 第一部分：整体建议
    md_parts.append("## 一、面试表现分析与建议\n")

    md_parts.append(f"### 1、整体点评：\n")
    md_parts.append(f"\n{overall_comment}\n")

    md_parts.append(f"### 2、整体评分：\n")
    md_parts.append(f"\n{overall_score}\n")

    md_parts.append("### 3、优势:\n")
    for strength in strengths:
        md_parts.append(f"- {strength}")

    md_parts.append("### 4、不足: \n")
    for weakness in weaknesses:
        md_parts.append(f"- {weakness}")

    md_parts.append("### 5、改进建议：\n")
    for suggestion in improvements_suggestions:
        md_parts.append(f"- {suggestion}")
    print(md_parts)

    # 第二部分：逐题问答
    md_parts.append("## 二、面试问答记录\n")
    for idx, qa in enumerate(detail_list, 1):
        interview_question = qa.get("interview_question", "")
        interviewee_answer = qa.get("interviewee_answer", "")
        reference_answer = qa.get("reference_answer", "")
        point_analysis = qa.get("point_analysis", "")
        answer_thoughts = qa.get("answer_thoughts", "")
        answer_evaluation = qa.get("answer_evaluation", "")
        answer_score = qa.get("answer_score", 0.0)

        md_parts.append(f"### 问题 {idx}: {interview_question}\n")
        md_parts.append(f"- **面试者回答**：\n\n{interviewee_answer}\n")
        md_parts.append(f"- **参考答案**：\n\n{reference_answer}\n")

        # 面试题分析
        md_parts.append(f"- **面试题分析**：\n")
        md_parts.append(f"  - **考点分析**：{point_analysis}\n")
        md_parts.append(f"  - **答题思路**：{answer_thoughts}\n")
        md_parts.append(f"  - **回答评价**：{answer_evaluation}\n")
        md_parts.append(f"  - **回答评分**：{answer_score}\n")

    # 拼接 markdown
    markdown_doc = "\n".join(md_parts)
    return markdown_doc
