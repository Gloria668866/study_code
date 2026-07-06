from zhipuai import ZhipuAI
from dotenv import load_dotenv, find_dotenv
import os
import json
from sql_function_tool import tools, ask_database, parse_response

_ = load_dotenv(find_dotenv())

# 初始化客户端
zhupu_ak = os.environ.get('zhupu_api', '9413ddf3ac92414e84e0a855a299a359.hRCwGtZZI6jjMqKv')
client = ZhipuAI(api_key=zhupu_ak)
MODEL_NAME = "glm-4"


def main():
    messages = [
        {"role": "system",
         "content": "你是一个智能数据分析师。通过针对业务数据库生成 SQL 查询来回答用户的问题。注意：如果需要创建临时表，请使用 'yxluodanping' 作为标识符。"},
        {"role": "user", "content": "查询一下最高工资的员工姓名及对应的工资"}
    ]

    # 1. 第一次请求：模型判断是否需要调用工具
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message
    #messages.append(assistant_message)  # 直接添加对象，SDK会自动处理
    messages.append(assistant_message.model_dump(exclude_none=True))  # 将对象转为字典，并过滤掉为空的字段
    # 2. 检查是否有工具调用
    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            function_id = tool_call.id
            function_name = tool_call.function.name

            # 执行本地函数
            print(f"\n[模型动作]: 调用函数 {function_name}")
            function_result = parse_response(assistant_message)

            # 3. 将工具执行结果存入对话历史
            messages.append({
                "role": "tool",
                "tool_call_id": function_id,
                "content": json.dumps(function_result, ensure_ascii=False)
            })

        # 4. 第二次请求：模型根据数据库返回的数据生成自然语言回答
        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )

        print("\n[最终回答]:")
        print(final_response.choices[0].message.content)
    else:
        print("\n[直接回答]:")
        print(assistant_message.content)


if __name__ == '__main__':
    main()