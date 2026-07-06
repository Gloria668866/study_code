import json
import re

def parse_llm_json_output(response: str) -> dict:
    """
    提供一个工业级健壮性的方法，用于解析大模型返回的、格式不完美的JSON字符串。
    它采用分层回退的策略，并加入了智能起点探测，以最大化解析成功率。
    整体思路：通过三层策略（JSON边界探测、Markdown代码块提取、最终清理）逐步尝试解析JSON，
    每层策略针对不同类型的输入异常，逐步放宽条件以确保尽可能解析成功。
    """
    # 步骤1：智能贪婪边界策略 - 优先尝试找到“真实”的JSON起点和终点
    # 思路：通过查找合法的JSON对象或数组的起点（{ 或 [），并结合上下文验证（如 { 后是否跟 "），
    #       确定JSON的有效范围，然后提取并修复常见语法错误（如悬挂逗号），尝试解析。
    try:
        # 步骤1.1：寻找JSON对象的真实起点
        # 思路：逐个查找 {，检查其后是否为合法JSON对象的特征（即 "），以排除无关的 {。
        json_object_start = -1
        last_pos = 0
        while True:
            start_pos = response.find('{', last_pos)
            if start_pos == -1:
                break
            substr = response[start_pos + 1:]
            next_char_index = next((i for i, char in enumerate(substr) if not char.isspace()), -1)
            if next_char_index != -1 and substr[next_char_index] == '"':
                json_object_start = start_pos
                break
            last_pos = start_pos + 1

        # 步骤1.2：寻找JSON数组的起点
        # 思路：直接查找 [，因为数组起点较简单，无需复杂验证。
        json_array_start = response.find('[')

        # 步骤1.3：决定最终的JSON起点
        # 思路：比较对象和数组的起点，选择最靠前的有效位置。
        start_positions = [p for p in [json_object_start, json_array_start] if p != -1]
        if not start_positions:
            raise ValueError("找不到有效的JSON起始符号")
        final_start_pos = min(start_positions)

        # 步骤1.4：查找JSON的结束符号
        # 思路：查找最后一个 } 或 ]，以确保捕获完整的JSON结构。
        end_positions = [p for p in [response.rfind('}'), response.rfind(']')] if p != -1]
        if not end_positions:
            raise ValueError("找不到有效的JSON结束符号")
        final_end_pos = max(end_positions)

        # 步骤1.5：提取可能的JSON字符串
        # 思路：根据确定的起点和终点，截取可能包含JSON的子字符串。
        potential_json_str = response[final_start_pos: final_end_pos + 1]

        # 步骤1.6：修复常见的JSON语法错误
        # 思路：通过正则表达式移除对象或数组末尾的悬挂逗号（如 ,} 或 ,]），提高解析成功率。
        fixed_str = re.sub(r",(?=\s*[}\]])", "", potential_json_str)

        # 步骤1.7：尝试解析提取的JSON字符串
        # 思路：将修复后的字符串传入 json.loads，尝试转换为Python对象。
        return json.loads(fixed_str)
    except (ValueError, json.JSONDecodeError):
        # 步骤1.8：如果智能边界策略失败，回退到第2层策略（Markdown解析）
        # 思路：避免立即抛出错误，继续尝试更宽松的解析方法。
        pass

    # 步骤2：Markdown块回退策略
    # 思路：针对大模型返回的Markdown格式（如 ```json ... ```），通过正则表达式提取代码块中的JSON字符串，
    #       修复常见错误后尝试解析，适合处理包含Markdown标记的响应。
    try:
        # 步骤2.1：使用正则表达式查找```json```代码块
        # 思路：捕获 ```json 和 ``` 之间的内容，假设其为JSON字符串。
        match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
        if match:
            # 步骤2.2：提取代码块中的JSON字符串
            potential_json_str = match.group(1)
            # 步骤2.3：修复悬挂逗号
            # 思路：同步骤1.6，移除末尾多余逗号以修复语法。
            fixed_str = re.sub(r",(?=\s*[}\]])", "", potential_json_str)
            # 步骤2.4：尝试解析提取的JSON字符串
            return json.loads(fixed_str)
    except json.JSONDecodeError:
        # 步骤2.5：如果Markdown策略失败，回退到第3层策略（最终清理）
        # 思路：继续尝试更宽松的解析方法，避免直接失败。
        pass

    # 步骤3：最终清理与放手一搏
    # 思路：作为最后手段，清理整个字符串（替换中式标点、修复语法错误），尝试解析整个响应，
    #       适合处理非标准格式或严重错误的输入。
    try:
        # 步骤3.1：替换常见的中式标点
        # 思路：将中式标点（如顿号、引号）转换为JSON标准符号，增加解析成功率。
        cleaned_response = response.replace("、", ",").replace("“", '"').replace("”", '"')
        # 步骤3.2：再次修复悬挂逗号
        # 思路：确保清理后的字符串没有语法错误。
        fixed_str = re.sub(r",(?=\s*[}\]])", "", cleaned_response)
        # 步骤3.3：尝试解析清理后的字符串
        return json.loads(fixed_str)
    except json.JSONDecodeError:
        # 步骤3.4：如果所有策略失败，返回错误信息
        # 思路：记录原始响应，提供调试信息，避免程序崩溃。
        return {"error": "所有解析策略均告失败。", "original_response": response}
