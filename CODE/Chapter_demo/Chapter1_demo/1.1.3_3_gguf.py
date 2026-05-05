# 确保已安装 llama-cpp-python 库，用于加载和运行 GGUF 格式的模型
# pip install llama-cpp-python

from llama_cpp import Llama  # 导入 llama_cpp 库的 Llama 类，用于加载和推理 GGUF 模型

# 定义 GGUF 模型文件的本地路径，需替换为实际路径
gguf_file_path = r"D:\CZ\bj_25AI_LLM\1_Lecture\models\Qwen3-0.6B-GGUF\Qwen3-0.6B-Q4_K_M.gguf"

# 打印正在加载模型的信息，显示模型文件路径
print(f"正在加载 GGUF 量化模型: {gguf_file_path}...")

# 使用 try-except 块捕获可能的错误（如文件路径错误或模型加载失败）
try:
    # 初始化 Llama 模型对象，加载指定的 GGUF 模型
    llm = Llama(
        model_path=gguf_file_path,  # 指定 GGUF 模型文件的路径
        n_gpu_layers=-1,  # 设置为 -1 表示尝试将所有层卸载到 GPU 以加速推理（需 GPU 支持）
        verbose=True,  # 启用详细日志输出，显示模型加载和推理的详细信息
        n_ctx=4096,  # 设置上下文窗口大小为 4096 令牌，支持更长的对话或输入
        n_batch=512,  # 设置批量处理大小为 512 令牌，影响推理速度和内存使用
        seed=42  # 设置随机种子为 42，确保推理结果可复现
    )
    # 模型加载成功后打印确认信息
    print("GGUF 模型加载完成。")

    # 定义符合 Qwen3 聊天模板的提示，包含用户问题和助手角色标记
    prompt = "<|im_start|>user\n中国首都是哪里？\n<|im_end|>\n<|im_start|>assistant\n"

    # 调用模型进行推理，生成回答
    output = llm(
        prompt,  # 输入的提示文本，包含用户问题
        max_tokens=150,  # 限制生成的最大令牌数为 150，避免过长输出
        stop=["<|im_end|>"],  # 设置停止标记，当生成 <|im_end|> 时停止
        temperature=0.7,  # 设置温度参数为 0.7，控制生成文本的随机性
        top_p=0.9  # 设置核采样参数为 0.9，控制生成文本的多样性
    )
    # 打印推理结果，去除首尾空白字符以确保输出整洁，这是 llama_cpp 库的标准输出结构，适用于该库的推理调用
    print("推理结果:", output['choices'][0]['text'].strip())

# 捕获并处理加载或推理过程中的异常
except Exception as e:
    # 打印错误信息，提示用户检查文件路径或环境配置
    print(f"加载模型失败，请检查文件路径是否正确: {e}")


"""
llama_cpp 是一个开源的 Python 库，全称 llama.cpp Python Bindings，用于加载和运行基于 GGUF 格式的大型语言模型（LLM），特别是由 llama.cpp 项目支持的模型。
它是 C++ 项目 llama.cpp 的 Python 接口，旨在让开发者能够高效地在本地设备上运行量化后的语言模型（如 LLaMA、Qwen 等），支持 CPU 和 GPU 推理。
"""