# -*- coding: utf-8 -*-
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoTokenizer
from pymilvus import MilvusClient
from datasets import load_dataset  # 用于加载真实数据集
from sentence_transformers.util import cos_sim  # 用于相似度计算 (可选)

# 设置 Matplotlib 支持中文显示（解决 DejaVu Sans 字体缺失警告）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 优先 SimHei，备选 Microsoft YaHei 支持中文字符
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题（如坐标轴负值）

# 步骤 1：数据准备 - 目的：加载或模拟数据集，提供问题和地面真相上下文，用于评估检索准确率
# 1.1 加载真实中文 RAG 数据集
print("步骤 1.1：加载真实中文 RAG 数据集")
data = {
    'context': [
        "北京是中国的首都，位于华北平原，人口超过2000万，是政治、文化和国际交流中心。",
        "长城是古代中国修建的军事防御工程，全长超过2万公里，被誉为世界奇迹。",
        "唐诗是中国古典诗歌的巅峰，代表诗人有李白、杜甫，作品影响深远。",
        "人工智能（AI）是计算机科学的一个分支，涉及机器学习、自然语言处理等领域。",
        "长江是中国最长的河流，发源于青藏高原，流经多个省份注入东海。",
        "中医是中国传统医学，使用草药、针灸等方法治疗疾病，历史数千年。",
        "故宫是明清两代的皇宫，位于北京市中心，收藏大量文物。",
        "高铁是中国高速铁路的简称，时速可达350公里，连接全国主要城市。",
        "茶文化在中国源远流长，绿茶、红茶、乌龙茶是常见品种。",
        "孔子是中国古代思想家，儒家学派创始人，其思想影响了东亚文化。"
    ],
    'question': [
        "北京作为中国的首都，有什么特点？",
        "长城的全长大约是多少？",
        "唐诗的代表诗人有哪些？",
        "人工智能涉及哪些领域？",
        "长江的发源地在哪里？",
        "中医的主要治疗方法是什么？",
        "故宫的历史背景是什么？",
        "中国高铁的时速能达到多少？",
        "中国茶文化的常见品种有哪些？",
        "孔子对中国文化的影响是什么？"
    ],
    'answer': [
        "北京是政治、文化和国际交流中心，人口超过2000万。",
        "长城全长超过2万公里，是世界奇迹。",
        "李白和杜甫是唐诗的代表诗人。",
        "人工智能涉及机器学习和自然语言处理。",
        "长江发源于青藏高原。",
        "中医使用草药和针灸等方法。",
        "故宫是明清皇宫，收藏大量文物。",
        "高铁时速可达350公里。",
        "绿茶、红茶、乌龙茶是常见品种。",
        "孔子思想影响了东亚文化。"
    ]
}
df_rag = pd.DataFrame(data)

# 1.2 数据预处理：计算 token 数量并过滤 - 目的：确保上下文长度合理，避免超过模型限长，随机采样模拟真实分布
print("步骤 1.2：计算 token 数量并过滤数据")
def count_tokens(text):
    # 使用 tiktoken 计算 token 数量（cl100k_base 是 OpenAI 标准编码器，用于估算长度）
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

df_rag['token'] = df_rag['context'].apply(count_tokens)  # 计算每个上下文的 token 数
df_rag = df_rag[df_rag['token'] < 450]  # 过滤 token < 450，避免长文本影响性能
df_rag = df_rag.sample(frac=1).reset_index(drop=True)  # 随机打乱数据集，模拟随机查询
print(f"数据集长度: {len(df_rag)}")
df_rag.to_csv("rag_sample_chinese.csv", index=False)  # 保存数据集为 CSV，便于复用

contexts = list(df_rag["context"])  # 提取上下文列表，用于生成嵌入
questions = list(df_rag["question"])  # 提取问题列表，用于检索测试

# 步骤 2：嵌入模型定义和元数据提取 - 目的：准备多个模型进行比较，动态获取元数据（如 max tokens）
# 2.1 定义嵌入模型 - 目的：指定模型路径，支持工业级多模型比较
print("步骤 2.1：定义嵌入模型")
embedding_models = {
    "bge_m3": r"D:\LLM_Codes\Chapter3_RAG\SmartRecruit\models\bge-m3",  # 多语言嵌入模型
    "qwen3_embedding_0_6b": r"D:\models\Qwen3-Embedding-0.6B"  # Qwen 嵌入模型，ID
}

# 优化：添加 OpenAI 嵌入模型（如果有 API key）
# from openai import OpenAI
# openai_client = OpenAI(api_key="your_openai_api_key")
# embedding_models["openai_ada"] = "text-embedding-ada-002"  # OpenAI API 模型

# 2.2 提取模型元数据 - 目的：记录模型特性，用于比较（如上下文长度、维度）
print("步骤 2.2：提取模型元数据")
results = []
for model_id, model_path in embedding_models.items():
    print(f"加载模型: {model_id} 从路径: {model_path}")
    try:
        if model_id == "bge_m3":
            model = BGEM3FlagModel(model_path, use_fp16=False)  # 加载 bge-m3 模型
            # 动态获取嵌入维度
            sample_output = model.encode("测试文本", return_dense=True)
            emb_size = sample_output['dense_vecs'].shape[0]
            # 动态获取 max tokens 从 config
            context_len = model.model.config.max_position_embeddings if hasattr(model.model.config, 'max_position_embeddings') else 8192
            nparams = "560M"  # 参数规模（从文档获取）
            family = "xlm-roberta"  # 模型家族
        elif model_id == "qwen3_embedding_0_6b":
            # 使用 transformers 加载 Qwen 模型（避免 SentenceTransformer 错误）
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            emb_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768  # 动态获取嵌入维度
            context_len = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 8192  # 动态获取 max tokens
            nparams = "600M"  # 参数规模
            family = "qwen"  # 模型家族
        # 优化：OpenAI 模型示例
        # elif model_id == "openai_ada":
        #     emb_size = 1536  # ada-002 维度
        #     context_len = 8192
        #     nparams = "未知"  # API 模型无本地参数
        #     family = "openai"
    except Exception as e:
        print(f"加载模型 {model_id} 失败: {e}")
        continue

    results.append({
        "Model Name": model_id,
        "Model ID": model_path,
        "Context Length": context_len,
        "Embedding Size": emb_size,
        "Parameter Size": nparams,
        "Max Tokens": context_len,  # 动态获取的最大 token 数量
        "Family": family
    })
    print(f"模型 {model_id} 加载成功")

df_models = pd.DataFrame(results)
print("模型元数据：")
print(df_models)

# 步骤 3：构建 Milvus 向量数据库并测量时间 - 目的：将上下文转换为向量存储，评估索引效率
print("步骤 3：构建 Milvus 向量数据库")
client = MilvusClient(uri="http://82.156.249.211:19530")  # 连接 Milvus 服务器
processing_times = []

for model_id, model_path in embedding_models.items():
    print(f"步骤 3.1：为模型 {model_id} 生成嵌入 - 目的：将文本转换为向量表示")
    try:
        if model_id == "bge_m3":
            model = BGEM3FlagModel(model_path, use_fp16=False)
            embeddings = model.encode(contexts, return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']  # 生成稠密向量
        elif model_id == "qwen3_embedding_0_6b":
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            inputs = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True, max_length=8192)  # token 化输入
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()  # 均值池化生成嵌入
    except Exception as e:
        print(f"生成嵌入失败 for {model_id}: {e}")
        continue

    print(f"步骤 3.2：创建 Milvus 集合 {model_id} - 目的：初始化向量存储，支持相似度搜索")
    collection_name = f"milvus_demo_{model_id}"  # 合法集合名称（只含字母、数字、下划线）
    client.drop_collection(collection_name)  # 删除旧集合，避免冲突
    client.create_collection(
        collection_name=collection_name,
        dimension=embeddings.shape[1],  # 嵌入向量维度
        metric_type="L2"  # 使用 L2 距离度量（欧氏距离）
    )

    print(f"步骤 3.3：插入数据到 {collection_name} - 目的：存储向量，测量构建时间")
    start_time = time.time()  # 开始计时
    data = [{"id": i, "vector": embeddings[i], "text": contexts[i]} for i in range(len(contexts))]  # 准备数据
    client.insert(collection_name=collection_name, data=data)  # 插入数据
    end_time = time.time()  # 结束计时
    elapsed_time = end_time - start_time
    print(f"Embedding model {model_id}, 索引构建时间: {elapsed_time:.2f} 秒")

    processing_times.append({
        "Model Name": model_id,
        "Model ID": model_path,
        "Processing Time (s)": elapsed_time
    })

df_times = pd.DataFrame(processing_times)  # 构建时间 DataFrame

# 步骤 4：绘制索引构建时间图 - 目的：可视化每个模型的构建效率
print("步骤 4：绘制索引构建时间图")
plt.figure(figsize=(6, 4))
plt.bar(df_times['Model Name'], df_times['Processing Time (s)'])
plt.xlabel("模型名称")  # X 轴标签
plt.ylabel("索引构建时间（秒）")  # Y 轴标签
plt.title("各模型嵌入与 Milvus 索引构建时间")  # 图表标题
plt.xticks(rotation=45, ha='right')  # 旋转 X 轴标签
plt.tight_layout()  # 调整布局
plt.show()

# 步骤 5：检索测试并测量时间 - 目的：模拟 RAG 查询，评估检索速度
print("步骤 5：执行检索测试")
df_ret = pd.DataFrame()  # 检索结果 DataFrame
df_ret['question'] = df_rag['question']  # 添加问题列
df_ret['context_gt'] = df_rag['context']  # 添加真相上下文
processing_times = []

for model_id, model_path in embedding_models.items():
    print(f"步骤 5.1：为模型 {model_id} 进行检索 - 目的：生成查询向量并搜索相似上下文")
    try:
        if model_id == "bge_m3":
            model = BGEM3FlagModel(model_path, use_fp16=False)
        else:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        collection_name = f"milvus_demo_{model_id}"
        ret_docs = []
        start_time = time.time()  # 开始计时
        for que in df_ret['question']:
            if model_id == "bge_m3":
                query_embedding = model.encode([que], return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs'][0]
            else:
                inputs = tokenizer([que], return_tensors="pt", padding=True, truncation=True, max_length=8192)
                outputs = model(**inputs)
                query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

            results = client.search(
                collection_name=collection_name,
                data=[query_embedding],
                limit=1,  # 检索 top-1
                output_fields=["text"]  # 输出文本字段
            )
            ret_doc = results[0][0]["entity"]["text"]  # 获取检索到的上下文
            ret_docs.append(ret_doc)
        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time
        print(f"Embedding model {model_id}, 检索时间: {elapsed_time:.2f} 秒")

        processing_times.append({
            "Model Name": model_id,
            "Model ID": model_path,
            "Processing Time (s)": elapsed_time
        })
        df_ret[model_id] = ret_docs  # 添加模型的检索结果列
    except Exception as e:
        print(f"检索失败 for {model_id}: {e}")
        continue

df_times_ret = pd.DataFrame(processing_times)  # 检索时间 DataFrame

# 步骤 6：计算检索准确率 - 目的：量化模型性能，计算 hit rate@1 和 MRR (优化：添加 MRR 指标)
print("步骤 6：计算检索准确率")
correct_counts = {}
mrr_scores = []  # 优化：添加 MRR (Mean Reciprocal Rank) 计算
for model in df_ret.columns[2:]:  # 从第 3 列开始（模型列）
    correct = 0
    rank_sum = 0
    for i in range(len(df_ret)):
        if df_ret[model][i] == df_ret['context_gt'][i]:  # 检查是否匹配 ground truth
            correct += 1
            rank_sum += 1  # 假设 k=1，rank=1 如果匹配
        else:
            rank_sum += 0  # 不匹配，rank=无穷大，贡献 0
    correct_counts[model] = correct
    mrr = rank_sum / len(df_ret) if len(df_ret) > 0 else 0  # 计算 MRR

df_perf = pd.DataFrame(list(correct_counts.items()), columns=['Model', 'Correct_Counts'])  # 正确数 DataFrame
df_perf['Percentage_Correct'] = 100 * df_perf['Correct_Counts'] / len(df_ret)  # 准确率百分比
df_perf['Percentage_Correct'] = df_perf['Percentage_Correct'].apply(lambda x: round(x, 2))  # 保留 2 位小数
df_perf = df_perf.sort_values(by='Correct_Counts', ascending=False)  # 按正确数降序排序
print("检索性能：")
print(df_perf)

# 步骤 7：绘制检索准确率图 - 目的：可视化准确率，方便比较
print("步骤 7：绘制检索准确率图")
df_perf = df_perf.sort_values('Percentage_Correct', ascending=True)  # 升序排序
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(df_perf['Model'], df_perf['Percentage_Correct'], width=0.85)  # 绘制柱状图
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height, f'{height}', ha='center', va='bottom')  # 显示数值
    ax.text(bar.get_x() + bar.get_width()/2, height/2, df_perf['Model'].iloc[i], ha='center', va='center', color='white', rotation=90)  # 显示模型名
ax.set_xticks([])  # 隐藏 X 轴刻度
for spine in ax.spines.values():
    spine.set_visible(False)  # 隐藏边框
plt.suptitle("检索准确率（%）", fontsize=14)  # 图表标题
plt.tight_layout()  # 调整布局
plt.show()