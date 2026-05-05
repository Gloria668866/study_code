import math
# --- 数据准备 ---
sentences = [['the', 'cat', 'sat']]
unigram = {'the': 0.5, 'cat': 0.4, 'sat': 0.2}
# --- 使用“联合概率”公式计算困惑度 ---
# 用于累加所有句子的困惑度
total_ppl = 0
print("开始使用 '联合概率' 公式计算困惑度...")
# 遍历语料库中的每一个句子
for sentence in sentences:
    print(f"\n--- 正在处理句子: {sentence} ---")
    # 步骤 1: 计算整个句子的联合概率 P(W)
    # 对应公式: P(W) = P(w_1) * P(w_2) * ... * P(w_N)
    joint_prob = 1.0  # 初始化联合概率为 1.0
    for word in sentence:
        prob = unigram[word]
        print(f"  - 词 '{word}' 的概率是: {prob}")
        joint_prob *= prob  # 将每个词的概率连乘起来

    print(f" -> 计算出的联合概率 P(W) = {joint_prob:.4f}")

    # 步骤 2: 计算该句子的困惑度 PPL(W)
    # 对应公式: PPL(W) = P(W)^(-1/N)
    N = len(sentence)  # 获取句子长度 N
    print(f" -> 句子长度 N = {N}")

    # 使用 a ** b 表示 a的b次方, 这行代码是公式的直接翻译
    ppl = joint_prob ** (-1.0 / N)
    print(f" -> 计算出的困惑度 PPL = {joint_prob:.4f} ** (-1.0 / {N}) = {ppl:.4f}")
    # 将当前句子的PPL累加到总和中
    total_ppl += ppl
# 计算最终在整个测试集上的平均困惑度
average_ppl = total_ppl / len(sentences)
print(f"\n==============================================")
print(f"所有句子的平均困惑度为: {average_ppl:.4f}")
print(f"=================当前仅{len(sentences)} 个句子=============================")