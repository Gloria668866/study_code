from rouge import Rouge
generater = 'the cat is on the mat'
reference = ['a cat is on the mat']
rouge =Rouge()
print(rouge.get_scores(generater, reference[0]))

#
# from rouge_chinese import Rouge
#
# reference = '雷军 创立 了 小米 科技'
# candidate = '雷军 创立 小米 公司'
# rouge = Rouge()
# scores = rouge.get_scores(candidate, reference)
# print(scores[0])