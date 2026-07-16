from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
candidate = ['this','cat','sat','on','the','car']
reference = [['this','cat','is','one','the','mat'],['this','cat','is','two','the','mat'],['this','cat','is','two','the','mat']]
print(sentence_bleu(references=reference, hypothesis=candidate, weights=(1, 0, 0, 0))) #1- gram
print(sentence_bleu(references=reference, hypothesis=candidate, weights=(0.5, 0.5, 0, 0))) #2- gram
print(sentence_bleu(references=reference, hypothesis=candidate, weights=(0.33, 0.33, 0.33, 0))) #3- gram
print(sentence_bleu(references=reference, hypothesis=candidate, weights=(0.25, 0.25, 0.25, 0.25))) #4- gram