import jieba
import pandas as pd
data = pd.read_csv("湖北师范大学问答.csv",encoding='utf-8')
sentence_corpus = data['问题'].values[:200]
input_sentence = '武汉大学是一所什么类型的学校'
# 使用 jieba 进行中文分词
corpus_tokens = [set(jieba.lcut(sentence)) for sentence in sentence_corpus]
input_tokens = set(jieba.lcut(input_sentence))

# 计算相似度
similarities = [len(input_tokens.intersection(tokens)) / len(input_tokens.union(tokens)) for tokens in
                corpus_tokens]

# 找到最相似的句子
most_similar_index = similarities.index(max(similarities))
most_similar_sentence = sentence_corpus[most_similar_index]
print(f"输入句子: {input_sentence}")
print(f"最相似的句子: {most_similar_sentence}")