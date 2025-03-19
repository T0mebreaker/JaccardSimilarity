# #  pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 transformers==4.26.0 -f

# 自动智慧问答系统相似度计算

import re
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import euclidean_distances  # 欧氏距离
from sklearn.metrics.pairwise import cosine_similarity  # 余弦距离
from transformers import BertTokenizer, BertModel
from transformers import BertConfig

model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'chinese-bert-wwm-ext')
#                                                    模型             分词器            词汇表
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)  # 定义分词器
bert_model = model_class.from_pretrained(pretrained_weights)  # 定义模型

def similar_count(vec1, vec2, model="cos"):
    if model == "eu":
        return euclidean_distances([vec1, vec2])[0][1]
    if model == "cos":
        return cosine_similarity([vec1, vec2])[0][1]

def bert_vec(text):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    # 使用这个分词器进行分词
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 把上边的文字转化为词汇表中对应的索引数字
    batch_tokenized = tokenizer.batch_encode_plus([text], padding=True, truncation=True, max_length=20)
    input_ids = torch.tensor(batch_tokenized['input_ids'])
    attention_mask = torch.tensor(batch_tokenized['attention_mask'])
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    bert_cls_hidden_state = bert_output[0][:, 0, :]
    return np.array(bert_cls_hidden_state[0].detach().numpy())

def re_pipei_word(text):
    data = pd.read_csv("湖北师范大学问答.csv",encoding='utf-8')
    sentence_corpus=data['问题'].values
    input_sentence = text

    # 将句子转换为小写，并使用空格分割成单词
    corpus_tokens = [set(sentence.lower().split()) for sentence in sentence_corpus]
    input_tokens = set(input_sentence.lower().split())

    # 计算相似度
    similarities = [len(input_tokens.intersection(tokens)) / len(input_tokens.union(tokens)) for tokens in
                    corpus_tokens]

    # 找到最相似的句子
    most_similar_index = similarities.index(max(similarities))
    most_similar_sentence = sentence_corpus[most_similar_index]
    # print(f"输入句子: {input_sentence}")
    # print(f"最相似的句子: {most_similar_sentence}")
    return most_similar_sentence
data = pd.read_csv("湖北师范大学问答.csv",encoding='utf-8')
print(data.head())
data_question=data["问题"].values
data_answer=data["答案"].values
print(len(data_question),len(data_answer))# 324393 324393
data=[]
for i in range(324393):
    temp=[]
    q=data_question[i]
    a=data_answer[i]
    if len(q) >2 and len(a)>2:
        score=similar_count(bert_vec(q),bert_vec(a), model="cos")
        temp.append(q)
        temp.append(a)
        temp.append(score)
        print("问题--：",q,"答案--：",a,"匹配度--：",score)
        print('*'*50)
        data.append(temp)
    if i==20:break # 设置数据只测试前20条 一共有324393数据
# data是一个以为度得列表 ，这样是把数据写进去一列
# name=["提问内容","回答内容","相似性分数"]
# test=pd.DataFrame(columns=name,data=data)
# test.to_csv('结果.csv')
