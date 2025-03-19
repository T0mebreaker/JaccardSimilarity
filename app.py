# coding:utf-8
# encoding='utf-8'
# pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
from flask import Flask, request, render_template, redirect, url_for, Flask,session
from datetime import timedelta
import os
# import pandas as pd
# import torch
# from torch import nn
# from torch import optim
# import transformers as tfs
# import math
# import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import warnings
import re
import jieba
# from transformers import BertTokenizer, BertModel
# from transformers import BertConfig
# from transformers import AutoTokenizer, AutoModel,AutoConfig
warnings.filterwarnings('ignore')
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
plt.rcParams["font.sans-serif"] = ['Simhei']
plt.rcParams["axes.unicode_minus"] = False
from pylab import *
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# import torch.optim as optim
# from torch.utils.data import random_split
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import joblib
# from gensim.models import KeyedVectors
# from gensim.models import Word2Vec
# import gensim
# from gensim.models.word2vec import LineSentence

# 输入句子测试：
# data = pd.read_csv('湖北师范大学问答.csv', header=0)
# ceshi =data.values[:300]  #
# res=[]
# for line in ceshi:

# print(line)
# result=ceshi_def(line, model)
# print("结果：",result)
def re_pipei_word(text):
    data = pd.read_csv("湖北师范大学问答.csv",encoding='utf-8')
    sentence_corpus=data['问题'].values[:200]
    input_sentence = text
    # 使用 jieba 进行中文分词
    corpus_tokens = [set(jieba.lcut(sentence)) for sentence in sentence_corpus]
    input_tokens = set(jieba.lcut(input_sentence))

    # 计算相似度
    similarities = [len(input_tokens.intersection(tokens)) / len(input_tokens.union(tokens)) for tokens in
                    corpus_tokens]

    # 找到最相似的句子
    most_similar_index = similarities.index(max(similarities))
    most_similar_sentence = sentence_corpus[most_similar_index]
    # print(f"输入句子: {input_sentence}")
    # print(f"最相似的句子: {most_similar_sentence}")
    return most_similar_sentence
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

#首页
@app.route('/')  # 接口地址
def index():
    return redirect("/main")
# 首页
@app.route("/main")
def main():
    return render_template('main.html', keyword="", results=[])
# 搜索
df= pd.read_csv("湖北师范大学问答.csv",encoding='utf-8')
@app.route('/search')  #接口地址 获取数据返回数据的
def search():
    keyword = request.args.get('keyword')
    print("keyword:",keyword)
    # 获取 'Bob' 的整行数据
    res_juzi=re_pipei_word(keyword)
    print('res_juzi',res_juzi)
    row_data = df[df['问题'] == res_juzi]
    answer = row_data['答案'].values[0]
    if len(str(keyword))!=0:
          score=[str(str("问题：")+str(res_juzi)+str("，")+str("答案：")+str(answer))]
    else:score=[]
    print("score:",score)
    return render_template('main.html', keyword=keyword, results=score)

@app.route('/detail')  #接口地址
def detail():
    news_id = request.args.get('id')
    news = pd.read_pickle("data/news.pkl")
    for info in news:
        if str(info["id"]) == str(news_id):
            break
    return render_template('detail.html', news=info)
# web 服务器
if __name__ == '__main__':
    app.run(port=8001, debug=True)
