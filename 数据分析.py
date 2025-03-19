# pip install pip install --upgrade numpy
#TFIDF word2vec 关键词聚类分析 词云 提取关键词 聚类分析词性
# 词性相近的词语 聚类分析是相近的
#figure1纵向是答案出现频率，横向是问题,图一表示
import gensim
import requests
import pandas as pd
import math
import re
import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import Counter
#jay
# sentences = [['first', 'sentence','is'], ['second', 'sentence','is']]
# w2v_model = gensim.models.Word2Vec(sentences, min_count=1,vector_size=64,window=3)
# vec = w2v_model.wv["is"]
# print(vec)
def dict_sort(dic):
    """
    字典排序
    """
    count = sorted(dic.items(), key=lambda x: x[1], reverse=True)  # True 是降序 False是升序
    return count

def plt_picture_ciyun(n):
    """
    词云
    
    """
    wc = WordCloud(
        # 设置字体，不指定就会出现乱码
        font_path='simhei.ttf',
        # 设置背景色
        background_color='white',
        # 设置背景宽
        width=500,
        # 设置背景高
        height=350,
        # 最大字体
        max_font_size=50,
        # 最小字体
        min_font_size=10,
        mode='RGBA'
        # colormap='pink'
    )
    # 产生词云
    wc.generate(n)
    # 显示图片
    # 指定所绘图名称
    plt.figure("jay")

    # 以图片的形式显示词云
    plt.imshow(wc)
    # 关闭图像坐标系
    plt.axis("off")
    # 保存词云图片
    # plt.savefig("2209070221.png")
    plt.show()

def re_pipei_word(text):
    """
    正则提取文本的汉字
    text = "也像疼111WWW%%    __"
    """
    res = re.findall('[\u4e00-\u9fa5]', str(text))
    # print(res) #['也', '像', '疼']
    # print("".join(res)) 也像疼
    return "".join(res)

def tfidf_word(corpus):
    """
    tfodf 提取关键词：
    import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = ['TF-IDF 主要 思想 是', '算法 一个 重要 特点 可以 脱离 语料库 背景',
              '如果 一个 网页 被 很多 其他 网页 链接 说明 网页 重要',
              '原始 文本 进行 标记',
              '主要 思想']
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    data = {'word': vectorizer.get_feature_names_out(),
            'tfidf': X.toarray().sum(axis=0).tolist()}
    df = pd.DataFrame(data)
    df = df.sort_values(by='tfidf', ascending=True)  # True是从小达到 False是从大到小
    res = {}
    for i in df.values:
        res[i[0]] = i[1]
    # print(res){'链接': 0.2556396904397093, '说明': 0.2556396904397093...}
    return res


def jieba_cut_word(text):
    """
    jieba分词
    text=滴滴代驾不靠谱，在司机端总是接不到单子
    """
    res1 = jieba.lcut(text, cut_all=False)  # 不存在多余词
    # ['滴滴', '代驾', '不靠', '谱', '，', '在', '司机', '端', '总是', '接', '不到', '单子', '。']

    res2 = jieba.lcut("今天空车返回，在路上遇到行政执法，平台不派单。", cut_all=True)  # 有多余词
    # ['今天', '天空', '空车', '返回', '，', '在', '路上', '遇到', '行政', '执法', '，', '平台', '不', '派', '单', '。']
    return res1
data = pd.read_csv("湖北师范大学问答.csv", encoding='utf-8')
sentence_corpus = data['答案'].values[:200]
corpus=[]
sentences=[]
for idx,line in enumerate(sentence_corpus):
    line=re_pipei_word(line)
    line=jieba_cut_word(line)
    sentences.append(line)
    line=" ".join(line)
    # print(line)
    corpus.append(line)
    # if idx==500:break
w2v_model = gensim.models.Word2Vec(sentences, min_count=1,vector_size=64,window=5)
# vec = w2v_model.wv["咳嗽"]
res=tfidf_word(corpus)

res=dict_sort(res)
print(res)
n=""
word_list=[]
for idx,(k ,v) in enumerate(res):
    # print(k,v)
    word_list.append(k)
    n=n+k+" "
    # if idx==100:break
plt_picture_ciyun(n)
print(word_list)
word_list_vec=[]
for word in word_list:
    vec = w2v_model.wv[word]#得到词的向量
    # print(vec)
    word_list_vec.append([word,vec])

from gensim.models import Word2Vec
from random import sample
from sklearn.manifold import TSNE
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #中文字体
mpl.rcParams['axes.unicode_minus'] = False #防止负号出现异常显示
#进行图的选取 选取两个图的点在一个图中显示！！！！！！！
plt.figure(figsize=(15,15)) #定义画布大小
color=['b',"r","g","k"] # 定义颜色 参数c 可以等于：['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
color_label=['b',"r","g","k"]
marker=[" "," "," "," "]
tokens = []
labels = []
for line in word_list_vec:
    labels.append(line[0])
    tokens.append(line[1]) # 存储的是向量

tsne_model = TSNE(perplexity=10, n_components=2, init='pca', n_iter=2500, random_state=23)
#  perplexity: 默认为30，数据集越大，需要参数值越大，建议值位5-50 , n_components=2 默认为2，嵌入空间的维度（嵌入空间的意思就是结果空间）,别的参数估计不重要
print(len(tokens))
# print(tokens)
new_values = tsne_model.fit_transform(np.array(tokens))
#     将X投影到一个嵌入空间并返回转换结果
#降维处理
#     print(new_values)
x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])
for i in range(len(x)):
    plt.scatter(x[i],y[i],c=color[1],marker=marker[1])
    plt.text(x[i],y[i], labels[i], fontsize=10,color=color_label[1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()
