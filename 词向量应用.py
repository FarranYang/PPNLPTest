#!/usr/bin/env python
# coding: utf-8

# # 作业
# 
# 更换TokenEmbedding预训练模型，使用VisualDL查看相应的TokenEmbedding可视化效果，并尝试更换后的TokenEmbedding计算句对语义相似度。
# 本作业详细步骤，可参考[Day01作业教程](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/education/day01.md)，记得star PaddleNLP，收藏起来，随时跟进最新功能噢。
# 
# **作业结果提交**：
# 1. 截图提交可视化结果（图片注明作业可视化结果）。
# 2. 通篇执行每段代码，并保留执行结果。

# # PaddleNLP词向量应用展示
# 
# 6.7日NLP直播打卡课开始啦
# 
# **[直播链接请戳这里，每晚20:00-21:30👈](http://live.bilibili.com/21689802)**
# 
# **[课程地址请戳这里👈](https://aistudio.baidu.com/aistudio/course/introduce/24177)**
# 
# 欢迎来课程**QQ群**（群号:618354318）交流吧~~
# 
# 
# 词向量（Word embedding），即把词语表示成实数向量。“好”的词向量能体现词语直接的相近关系。词向量已经被证明可以提高NLP任务的性能，例如语法分析和情感分析。
# 
# <p align="center">
# <img src="https://ai-studio-static-online.cdn.bcebos.com/54878855b1df42f9ab50b280d76906b1e0175f280b0f4a2193a542c72634a9bf" width="60%" height="50%"> <br />
# </p>
# <br><center>图1：词向量示意图</center></br>
# 
# PaddleNLP已预置多个公开的预训练Embedding，您可以通过使用`paddlenlp.embeddings.TokenEmbedding`接口加载预训练Embedding，从而提升训练效果。本篇教程将依次介绍`paddlenlp.embeddings.TokenEmbedding`的初始化和文本表示效果，并通过文本分类训练的例子展示其对训练提升的效果。

# In[1]:


get_ipython().system('pip install --upgrade paddlenlp -i https://pypi.org/simple')


# ## 加载TokenEmbedding
# 
# `TokenEmbedding()`参数
# - `embedding_name`
# 将模型名称以参数形式传入TokenEmbedding，加载对应的模型。默认为`w2v.baidu_encyclopedia.target.word-word.dim300`的词向量。
# - `unknown_token`
# 未知token的表示，默认为[UNK]。
# - `unknown_token_vector`
# 未知token的向量表示，默认生成和embedding维数一致，数值均值为0的正态分布向量。
# - `extended_vocab_path`
# 扩展词汇列表文件路径，词表格式为一行一个词。如引入扩展词汇列表，trainable=True。
# - `trainable`
# Embedding层是否可被训练。True表示Embedding可以更新参数，False为不可更新。默认为True。

# In[3]:


from paddlenlp.embeddings import TokenEmbedding

# 初始化TokenEmbedding， 预训练embedding未下载时会自动下载并加载数据
# 需要更换所选的词向量
token_embedding = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300")

# 查看token_embedding详情
print(token_embedding)


# ### 认识一下Embedding
# **`TokenEmbedding.search()`**
# 获得指定词汇的词向量。

# In[4]:


test_token_embedding = token_embedding.search("中国")
print(test_token_embedding)


# **`TokenEmbedding.cosine_sim()`**
# 计算词向量间余弦相似度，语义相近的词语余弦相似度更高，说明预训练好的词向量空间有很好的语义表示能力。

# In[6]:


score1 = token_embedding.cosine_sim("女孩", "男孩")
score2 = token_embedding.cosine_sim("女孩", "书籍")
print('score1:', score1)
print('score2:', score2)


# ### 词向量映射到低维空间
# 
# 使用深度学习可视化工具[VisualDL](https://github.com/PaddlePaddle/VisualDL)的[High Dimensional](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md#High-Dimensional--%E6%95%B0%E6%8D%AE%E9%99%8D%E7%BB%B4%E7%BB%84%E4%BB%B6)组件可以对embedding结果进行可视化展示，便于对其直观分析，步骤如下：
# 
# 1. 升级 VisualDL 最新版本。
# 
# `pip install --upgrade visualdl`
# 
# 2. 创建LogWriter并将记录词向量。
# 
# 3. 点击左侧面板中的可视化tab，选择‘token_hidi’作为文件并启动VisualDL可视化

# In[7]:


get_ipython().system('pip install --upgrade visualdl')


# In[8]:


# 获取词表中前1000个单词
labels = token_embedding.vocab.to_tokens(list(range(0, 1000)))
# 取出这1000个单词对应的Embedding
test_token_embedding = token_embedding.search(labels)

# 引入VisualDL的LogWriter记录日志
from visualdl import LogWriter

with LogWriter(logdir='./token_hidi') as writer:
    writer.add_embeddings(tag='test', mat=[i for i in test_token_embedding], metadata=labels)


# #### 启动VisualDL查看词向量降维效果
# 启动步骤：
# - 1、切换到「可视化」指定可视化日志
# - 2、日志文件选择 'token_hidi'
# - 3、点击「启动VisualDL」后点击「打开VisualDL」，选择「高维数据映射」，即可查看词表中前1000词UMAP方法下映射到三维空间的可视化结果:
# 
# ![](https://user-images.githubusercontent.com/48054808/120594172-1fe02b00-c473-11eb-9df1-c0206b07e948.gif)
# 
# 可以看出，语义相近的词在词向量空间中聚集(如数字、章节等)，说明预训练好的词向量有很好的文本表示能力。
# 
# 使用VisualDL除可视化embedding结果外，还可以对标量、图片、音频等进行可视化，有效提升训练调参效率。关于VisualDL更多功能和详细介绍，可参考[VisualDL使用文档](https://github.com/PaddlePaddle/VisualDL/tree/develop/docs)。

# ## 基于TokenEmbedding衡量句子语义相似度
# 
# 在许多实际应用场景（如文档检索系统）中， 需要衡量两个句子的语义相似程度。此时我们可以使用词袋模型（Bag of Words，简称BoW）计算句子的语义向量。
# 
# **首先**，将两个句子分别进行切词，并在TokenEmbedding中查找相应的单词词向量（word embdding）。
# 
# **然后**，根据词袋模型，将句子的word embedding叠加作为句子向量（sentence embedding）。
# 
# **最后**，计算两个句子向量的余弦相似度。
# 
# ### 基于TokenEmbedding的词袋模型
# 
# 
# 使用`BoWEncoder`搭建一个BoW模型用于计算句子语义。
# 
# * `paddlenlp.TokenEmbedding`组建word-embedding层
# * `paddlenlp.seq2vec.BoWEncoder`组建句子建模层
# 

# In[9]:


import paddle
import paddle.nn as nn
import paddlenlp


class BoWModel(nn.Layer):
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        emb_dim = self.embedder.embedding_dim
        self.encoder = paddlenlp.seq2vec.BoWEncoder(emb_dim)
        self.cos_sim_func = nn.CosineSimilarity(axis=-1)

    def get_cos_sim(self, text_a, text_b):
        text_a_embedding = self.forward(text_a)
        text_b_embedding = self.forward(text_b)
        cos_sim = self.cos_sim_func(text_a_embedding, text_b_embedding)
        return cos_sim

    def forward(self, text):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # Shape: (batch_size, embedding_dim)
        summed = self.encoder(embedded_text)

        return summed

model = BoWModel(embedder=token_embedding)


# ### 构造Tokenizer
# 使用TokenEmbedding词表构造Tokenizer。

# In[10]:


from data import Tokenizer
tokenizer = Tokenizer()
tokenizer.set_vocab(vocab=token_embedding.vocab)


# ### 相似句对数据读取
# 
# 以提供的样例数据text_pair.txt为例，该数据文件每行包含两个句子。
# 

# In[11]:


text_pairs = {}
with open("text_pair.txt", "r", encoding="utf8") as f:
    for line in f:
        text_a, text_b = line.strip().split("\t")
        if text_a not in text_pairs:
            text_pairs[text_a] = []
        text_pairs[text_a].append(text_b)


# ### 查看相似语句相关度

# In[12]:


for text_a, text_b_list in text_pairs.items():
    text_a_ids = paddle.to_tensor([tokenizer.text_to_ids(text_a)])

    for text_b in text_b_list:
        text_b_ids = paddle.to_tensor([tokenizer.text_to_ids(text_b)])
        print("text_a: {}".format(text_a))
        print("text_b: {}".format(text_b))
        print("cosine_sim: {}".format(model.get_cos_sim(text_a_ids, text_b_ids).numpy()[0]))
        print()


# ### 使用VisualDL查看句子向量

# In[13]:


# 引入VisualDL的LogWriter记录日志
import numpy as np
from visualdl import LogWriter    
# 获取句子以及其对应的向量
label_list = []
embedding_list = []

for text_a, text_b_list in text_pairs.items():
    text_a_ids = paddle.to_tensor([tokenizer.text_to_ids(text_a)])
    embedding_list.append(model(text_a_ids).flatten().numpy())
    label_list.append(text_a)

    for text_b in text_b_list:
        text_b_ids = paddle.to_tensor([tokenizer.text_to_ids(text_b)])
        embedding_list.append(model(text_b_ids).flatten().numpy())
        label_list.append(text_b)


with LogWriter(logdir='./sentence_hidi') as writer:
    writer.add_embeddings(tag='test', mat=embedding_list, metadata=label_list)


# ### 启动VisualDL观察句子向量降维效果
# 
# 步骤如上述观察词向量降维效果一模一样。
# ![](https://ai-studio-static-online.cdn.bcebos.com/0e876f3cf1724e90a317ad3f4be233a9eb0313b0e92f475b95675c2ad52d3eb0)
# 
# 
# 可以看出，语义相近的句子在句子向量空间中聚集(如有关课堂的句子、有关化学描述句子等)。

# # PaddleNLP更多预训练词向量
# PaddleNLP提供61种可直接加载的预训练词向量，训练自多领域中英文语料、如百度百科、新闻语料、微博等，覆盖多种经典词向量模型（word2vec、glove、fastText）、涵盖不同维度、不同语料库大小，详见[PaddleNLP Embedding API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/embeddings.md)。

# # 预训练词向量辅助分类任务
# 
# 想学习词向量更多应用，来试试预训练词向量对分类模型的改善效果吧，[这里](https://aistudio.baidu.com/aistudio/projectdetail/1283423) 试试把`paddle.nn.Embedding`换成刚刚学到的预训练词向量吧。

# # 加入课程交流群，一起学习吧
# 
# 现在就加入课程群，一起交流NLP技术吧！
# 
# <img src="https://ai-studio-static-online.cdn.bcebos.com/d953727af0c24a7c806ab529495f0904f22f809961be420b8c88cdf59b837394" width="200" height="250" >
# 
# 
# 
# **[直播链接请戳这里，每晚20:00-21:30👈](http://live.bilibili.com/21689802)**
# 
# **[还没有报名课程？赶紧戳这里，课程、作业安排统统在课程区哦👉🏻](https://aistudio.baidu.com/aistudio/course/introduce/24177)**
