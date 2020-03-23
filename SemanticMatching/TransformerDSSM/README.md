---
layout:     post
title:      基于Transformer的深度语义匹配模型DSSM
subtitle:   语义匹配系列（一）
date:       2020-03-23
author:     MeteorMan
header-img: img/post-bg-map.jpg
catalog: true
tags:
    - Natural Language Processing
    - Semantic Matching
    - Deep Learning



---

>万丈高楼平地起！

​	MeteorMan目前的研究方向是问答系统和知识图谱，在研究的过程中愈加感受到语义匹配的重要性，因此特此开篇，记录并讲述语义匹配相关知识。

​	一个完整的问答系统处理流程主要分为3个部分：1.解析；2.匹配；3.生成。本文首先对匹配部分所用方法进行简单介绍，后面将详细介绍基于Transformer的语义匹配模型以及开源实现。

# 1.业界采用的匹配技术方案

​	匹配的过程本质上可以说是信息检索的过程，其主要分为两类：基于关键词的信息检索、基于语义的信息检索。

## 1.1 基于关键词的信息检索

​	基于关键词的信息检索，主要是通过计算输入问题中的每个单词与待匹配答案中单词的相关性来实现检索功能，这个相关性由以下三部分组成：

- 问题中的单词w和答案A之间的相关性；
- 问题中的单词w和问题Q本身之间的相关性；
- 单词w所占权重。

常见的基于关键词信息检索的方法主要都是基于上述三个部分来计算相关性的，例如在TF-IDF中，TF（词频）主要就是用于计算问题中的单词w和问题Q本身的相关性，而IDF（逆文档频率）则用于计算单词w的权重。BM25则是在原有基础上考虑了答案长度等因素，是一种对TF-IDF的改进的检索方法。

## 1.2 基于语义的信息检索

​	基于语义的信息检索，即通过深度学习的方法，以一种端到端的方式来计算问题和答案、问题和文章、问题与问题甚至单词与单词之间的相关性，其模型实现框架主要分为4类：

- DSSM
- Q-Q Match
- 记忆网络（Memory Network）
- 基于深层语义的图匹配算法

其中，对于上面4种方法来说，首先需要解决的都是如何对一个句子或一篇文章进行表示，将其转换成对应的特征向量，常见的表示方法可通过BiRNN、Attention方式实现。DSSM与Q-Q Match的主要区别就在于，DSSM得到文章和问题的表示之后，通过计算二者之间的余弦相似度来计算相关性，而Q-Q Match则是将二者的表示进行拼接，带入到一个MLP中，进行分类，1则为相关，0则为不相关，也可以进行更细粒度的分类（如相关性级别1-5，5为最高级别的相关）。

而记忆网络和图匹配算法，更多的用于机器阅读理解（基于开放领域的问答）和基于知识图谱的问答中，用于解决知识推理相关的问题。

# 2. 基于Transformer的语义匹配模型DSSM

​	该模型在DSSM模型的基础上，将模型的表示层使用基于Transformer的Encoder部分来实现，匹配层将通过表示层得到问题query和答案answer的特征表示后进行余弦相似度计算，由于问题i除了与答案i相匹配以外，其余答案均为问题i的负样本，因此需要对每一个问题进行负采样。

​	在下面内容中，本文将结合实际代码进行叙述4个部分的内容：

- 模型实现代码
  - 1.输入层
  - 2.表示层
  - 3.匹配层
  - 4.梯度更新部分
- 模型调用方式
- 模型训练数据
- 已训练模型库

## 2.1 模型实现代码

模型实现代码位于Model/TransformerDSSM.py，其实现顺序从TransformerDSSM类开始。

首先，通过调用build_graph_by_cpu或者build_graph_by_gpu对模型整个数据流图进行构建，以上两种构建方式，分别对应着模型的cpu版本和gpu版本。在构建数据流图的过程中，需要依次去定义模型的如下部分：

（1）输入层

在输入层中，主要将输入的问题集和答案集转换成每个字符对应的字向量，最终形成三维矩阵q、t：

```python
with tf.name_scope('InputLayer'):
    # 定义词向量
    embeddings = tf.constant(self.vec_set)

    # 将句子中的每个字转换为字向量
    if not self.is_extract:
        q_embeddings = tf.nn.embedding_lookup(embeddings, self.q_inputs)
    if self.is_train:
        t_embeddings = tf.nn.embedding_lookup(embeddings, self.t_inputs)
```

（2）表示层

表示层的实现为函数presentation_transformer()。

在原有的Transformer中，对于其输入的三维矩阵来说，为了能够引入单词在句子中的位置信息，需要在原有单词语义向量的基础上，通过规则的方式加上每个单词在句子中的位置编码向量。在本模型中，输入数据直接通过一个双向GRU来对句子中每个字的上下文信息进行编码：

```python
with tf.name_scope('structure_presentation_layer'):
    # 正向
    fw_cell = GRUCell(num_units=self.hidden_num)
    fw_drop_cell = DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
    # 反向
    bw_cell = GRUCell(num_units=self.hidden_num)
    bw_drop_cell = DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出是一个元组 每一个元素也是这种形状
    if self.is_train and not self.is_extract:
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_drop_cell,                            cell_bw=bw_drop_cell, inputs=inputs,                                                sequence_length=inputs_actual_length,
                  dtype=tf.float32)
    else:
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, 						cell_bw=bw_cell, inputs=inputs,
                                                        						           sequence_length=inputs_actual_length, dtype=tf.float32)

    # hiddens的长度为2，其中每一个元素代表一个方向的隐藏状态序列，将每一时刻的输出合并成一个输出
    structure_output = tf.concat(output, axis=2)
    structure_output = self.layer_normalization(structure_output)
```

对输入数据进行上下文编码后，再将其带入到Transformer的Encoder部分，进行Self-Attention、AddNorm、Full-Connect计算，其实现类依次为SelfAttention、LayNormAdd、FeedForwardNetwork，这三个类通过类TransformerEncoder进行封装：

```python
for _ in range(params["num_layers"]):
    self_attention_layer = SelfAttention(params["hidden_size"], params["num_heads"],
                                         params["keep_prob"])
    feed_forward_network = FeedFowardNetwork(params["hidden_size"],                                                              params["keep_prob"])

    self.layers.append([LayNormAdd(self_attention_layer, params),
                        LayNormAdd(feed_forward_network, params)])

    self.output_normalization = LayerNormalization(params["hidden_size"])
```

在得到Transformer的输出以后，由于并没有得到每个句子的特征向量表示，需要在其基础上引入Global-Attention，对每个句子的最终特征向量进行计算：

```python
with tf.name_scope('global_attention_layer'):
    w_omega = tf.get_variable(name='w_omega', shape=[self.hidden_num * 2,                                         self.attention_num],
                              initializer=tf.random_normal_initializer())
    b_omega = tf.get_variable(name='b_omega', shape=[self.attention_num],
                              initializer=tf.random_normal_initializer())
    u_omega = tf.get_variable(name='u_omega', shape=[self.attention_num],
                              initializer=tf.random_normal_initializer())

    v = tf.tanh(tf.tensordot(transformer_output, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # tf.expand_dims用于在指定维度增加一维
    global_attention_output = tf.reduce_sum(transformer_output *                                                                 tf.expand_dims(alphas, -1), 1)
```

（3）匹配层

匹配层的实现函数为matching_layer_training和matching_layer_infer。这是由于模型在进行training时需要进行负采样，在infer时不需要，因此需要定义两个不同的余弦相似度计算函数。具体实现请去源码中查看。

（4）梯度更新部分

匹配层最终的输出是一个二维矩阵，矩阵中的每一行代表一个问题与其所对应答案（第一列）及负样本的余弦相似度值。对于这样一个矩阵，经过Softmax归一化后，截取第一列数据，采用交叉熵损失计算模型最终loss，最后使用Adam优化器对模型进行训练以及梯度更新。

```python
# softmax归一化并输出
prob = tf.nn.softmax(cos_sim)
with tf.name_scope('Loss'):
    # 取正样本
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    self.loss = -tf.reduce_sum(tf.log(hit_prob)) / self.batch_size

with tf.name_scope('Accuracy'):
    output_train = tf.argmax(prob, axis=1)
    self.accuracy = tf.reduce_sum(tf.cast(tf.equal(output_train,                                                       tf.zeros_like(output_train)),
                                  dtype=tf.float32)) / self.batch_size

# 优化并进行梯度修剪
with tf.name_scope('Train'):
      ptimizer = tf.train.AdamOptimizer(self.learning_rate)
      # 分解成梯度列表和变量列表
      grads, vars = zip(*optimizer.compute_gradients(self.loss))
      # 梯度修剪
      gradients, _ = tf.clip_by_global_norm(grads, 5)  # clip gradients
      # 将每个梯度以及对应变量打包
      self.train_op = optimizer.apply_gradients(zip(gradients, vars))
```

## 2.2 模型调用方式

​	模型的调用代码位于Debug.py，其调用方式主要分为以下3种。

（1）模型训练

TransformerDSSM模型的训练通过调用文件中的函数`dssm_model_train`实现，该函数以两个参数作为输入:

**faq_dict**，该参数是一个问答对组成的列表，列表中的每一个元素均为一个问答对字典；

**embedding_dict**，该参数是一个字典，字典中的每一个key是一个字符，value是该字符对应的字向量。字向量的提供位于目录：`WordEmbedding/CharactersEmbedding.json`

（2）模型推理

TransformerDSSM模型的推理通过调用文件中的函数`dssm_model_infer`实现，该函数以五个参数作为输入，需要注意的是，模型的**推理返回结果**，是输入答案的**位置索引**：

**queries**，该参数是一系列需要去匹配的问题组成的列表，列表中的每一个元素是一个问题字符串；

**answer_embedding**，该参数是由一系列待匹配的答案经过表示层所提取的特征向量组成的列表，列表中的每一个元素是一个答案对应的特征向量，之所以用特征向量直接作为待匹配答案的输入，是为了减少数据经过表示层的计算时间，提高匹配效率；

**embedding_dict**，该参数是一个字典，字典中的每一个key是一个字符，value是该字符对应的字向量。字向量的提供位于目录：`/NlpModel/WordEmbedding/Word2Vec/CharactersEmbedding.json`

**top_k**，该参数表示当输入一个问题时，需要从待匹配的答案中返回top_k个候选答案，默认时，该参数的值为1；

**threshold**，该参数通过设置语义相似度计算的阈值，当待匹配的答案其相似度低于给定阈值时，则不返回，高于则返回。

（3）表示层特征向量提取

TransformerDSSM模型的表示层特征向量提取通过调用文件中的函数`dssm_model_extract_t_pre`实现，该函数以两个参数作为输入:

**faq_dict**，该参数是一个问答对组成的列表，列表中的每一个元素均为一个问答对字典；

**embedding_dict**，该参数是一个字典，字典中的每一个key是一个字符，value是该字符对应的字向量。字向量的提供位于目录：`/NlpModel/WordEmbedding/Word2Vec/CharactersEmbedding.json`

# 3.模型训练数据

本模块提供的训练数据，是作为预训练模型的训练数据，主要分为以下3种，其中SameFAQ表示问题，答案指向同一句子，各问答对间的语义完全独立，可用于进行语义空间划分，SimFAQ中的问答对则是语义相近的，用于语义相似度训练；LCQMC为Q-Q格式的数据集，可用于问题语义相似度训练。该训练数据位于目录：`TransformerDSSM/TrainData/`：

# 4.已训练模型库

本模块提供一种已经过训练的模型，新的问答对数据或者问题对数据可在这个预训练模型的基础上进行训练，能够达到较好效果。（将下面的模型model-Sim下载后移动到TransformerDSSM/ModelMemory目录下,然后运行debug.py即可）

下载链接：https://pan.baidu.com/s/1WYCHhDaHFHIg0wZDYH-YrA 
提取码：b8w2 



如果你对该模型感兴趣，可去[github地址]()给个star

