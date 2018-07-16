'''
word2vec是从大量文本预料中以无监督的方式学习语义知识的一种模型，他被大量的用在自然语言中
。那么他是如何帮助我们做自然语言的呢？word2vec其实就是通过学习文本用词向量的方式来表征词的语义信息
即通过一个嵌入空间使得语义上相似的单词在该空间内距离很近。Embedding其实就是一个映射，将单词从原先所属的空间映射到新的多维空间中，也就是把原先词所在
空间嵌入到一个新的空间中去。

我们从直观角度上来理解一下，cat这个单词和kitten属于语义上很相近的词，而dog和kitten则不是那么相近，iphone这个单词和kitten的语义就差的更远了。通过对词汇表中单词进行这种数值表示方式的学习（也就是将单词转换为词向量），能够让我们基于这样的数值进行向量化的操作从而得到一些有趣的结论。比如说，如果我们对词向量kitten、cat以及dog执行这样的操作：kitten - cat + dog，那么最终得到的嵌入向量（embedded vector）将与puppy这个词向量十分相近。


第一部分：模型
word2vec模型中，主要有skip_gram和VBOW两种模型，从直观上理解，skip——gram是给定input word来预测上下文。而CBOw是给定上下文，来预测input word

word2vec模型实际上分为两个部分，第一部分为建立模型，第二部分是通过模型获取嵌入词向量。
word2vec的整个建模过程现基于训练数据构建一个神经网络，当这个模型训练好以后，我们并不用这个训练好的模型处理新的任务，
我们真正需要的是这个模型通过训练数据所学的的参数，例如隐层的权重矩阵---后面我们将会看到这些权重在word2vec中实际上就是我们试图去学习的“word vectors".基于
训练数据建模的过程，我们给他一个名字叫“Fake Task”，意味着建模并不是我们最终的目的。



The Fake Task
我们在上面提到，训练模型的真正目的是获得模型基于训练数据得到隐层权重。为了得到这些权重，我们首先要构架你一个完整的神经网络作为我们的“Fake Task
后面再返回来看通过“Fake Task”我们如何间接的得到这些词向量

接下来我们来看看如何训练我们的神经网络，假如我们有一个句子“The dog barked at the mailcom
首先我们选句子中间的一个词作为我们的输入词，例如我们选取“dog”作为input word；
有了input word以后，我们再定义一个叫做skip_window的参数，它代表着我们从当前input word的一侧（左边或右边）选取词的数量。如果我们设置skip\_window=2，那么我们最终获得窗口中的词（包括input word在内）就是['The', 'dog'，'barked', 'at']。skip\_window=2代表着选取左input word左侧2个词和右侧2个词进入我们的窗口，所以整个窗口大小span=2\times 2=4。另一个参数叫num_skips，它代表着我们从整个窗口中选取多少个不同的词作为我们的output word，当skip\_window=2，num\_skips=2时，我们将会得到两组 (input word, output word) 形式的训练数据，即 ('dog', 'barked')，('dog', 'the')。
神经网络基于这些训练数据将会输出一个概率分布，这个概率代表着我们的词典中的每个词是output word的可能性。这句话有点绕，我们来看个栗子。第二步中我们在设置skip_window和num_skips=2的情况下获得了两组训练数据。假如我们先拿一组数据 ('dog', 'barked') 来训练神经网络，那么模型通过学习这个训练样本，会告诉我们词汇表中每个单词是“barked”的概率大小。
模型的输出概率代表着到我们词典中每个词有多大可能性跟input word同时出现。举个栗子，如果我们向神经网络模型中输入一个单词“Soviet“，那么最终模型的输出概率中，像“Union”， ”Russia“这种相关词的概率将远高于像”watermelon“，”kangaroo“非相关词的概率。因为”Union“，”Russia“在文本中更大可能在”Soviet“的窗口中出现。
我们将通过给神经网络输入文本中成对的单词来训练它完成上面所说的概率计算。下面的图中给出了一些我们的训练样本的例子。我们选定句子“The quick brown fox jumps over lazy dog”，设定我们的窗口大小为2（window\_size=2），也就是说我们仅选输入词前后各两个词和输入词进行组合。下图中，蓝色代表input word，方框内代表位于窗口内的单词。

'''
import tensorflow as tf
import time
import random
from collections import Counter

with open('G:/NLP/zhihu-master/skip_gram/data/Javasplittedwords',encoding="utf-8") as f:
    words = f.read()


#数据预处理
'''
替换文本特殊符号并去除低频词
对文本分词
构建预料
单词映射表
'''

#帅选低频词
words_count = Counter(words)
words = [w for w in words if words_count[w]>50]

#构建映射表
vocab = set(words)
vocab_to_int = {w:c for c,w in enumerate(vocab)}
int_to_vocab = {c:w for c,w in enumerate(vocab)}

print("total words: {}".format(len(words)))
print("unique words: {}".format(len(set(words))))

#对原文本进行vocab到int的转换
int_words = [vocab_to_int[w] for w in words]

import numpy as np
'''对停用词进行采样，例如“the”， “of”以及“for”这类单词进行剔除。剔除这些单词以后能够加快我们的训练过程，同时减少训练过程中的噪音。'''
t = 1e-5 # t值
threshold = 0.9 # 剔除概率阈值

# 统计单词出现频次
int_word_counts = Counter(int_words)
total_count = len(int_words)
# 计算单词频率
word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
# 计算被删除的概率
prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
# 对单词进行采样
train_words = [w for w in int_words if prob_drop[w] < threshold]


'''
skip-Gram模型是通过输入词来预测上下文，因此我们要构造我们的训练样本

对于一个给定词，离他越近的词可能与他越相关，离他越远的词越不相关，这里我们设置窗口大小为5
对于每个训练单词，我们还会在[1:5]之间随机生成一个整数R，用R作为我们最终选择output word的窗口大小。这里之所以多加了一个随机数的窗口
重新选择步骤，是为了能能够让模型更聚焦于当前input word的邻近词
'''


def get_targets(words, idx, window_size=5):
    '''
    获得input word的上下文单词列表

    参数
    ---
    words: 单词列表
    idx: input word的索引号
    window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size + 1)
    # 这里要考虑input word前面单词不够的情况
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    # output words(即窗口中的上下文单词)
    targets = set(words[start_point: idx] + words[idx + 1: end_point + 1])
    return list(targets)


def get_batches(words, batch_size, window_size=5):
    '''
    构造一个获取batch的生成器
    '''
    n_batches = len(words) // batch_size

    # 仅取full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            # 由于一个input word会对应多个output word，因此需要长度统一
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y

#构建网络
'''
该部分主要包括：
输入层
Embedding
negative Sampling
'''

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

'''
Embedding
嵌入矩阵的矩阵形状为vocab_size*hidden_units_size
'''
vocab_size = len(int_to_vocab)
embedding_size = 200 #嵌入维度

with train_graph.as_default():
    # 嵌入层权重矩阵
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    # 实现lookup
    embed = tf.nn.embedding_lookup(embedding, inputs)

'''
Negative Sampling

负采样主要是为了解决梯度下降计算速度慢的问题，详情同样参考我的上一篇知乎专栏文章。
'''

n_sampled = 100

with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(vocab_size))

    # 计算negative sampling下的损失
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with train_graph.as_default():
    # 随机挑选一些单词
    ## From Thushan Ganegedara's implementation
    valid_size = 7  # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size // 2))
    valid_examples = [vocab_to_int['word'],
                      vocab_to_int['ppt'],
                      vocab_to_int['熟悉'],
                      vocab_to_int['java'],
                      vocab_to_int['能力'],
                      vocab_to_int['逻辑思维'],
                      vocab_to_int['了解']]

    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

epochs = 10  # 迭代轮数
batch_size = 1000  # batch大小
window_size = 10  # 窗口大小

with train_graph.as_default():
    saver = tf.train.Saver()  # 文件存储

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        #
        for x, y in batches:

            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            # 计算相似的词
            if iteration % 1000 == 0:
                # 计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8  # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1

    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)


