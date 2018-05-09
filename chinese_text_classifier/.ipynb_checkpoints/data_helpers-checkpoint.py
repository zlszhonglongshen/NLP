#
import numpy as np
import re
import itertools
import codecs
from collections import Counter
import jieba

def clean_str(string):
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()


def load_data_and_labels(pos=None,neg=None):
    #load data from files
    path = "D:\python_spark\python_NLP\Chinese_Text_Classification\Chinese-Text-Classification-master"
    positive_examples = list(codecs.open(path+"/data/chinese/pos.txt", "r", "utf-8").readlines())
    positive_examples = [[item for item  in jieba.cut(s,cut_all=False)] for s in positive_examples]
    negative_examples = list(codecs.open(path+"/data/chinese/neg.txt","r","utf-8").readlines())
    negative_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in negative_examples]
    #split by words
    x_test = positive_examples+negative_examples

    #generate labels
    positive_labels = [[0,1] for _ in positive_examples]
    negative_labels = [[1,0] for _ in negative_examples]
    y = np.concatenate([positive_labels,negative_labels])
    return [x_test,y]

def load_test_data_and_labels(pos=None,neg=None):
  """
  Loads MR polarity data from files, splits the data into words and generates labels.
  Returns split sentences and labels.
  tf.flags.DEFINE_string("positive_data_file", "./data/test_text/pos.txt", "Data source for the positive data.")
  tf.flags.DEFINE_string("negative_data_file", "./data/test_text/neg.txt", "Data source for the negative data.")
  """
  # Load data from files
  path = "D:\python_spark\python_NLP\Chinese_Text_Classification\Chinese-Text-Classification-master"
  positive_examples = list(codecs.open(path+"/data/test_text/pos.txt", "r", "utf-8").readlines())
  positive_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in positive_examples]
  negative_examples = list(codecs.open(path+"/data/test_text/neg.txt", "r", "utf-8").readlines())
  negative_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in negative_examples]
  # Split by words
  x_text = positive_examples + negative_examples

  # Generate labels
  positive_labels = [[0, 1] for _ in positive_examples]
  negative_labels = [[1, 0] for _ in negative_examples]
  y = np.concatenate([positive_labels, negative_labels], 0)
  return [x_text, y]


def pad_sentences(sentences,padding_word="<PAD/>"):
    """
    pads all sentences to the same length,the length is defined by the loggest sentences
    :param sentences:
    :param padding_word:
    :return:
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length-len(sentence)
        new_sentence = sentence+[padding_word]*num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    build a vocabulary mapping from word to index based on the sentences
    returns vocabulary mapping and inverse vocabulary mapping
    :param sentences:
    :return:
    """
    #build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    #mapping from index to word
    vobulary_inv = [x[0] for x in word_counts.most_common()]
    #mapping from word to index
    vocabulary = {x:i for i,x in enumerate(vobulary_inv)}
    return {vocabulary,vobulary_inv}


def build_input_data(sentences,labels,vocabulary):
    """
    maps sentences and labels to vectors based on a vocabulary
    :param sentences:
    :param labels:
    :param vocabulary:
    :return:
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x,y]

def load_data():
    # Load and preprocess data for the MR dataset
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data,batch_size,num_epoches):
    """
    Generates a batch iterator fro a dataset
    :param data:
    :param batch_size:
    :param num_epoches:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)+1
    for epoch in range(num_epoches):
        #shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


