#coding:utf-8
from collections import OrderedDict
import pickle as pkl
import sys
sys.path.extend('..')
import time
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import textprocessing
import pdb
from gof.opt import optimizer
from dask.array.tests.test_array_core import test_size
from sklearn.cluster.tests.test_k_means import n_samples

datasets = {'my_data': (textprocessing.load_data, textprocessing.prepare_data)}

#为持久化设计升恒器的种子树
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=True):
    #打乱数据并按照minibatch拿到数据
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def get_dataset(name):
    #获得训练、校验、测试集的数据，准备集的数据
    return datasets[name][0], datasets[name][1]

def zipp(params,tparams):
    








