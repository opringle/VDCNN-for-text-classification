#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-

import pandas as pd
import mxnet as mx
import numpy as np
from itertools import chain
import argparse
import logging
import os
import ast
import copy
from random import randint


logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Neural Collaborative Filtering Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', nargs='?', default='./data/atb_model_41/',
                        help='Input data folder')
parser.add_argument('--output-dir', type=str, default='checkpoint',
                    help='directory to save model params/symbol to')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')

parser.add_argument('--num-epochs', type=int, default=256,
                    help='how  many times to update the model parameters')
parser.add_argument('--batch-size', type=int, default=512,
                    help='the number of training records in each minibatch')
parser.add_argument('--sequence-length', type=int, default=1024,
                    help='the number of characters in each training example')

parser.add_argument('--fc-size', type=int, default=2048,
                    help='the number of hidden units in each fully connected layer')

parser.add_argument('--blocks', type=str, default='1,1,1,1',
                    help='Number of conv blocks in each component of the network')
parser.add_argument('--channels', type=str, default='6,6,6,8',
                    help='Number of channels in each conv block')

parser.add_argument('--optimizer', type=str, default='SGD',
                    help='optimization algorithm to update model parameters with')
parser.add_argument('--lr', type=float, default=1.0,
                    help='learning rate for chosen optimizer')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='momentum for chosen optimizer')
parser.add_argument('--lr-reduce-factor', type=float, default=0.5,
                    help='multiply learning rate by this number when modifying')
parser.add_argument('--lr-reduce-epoch', type=int, default=10000,
                    help='modify learning rate every n epochs')
parser.add_argument('--l2', type=float, default=0.0,
                    help='l2 regularization coefficient')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout rate for penultimate layer')



class k_max_pool(mx.operator.CustomOp):
  def __init__(self, k):
    super(k_max_pool, self).__init__()
    self.k = int(k)
  def forward(self, is_train, req, in_data, out_data, aux):
    x = in_data[0].asnumpy()
    assert(4 == len(x.shape))
    ind = np.argsort(x, axis = 2)
    sorted_ind = np.sort(ind[:,:,-(self.k):,:], axis = 2)
    dim0, dim1, dim2, dim3 = sorted_ind.shape
    self.indices_dim0 = np.arange(dim0).repeat(dim1 * dim2 * dim3)
    self.indices_dim1 = np.transpose(np.arange(dim1).repeat(dim2 * dim3).reshape((dim1*dim2*dim3, 1)).repeat(dim0, axis=1)).flatten()
    self.indices_dim2 = sorted_ind.flatten()
    self.indices_dim3 = np.transpose(np.arange(dim3).repeat(dim2).reshape((dim3, dim2)).repeat(dim0 * dim1, axis = 1)).flatten()
    y = x[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3].reshape(sorted_ind.shape)
    self.assign(out_data[0], req[0], mx.nd.array(y))

  def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
    x = out_grad[0].asnumpy()
    y = in_data[0].asnumpy()
    assert(4 == len(x.shape))
    assert(4 == len(y.shape))
    y[:,:,:,:] = 0
    y[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3] \
      = x.reshape([x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3],])
    self.assign(in_grad[0], req[0], mx.nd.array(y))


@mx.operator.register("k_max_pool")
class k_max_poolProp(mx.operator.CustomOpProp):
  def __init__(self, k):
    self.k = int(k)
    super(k_max_poolProp, self).__init__(True)
  def list_argument(self):
    return ['data']
  def list_outputs(self):
    return ['output']
  def infer_shape(self, in_shape):
    data_shape = in_shape[0]
    assert(len(data_shape) == 4)
    out_shape = (data_shape[0], data_shape[1], self.k, data_shape[3])
    return [data_shape], [out_shape]

  def create_operator(self, ctx, shapes, dtypes):
    return k_max_pool(self.k)


class UtterancePreprocessor:
    """
    preprocessor that can be fit to data in order to preprocess it
    """
    def __init__(self, length, char_to_index=None, pad_char='👹', unknown_char='👺', space_char='🎃'):
        self.length = length
        self.pad_char = pad_char
        self.unknown_char = unknown_char
        self.space_char = space_char
        self.padded_data = 0
        self.sliced_data = 0
        self.char_to_index = char_to_index

    @staticmethod
    def build_vocab(data, depth):
        """
        :param data: list of data (can be nested)
        :param depth: how many levels the list is nested
        :return: dictionary where key is string and value is index
        """
        if depth > 1:
            data = list(chain.from_iterable(data))
        return {k: v for v, k in enumerate(set(data))}

    def pad_utterance(self, utterance):
        """
        :param utterance: list of int
        :param length: desired output length
        :param pad_value: integer to use for padding
        :return: list of int
        """
        diff = self.length - len(utterance)
        if diff > 0:
            self.padded_data += 1
            utterance.extend([self.pad_char] * diff)
            return utterance
        else:
            self.sliced_data += 1
            return utterance[:self.length]

    def fit(self, utterances, labels):
        """
        :param utterances: list of string
        :param labels: list of string
        """
        chars = [list(utterance.lower()) for utterance in utterances]
        self.char_to_index = self.build_vocab(chars, depth=2) if not self.char_to_index else self.char_to_index
        self.label_to_index = self.build_vocab(labels, depth=1)

        if self.pad_char not in self.char_to_index:
            self.char_to_index[self.pad_char] = len(self.char_to_index)

        if self.unknown_char not in self.char_to_index:
            self.char_to_index[self.unknown_char] = len(self.char_to_index)

        if self.space_char not in self.char_to_index:
            self.char_to_index[self.space_char] = len(self.char_to_index)
        print("Space tokens represented as {}\n"
              "Unknown tokens represented as {}\n"
              "Padded tokens represented as {}".format(self.space_char, self.unknown_char, self.pad_char))

    def transform_utterance(self, utterance):
        """
        :param utterance:
        :return: split and indexed utterance
        """
        split_utterance = list(utterance.lower())
        padded_utterance = self.pad_utterance(split_utterance)
        resolved_utterance = [self.space_char if char == ' ' else char for char in padded_utterance]
        resolved_utterance = [self.unknown_char if char not in self.char_to_index else char for char in resolved_utterance]
        indexed_utterance = [self.char_to_index.get(char) for char in resolved_utterance]
        return indexed_utterance

    def transform_label(self, label):
        """
        :param label: string
        :return: int
        """
        return self.label_to_index.get(label)


def build_iters(train_df, test_df, feature_col, label_col, alphabet):
    """
    :param train_df: pandas dataframe of training data
    :param test_df: pandas dataframe of test data
    :param feature_col: column in dataframe corresponding to text
    :param label_col: column in dataframe corresponding to label
    :return: mxnet data iterators
    """
    # Fit preprocessor to training data
    preprocessor = UtterancePreprocessor(length=args.sequence_length, char_to_index=alphabet)
    preprocessor.fit(train_df[feature_col].values.tolist(), train_df[label_col].values.tolist())


    # Transform data
    train_df['X'] = train_df[feature_col].apply(preprocessor.transform_utterance)
    test_df['X'] = test_df[feature_col].apply(preprocessor.transform_utterance)
    train_df['Y'] = train_df[label_col].apply(preprocessor.transform_label)
    test_df['Y'] = test_df[label_col].apply(preprocessor.transform_label)
    print("{} utterances were padded & {} utterances were sliced to length = {}".format(preprocessor.padded_data,
                                                                                        preprocessor.sliced_data,
                                                                                        preprocessor.length))

    print("vocabulary used in lookup table: {}".format(preprocessor.char_to_index))

    # Get data as numpy array
    X_train, X_test = np.array(train_df['X'].values.tolist()), np.array(test_df['X'].values.tolist())
    Y_train, Y_test = np.array(train_df['Y'].values.tolist()), np.array(test_df['Y'].values.tolist())

    # Build MXNet data iterators
    train_iter = mx.io.NDArrayIter(data=X_train, label=Y_train, batch_size=args.batch_size, shuffle=True,
                                   last_batch_handle='pad')
    test_iter = mx.io.NDArrayIter(data=X_test, label=Y_test, batch_size=args.batch_size, shuffle=True,
                                  last_batch_handle='pad')
    return preprocessor, train_iter, test_iter


def build_symbol(iterator, preprocessor, blocks, channels):
    """
    :return:  MXNet symbol object
    """
    def conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True,
                                  name='%s%s_conv2d' % (name, suffix))
        bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
        act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' % (name, suffix))
        return act

    def conv_block(data, num_filter, name):
        conv1 = conv(data, kernel=(1, 3), num_filter=num_filter, pad=(0, 1), name='conv1'+str(name))
        conv2 = conv(conv1, kernel=(1, 3), num_filter=num_filter, pad=(0, 1), name='conv2'+str(name))
        return conv2

    X_shape, Y_shape = iterator.provide_data[0][1], iterator.provide_label[0][1]

    data = mx.sym.Variable(name="data")
    softmax_label = mx.sym.Variable(name="softmax_label")
    print("data_input: ", data.infer_shape(data=X_shape)[1][0])
    print("label input: ", softmax_label.infer_shape(softmax_label=Y_shape)[1][0])

    # Embed each character to 16 channels
    embedded_data = mx.sym.Embedding(data, input_dim=len(preprocessor.char_to_index), output_dim=16)
    embedded_data = mx.sym.Reshape(mx.sym.transpose(embedded_data, axes=(0, 2, 1)), shape=(0, 0, 1, -1))
    print("embedded output: ", embedded_data.infer_shape(data=X_shape)[1][0])

    # Temporal Convolutional Layer (without activation)
    temp_conv_1 = mx.sym.Convolution(embedded_data, kernel=(1, 3), num_filter=64, pad=(0, 1))
    print("temp conv output: ", temp_conv_1.infer_shape(data=X_shape)[1][0])

    # Create convolutional blocks with pooling in-between
    for i, block_size in enumerate(blocks):
        print("section {} ({} blocks)".format(i, block_size))
        for j in list(range(block_size)):
            if i == 0 and j == 0:
                # first block follows the first temp conv layer
                block = conv_block(temp_conv_1, num_filter=channels[i], name='block'+str(i)+'_'+str(j))
            elif j == 0:
                # this block follows a pooling layer
                block = conv_block(pool, num_filter=channels[i], name='block' + str(i) + '_' + str(j))
            else:
                # this block follows a previous block
                block = conv_block(block, num_filter=channels[i], name='block'+str(i)+'_'+str(j))
            print('\tblock'+str(i)+'_'+str(j), block.infer_shape(data=X_shape)[1][0])
        if i != len(blocks)-1:
            # pool after each block size, excluding final layer
            pool = mx.sym.Pooling(block, kernel=(1, 3), stride=(1, 2), pad=(0, 1), pool_type='max')
            print('\tblock' + str(i) + '_p', pool.infer_shape(data=X_shape)[1][0])


    #block = mx.sym.transpose(mx.symbol.Custom(data=mx.sym.transpose(block, axes=(0, 1, 3, 2)), name='8_max_pool', op_type='k_max_pool', k=8), axes=(0, 1, 3, 2))
    n = int(args.sequence_length / (2**(len(args.blocks)-1)))
    block = mx.sym.flatten(mx.sym.Pooling(block, kernel=(1, n), stride=(1, 1), pad=(0, 0), pool_type='avg'))
    print("average pool kernel size {0}, stride 1. output: {1}".format(n, block.infer_shape(data=X_shape)[1][0]))
    block = mx.sym.Dropout(block, p=args.dropout)
    print("flattened dropout output: ", block.infer_shape(data=X_shape)[1][0])

    output = mx.sym.FullyConnected(block, num_hidden=len(preprocessor.label_to_index), flatten=True, name='output')
    sm = mx.sym.SoftmaxOutput(output, softmax_label)
    print("softmax output: ", sm.infer_shape(data=X_shape)[1][0])

    return sm


def train(symbol, train_iter, val_iter):
    """
    :param symbol: model symbol graph
    :param train_iter: data iterator for training data
    :param valid_iter: data iterator for validation data
    :return: model to predict label from features
    """
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    module = mx.mod.Module(symbol, context=devs)
    schedule = mx.lr_scheduler.FactorScheduler(step=int((train_df.shape[0] / args.batch_size) * args.lr_reduce_epoch),
                                                                                 factor=args.lr_reduce_factor,
                                                                                 stop_factor_lr=1e-04)
    module.fit(num_epoch=args.num_epochs,
               train_data=train_iter,
               eval_data=val_iter,
               optimizer=args.optimizer,
               eval_metric=mx.metric.Accuracy(),
               optimizer_params={'learning_rate': args.lr, 'wd': args.l2, 'momentum': args.momentum,
                                 'lr_scheduler': schedule},
               initializer=mx.initializer.Mixed(patterns=['conv2d_weight', '.*'],
                                                initializers=[mx.initializer.MSRAPrelu(factor_type='avg', slope=0.25),
                                                              mx.initializer.Normal(sigma=0.02)]),
               epoch_end_callback=mx.callback.do_checkpoint(prefix=os.path.join(args.output_dir, "checkpoint"), period=1))
    return module


def summarize_data(df, name):
    """
    computes statistics on text classification dataframes
    """
    df['n_chars'] = df['utterance'].apply(lambda x: len(x))
    print("\n{} summary:".format(name))
    print("\tAverage characters per utterance: {0:.2f}".format(df['n_chars'].mean()))
    print("\tStandard deviation characters per utterance: {0:.2f}".format(df['n_chars'].std()))
    print("\tTotal utterances in df: {}".format(df.shape[0]))
    print("\tText Classes: {}".format(len(df['intent'].unique())))


if __name__ == '__main__':

    # os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    # Parse args
    args = parser.parse_args()
    args.blocks = ast.literal_eval(args.blocks)
    args.channels = ast.literal_eval(args.channels)

    # Setup dirs
    os.mkdir(args.output_dir) if not os.path.exists(args.output_dir) else None

    # Read training data into pandas data frames
    train_df = pd.read_pickle(os.path.join(args.data, "train.pickle"))
    test_df = pd.read_pickle(os.path.join(args.data, "test.pickle"))
    summarize_data(train_df, 'Training data')
    summarize_data(test_df, 'Test data')

    # Define vocab (if unknown characters are encountered they are replaced with final value in alphabet)
    alph = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{}'
    char_to_index = {k: v for v, k in enumerate(list(alph))}

    # Build data iterators
    preprocessor, train_iter, val_iter = build_iters(train_df, test_df, feature_col='utterance', label_col='intent',
                                                     alphabet=char_to_index)

    # Build network graph
    symbol = build_symbol(train_iter, preprocessor, blocks=args.blocks, channels=args.channels)

    # Train the model
    trained_module = train(symbol, train_iter, val_iter)