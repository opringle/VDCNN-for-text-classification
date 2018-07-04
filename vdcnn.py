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
from random import randint


logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Neural Collaborative Filtering Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', nargs='?', default='./data',
                        help='Input data folder')
parser.add_argument('--output-dir', type=str, default='checkpoint',
                    help='directory to save model params/symbol to')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')

parser.add_argument('--num-epochs', type=int, default=256,
                    help='how  many times to update the model parameters')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the number of training records in each minibatch')
parser.add_argument('--sequence-length', type=int, default=299,
                    help='the number of characters in each training example')

# Optimizer
parser.add_argument('--optimizer', type=str, default='RMSProp',
                    help='optimization algorithm to update model parameters with')
parser.add_argument('--decay', type=float, default=0.9,
                    help='decay factor for moving average')
parser.add_argument('--epsilon', type=float, default=1.0,
                    help='avoids division by 0')
parser.add_argument('--lr', type=float, default=0.045,
                    help='learning rate for chosen optimizer')
parser.add_argument('--grad-clip', type=float, default=2.0,
                    help='Clips weights to +- this value')

# Regularization
parser.add_argument('--l2', type=float, default=0.0,
                    help='l2 regularization coefficient')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout regularization probability for penultimate layer')
parser.add_argument('--smooth-alpha', type=float, default=0.0,
                    help='label smoothing coefficient')

# Architecture
parser.add_argument('--blocks', type=str, default='3,5,2',
                    help='Number of conv blocks in each component of the network')
parser.add_argument('--channels', type=str, default='384,640,2048',
                    help='Number of channels in each conv block')


class UtterancePreprocessor:
    """
    preprocessor that can be fit to data in order to preprocess it
    """
    def __init__(self, length, unknown_char_index, char_to_index=None, pad_char='~'):
        self.length = length
        self.pad_char = pad_char
        self.unknown_char_index = unknown_char_index #should this be a random char from the dict?
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
            print("Warning, padded data char: `{}` is not in vocabulary. Adding now.".format(self.pad_char))
            self.char_to_index[self.pad_char] = len(self.char_to_index)

    def transform_utterance(self, utterance):
        """
        :param utterance:
        :return: split and indexed utterance
        """
        split_utterance = list(utterance.lower())
        padded_utterance = self.pad_utterance(split_utterance)
        # indexed_utterance = [self.char_to_index.get(char, randint(0,len(self.char_to_index))) for char in padded_utterance]
        indexed_utterance = [self.char_to_index.get(char, self.unknown_char_index) for char in padded_utterance]
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
    preprocessor = UtterancePreprocessor(length=args.sequence_length, unknown_char_index=len(alphabet),
                                         char_to_index=alphabet)
    preprocessor.fit(train_df[feature_col].values.tolist(), train_df[label_col].values.tolist())

    print("index of unknown characters = {}\n"
          "padded characters represented as = {}\n"
          "index of pad char in dict: {}".format(preprocessor.unknown_char_index, preprocessor.pad_char,
                                                 alphabet[preprocessor.pad_char]))


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

    def inception_block_5(data, filters, name, reduce_grid=False):
        """
        inception module with dimensionality reduction
        """
        x = int(filters/8)

        conv_1 = conv(data, num_filter=2 * x, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='1X1_conv' + str(name))
        # print("\t1X1 conv output: ", conv_1.infer_shape(data=X_shape)[1][0])

        reduce_3 = conv(data, num_filter=3*x, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='1X3_reduce' + str(name))
        if reduce_grid:
            conv_3 = conv(reduce_3, num_filter=5*x, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='1X3_conv' + str(name))
        else:
            conv_3 = conv(reduce_3, num_filter=4*x, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='1X3_conv' + str(name))
        # print("\t1X3 conv output: ", conv_3.infer_shape(data=X_shape)[1][0])

        reduce_5 = conv(data, num_filter=int(x/2), kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='1X5_reduce' + str(name))
        conv_5_0 = conv(reduce_5, num_filter=x, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='1X5_conv1' + str(name))
        if reduce_grid:
            conv_5 = conv(conv_5_0, num_filter=3*x, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='1X5_conv2' + str(name))
        else:
            conv_5 = conv(conv_5_0, num_filter=x, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='1X5_conv2' + str(name))
        # print("\t1X5 conv output: ", conv_5.infer_shape(data=X_shape)[1][0])

        if reduce_grid:
            pool = mx.sym.Pooling(data, kernel=(1, 3), stride=(1, 2), pad=(0, 0), pool_type='max', name='1X3_pool')
            conv_pool = conv(pool, num_filter=filters, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='1X1_pool_conv' + str(name))
        else:
            pool = mx.sym.Pooling(data, kernel=(1, 3), stride=(1, 1), pad=(0, 1), pool_type='max', name='1X3_pool')
            conv_pool = conv(pool, num_filter=x, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='1X1_pool_conv' + str(name))
        # print("\t1X1 pool output: ", conv_pool.infer_shape(data=X_shape)[1][0])

        # concatenate channels
        if reduce_grid:
            concat = mx.sym.Concat(*[conv_3, conv_5, conv_pool], dim=1, name=str(name))
        else:
            concat = mx.sym.Concat(*[conv_1, conv_3, conv_5, conv_pool], dim=1, name=str(name))
        # print("\tdepth concat output: ", concat.infer_shape(data=X_shape)[1][0])
        return concat

    X_shape, Y_shape = iterator.provide_data[0][1], iterator.provide_label[0][1]

    data = mx.sym.Variable(name="data")
    softmax_label = mx.sym.Variable(name="softmax_label")
    print("data_input: ", data.infer_shape(data=X_shape)[1][0])
    print("label input: ", softmax_label.infer_shape(softmax_label=Y_shape)[1][0])

    # Embed each character to 16 channels
    embedded_data = mx.sym.Embedding(data, input_dim=len(preprocessor.char_to_index), output_dim=16)
    embedded_data = mx.sym.Reshape(mx.sym.transpose(embedded_data, axes=(0, 2, 1)), shape=(0, 0, 1, -1))
    print("embedded output: ", embedded_data.infer_shape(data=X_shape)[1][0])

    # Initial conv layers
    conv1 = conv(embedded_data, num_filter=32, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv1')
    print("conv1 output: ", conv1.infer_shape(data=X_shape)[1][0])
    conv2 = conv(conv1, num_filter=32, kernel=(1, 3), stride=(1, 1), pad=(0, 0), name='conv2')
    print("conv2 output: ", conv2.infer_shape(data=X_shape)[1][0])
    conv_3 = conv(conv2, num_filter=64, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv3')
    print("conv3 output: ", conv_3.infer_shape(data=X_shape)[1][0])
    pool_1 = mx.sym.Pooling(conv_3, kernel=(1, 3), stride=(1, 2), pad=(0, 0), pool_type='max', name='pool')
    print("poool1 output: ", pool_1.infer_shape(data=X_shape)[1][0])
    conv_4 = conv(pool_1, num_filter=80, kernel=(1, 3), stride=(1, 1), pad=(0, 0), name='conv4')
    print("conv4 output: ", conv_4.infer_shape(data=X_shape)[1][0])
    conv_5 = conv(conv_4, num_filter=192, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv5')
    print("conv5 output: ", conv_5.infer_shape(data=X_shape)[1][0])
    conv_6 = conv(conv_5, num_filter=288, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv6')
    print("conv6 output: ", conv_6.infer_shape(data=X_shape)[1][0])

    # Inception blocks
    for i, (blocks, channels) in enumerate(zip(args.blocks, args.channels)):
        for block in list(range(blocks)):
            if i == 0 and block == 0:
                inception = inception_block_5(conv_6,
                                          filters=channels,
                                          name='inception_block_5_' + str(i) + str(block) + str(channels))
            elif block == len(range(blocks))-1 and i != len(args.blocks)-1:
                inception = inception_block_5(inception,
                                              filters=channels,
                                              name='inception_block_5_' + str(i) + str(block) + str(channels),
                                              reduce_grid=True)
            else:
                inception = inception_block_5(inception,
                                              filters=channels,
                                              name='inception_block_5_' + str(i) + str(block) + str(channels))

            print("Block {} inception module {} output shape: {}".format(i+1, block+1, inception.infer_shape(data=X_shape)[1][0]))


    avg_pool = mx.sym.Pooling(inception, kernel=(1, 8), stride=(1, 1), pad=(0, 0), pool_type='avg')
    print("average pool output: ", avg_pool.infer_shape(data=X_shape)[1][0])

    final_dropout = mx.sym.Dropout(mx.sym.flatten(avg_pool), p=args.dropout)

    output = mx.sym.FullyConnected(final_dropout, num_hidden=len(preprocessor.label_to_index), flatten=True, name='output')
    sm = mx.sym.SoftmaxOutput(output, softmax_label, args.smooth_alpha)
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
    module.fit(train_data=train_iter,
               eval_data=val_iter,
               optimizer=args.optimizer,
               eval_metric=mx.metric.Accuracy(),
               optimizer_params={'learning_rate': args.lr, 'wd': args.l2, 'gamma1': args.decay, 'epsilon': args.epsilon,
                                 'clip_weights': args.grad_clip},
               initializer=mx.initializer.Normal(),
               num_epoch=args.num_epochs)
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
    alph = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} ~'
    char_to_index = {k: v for v, k in enumerate(list(alph))}

    # Build data iterators
    preprocessor, train_iter, val_iter = build_iters(train_df, test_df, feature_col='utterance', label_col='intent',
                                                     alphabet=char_to_index)

    # Build network graph
    symbol = build_symbol(train_iter, preprocessor, blocks=args.blocks, channels=args.channels)

    # Train the model
    trained_module = train(symbol, train_iter, val_iter)