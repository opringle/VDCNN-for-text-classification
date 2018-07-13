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
import ast

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Neural Collaborative Filtering Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train-path', type=str, default='./data/ag_news/train.pickle',
                        help='path to pickled pandas train df')
parser.add_argument('--test-path', type=str, default='./data/ag_news/train.pickle',
                        help='path to pickled pandas test df')
parser.add_argument('--gpus', type=int, default=None,
                    help='list of gpus to run, e.g. 0 or 0,2,5. negate to use cpu.')

parser.add_argument('--epochs', type=int, default=256,
                    help='how  many times to update the model parameters')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the number of training records in each minibatch')
parser.add_argument('--sequence-length', type=int, default=299,
                    help='the number of characters in each training example')

# Optimizer
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimization algorithm to update model parameters with')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate for chosen optimizer')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for optimizer')
parser.add_argument('--lr-update-factor', type=float, default=0.97,
                    help='factor to reduce lr by')
parser.add_argument('--lr-update-interval', type=int, default=30000,
                    help='reduce lr every n epochs')
parser.add_argument('--grad-clip', type=float, default=0.0,
                    help='Clips weights to +- this value')

# Regularization
parser.add_argument('--l2', type=float, default=0.0,
                    help='l2 regularization coefficient')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout regularization probability for penultimate layer')
parser.add_argument('--smooth-alpha', type=float, default=0.0,
                    help='label smoothing coefficient')

# Architecture
parser.add_argument('--char-embed', type=int, default=16,
                    help='character vector embedding size')
parser.add_argument('--blocks', type=str, default='5, 10, 5',
                    help='Number of each constant grid size inception block')
parser.add_argument('--k', type=int, default=256)
parser.add_argument('--l', type=int, default=256)
parser.add_argument('--m', type=int, default=384)
parser.add_argument('--n', type=int, default=384)


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


def build_iters(train_df, test_df, feature_col, label_col, alphabet, hyperparameters):
    """
    :param train_df: pandas dataframe of training data
    :param test_df: pandas dataframe of test data
    :param feature_col: column in dataframe corresponding to text
    :param label_col: column in dataframe corresponding to label
    :param hyperparameters: dict of hyperparams
    :return: mxnet data iterators
    """
    # Fit preprocessor to training data
    preprocessor = UtterancePreprocessor(length=hyperparameters['sequence_length'], char_to_index=alphabet)
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
    train_iter = mx.io.NDArrayIter(data=X_train, label=Y_train, batch_size=hyperparameters['batch_size'], shuffle=True,
                                   last_batch_handle='pad')
    test_iter = mx.io.NDArrayIter(data=X_test, label=Y_test, batch_size=hyperparameters['batch_size'], shuffle=True,
                                  last_batch_handle='pad')
    return preprocessor, train_iter, test_iter


def build_symbol(iterator, preprocessor, hyperparameters):
    """
    :return:  MXNet symbol object
    """
    def conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True,
                                  name='%s%s_conv2d' % (name, suffix))
        bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
        act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' % (name, suffix))
        return act

    def stem(data, name):
        """
        figure 16 from https://arxiv.org/pdf/1602.07261.pdf
        """
        conv_1 = conv(data, num_filter=32, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv_1' + str(name))
        conv_2 = conv(conv_1, num_filter=32, kernel=(1, 3), stride=(1, 1), pad=(0, 0), name='conv_2' + str(name))
        conv_3 = conv(conv_2, num_filter=64, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv_3' + str(name))
        conv_4 = conv(conv_3, num_filter=96, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv_4' + str(name))
        maxpool = mx.sym.Pooling(conv_3, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='pool_1' + str(name), pool_type='max')

        concat_1 = mx.sym.Concat(*[conv_4, maxpool], dim=1, name='concat_1'+str(name))

        conv_5 = conv(concat_1, num_filter=64, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_5' + str(name))
        conv_6 = conv(conv_5, num_filter=64, kernel=(1, 7), stride=(1, 1), pad=(0, 3), name='conv_6' + str(name))
        conv_7 = conv(conv_6, num_filter=64, kernel=(1, 7), stride=(1, 1), pad=(0, 3), name='conv_7' + str(name))
        conv_8 = conv(conv_7, num_filter=96, kernel=(1, 3), stride=(1, 1), pad=(0, 0), name='conv_8' + str(name))

        conv_9 = conv(concat_1, num_filter=64, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_9' + str(name))
        conv_10 = conv(conv_9, num_filter=96, kernel=(1, 3), stride=(1, 1), pad=(0, 0), name='conv_10' + str(name))

        concat_2 = mx.sym.Concat(*[conv_8, conv_10], dim=1, name='concat_2' + str(name))

        maxpool = mx.sym.Pooling(concat_2, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='pool_2' + str(name), pool_type='max')
        conv_11 = conv(concat_2, num_filter=192, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv_11' + str(name))

        concat_3 = mx.sym.Concat(*[maxpool, conv_11], dim=1, name='concat_3' + str(name))
        return concat_3

    def inception_res_a(data, name):
        """
        figure 16 from https://arxiv.org/pdf/1602.07261.pdf
        """
        tower_1 = conv(data, num_filter=32, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_1.1' + str(name))
        tower_1 = conv(tower_1, num_filter=48, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv_1.2' + str(name))
        tower_1 = conv(tower_1, num_filter=64, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv_1.3' + str(name))

        tower_2 = conv(data, num_filter=32, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_2.1' + str(name))
        tower_2 = conv(tower_2, num_filter=32, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv_2.2' + str(name))

        tower_3 = conv(data, num_filter=32, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_3.1' + str(name))

        tower_concat = mx.sym.Concat(*[tower_1, tower_2, tower_3], dim=1, name='concat_1'+str(name))
        linear_conv = mx.sym.Convolution(tower_concat, num_filter=384, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                         name='linear_conv'+str(name))

        skip = linear_conv + data
        return mx.sym.Activation(skip, act_type='relu', name='block'+str(name))

    def inception_res_b(data, name, filters):
        """
        figure 11 from https://arxiv.org/pdf/1602.07261.pdf
        """
        tower_1 = conv(data, num_filter=128, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_1.1' + str(name))
        tower_1 = conv(tower_1, num_filter=128, kernel=(1, 7), stride=(1, 1), pad=(0, 3), name='conv_1.2' + str(name))
        tower_1 = conv(tower_1, num_filter=128, kernel=(1, 7), stride=(1, 1), pad=(0, 3), name='conv_1.3' + str(name))

        tower_2 = conv(data, num_filter=128, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_2.1' + str(name))

        tower_concat = mx.sym.Concat(*[tower_1, tower_2], dim=1, name='concat_1'+str(name))
        linear_conv = mx.sym.Convolution(tower_concat, num_filter=filters, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                         name='linear_conv'+str(name))

        skip = linear_conv + data
        return mx.sym.Activation(skip, act_type='relu', name='block'+str(name))

    def inception_res_c(data, name, filters):
        """
        figure 13 from https://arxiv.org/pdf/1602.07261.pdf
        """
        tower_1 = conv(data, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_1.1' + str(name))
        tower_1 = conv(tower_1, num_filter=192, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv_1.2' + str(name))
        tower_1 = conv(tower_1, num_filter=192, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv_1.3' + str(name))

        tower_2 = conv(data, num_filter=192, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_2.1' + str(name))

        tower_concat = mx.sym.Concat(*[tower_1, tower_2], dim=1, name='concat_1'+str(name))
        linear_conv = mx.sym.Convolution(tower_concat, num_filter=filters, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                         name='linear_conv'+str(name))

        skip = linear_conv + data
        return mx.sym.Activation(skip, act_type='relu', name='block'+str(name))

    def reduction_a(data, name, k, l, m, n):
        """
        figure 7 from https://arxiv.org/pdf/1602.07261.pdf
        """
        tower_1 = conv(data, num_filter=k, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_1.1' + str(name))
        tower_1 = conv(tower_1, num_filter=l, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv_1.2' + str(name))
        tower_1 = conv(tower_1, num_filter=m, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv_1.3' + str(name))

        tower_2 = conv(data, num_filter=n, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv_2.1' + str(name))

        maxpool = mx.sym.Pooling(data, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='pool_3.1' + str(name), pool_type='max')

        tower_concat = mx.sym.Concat(*[tower_1, tower_2, maxpool], dim=1, name='concat_1'+str(name))

        return tower_concat

    def reduction_b(data, name):
        """
        figure 12 from https://arxiv.org/pdf/1602.07261.pdf
        """
        tower_1 = conv(data, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_1.1' + str(name))
        tower_1 = conv(tower_1, num_filter=288, kernel=(1, 3), stride=(1, 1), pad=(0, 1), name='conv_1.2' + str(name))
        tower_1 = conv(tower_1, num_filter=320, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv_1.3' + str(name))

        tower_2 = conv(data, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_2.1' + str(name))
        tower_2 = conv(tower_2, num_filter=288, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv_2.2' + str(name))

        tower_3 = conv(data, num_filter=256, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name='conv_3.1' + str(name))
        tower_3 = conv(tower_3, num_filter=384, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='conv_3.2' + str(name))

        maxpool = mx.sym.Pooling(data, kernel=(1, 3), stride=(1, 2), pad=(0, 0), name='pool_3.1' + str(name), pool_type='max')

        tower_concat = mx.sym.Concat(*[tower_1, tower_2, tower_3, maxpool], dim=1, name='concat_1'+str(name))

        return tower_concat

    X_shape, Y_shape = iterator.provide_data[0][1], iterator.provide_label[0][1]

    data = mx.sym.Variable(name="data")
    softmax_label = mx.sym.Variable(name="softmax_label")
    print("data_input: ", data.infer_shape(data=X_shape)[1][0])
    print("label input: ", softmax_label.infer_shape(softmax_label=Y_shape)[1][0])

    # Embed each character
    embed = mx.sym.Embedding(data, input_dim=len(preprocessor.char_to_index), output_dim=hyperparameters['char_embed'])
    embed = mx.sym.Reshape(mx.sym.transpose(embed, axes=(0, 2, 1)), shape=(0, 0, 1, -1))
    print("embedded output: ", embed.infer_shape(data=X_shape)[1][0])

    # Initial conv layers
    stem = stem(embed, 'stem')
    print("stem output: ", stem.infer_shape(data=X_shape)[1][0])

    # Inception resnet blocks & reduction blocks
    for i in list(range(hyperparameters['blocks'][0])):
        if i == 0:
            block = inception_res_a(stem, 'inception_block_a_'+str(i))
        else:
            block = inception_res_a(block, 'inception_block_a_'+str(i))
        print("Inception A block {} module output shape: {}".format(i, block.infer_shape(data=X_shape)[1][0]))

    reduction_a = reduction_a(block, 'reduction_a', hyperparameters['k'], hyperparameters['l'],
                              hyperparameters['m'], hyperparameters['n'])
    print("Reduction A output shape: {}".format(reduction_a.infer_shape(data=X_shape)[1][0]))

    for i in list(range(hyperparameters['blocks'][1])):
        if i == 0:
            block = inception_res_b(reduction_a, 'inception_block_b_' + str(i),
                                    filters=hyperparameters['m'] + hyperparameters['n'] + 384)
        else:
            block = inception_res_b(block, 'inception_block_b_' + str(i),
                                    filters=hyperparameters['m'] + hyperparameters['n'] + 384)
        print("Inception B block {} module output shape: {}".format(i, block.infer_shape(data=X_shape)[1][0]))

    reduction_b = reduction_b(block, 'reduction_b')
    print("Reduction B output shape: {}".format(reduction_b.infer_shape(data=X_shape)[1][0]))

    for i in list(range(hyperparameters['blocks'][2])):
        if i == 0:
            block = inception_res_c(reduction_b, 'inception_block_c_' + str(i), filters=2144)
        else:
            block = inception_res_c(block, 'inception_block_c_' + str(i), filters=2144)
        print("Inception C block {} module output shape: {}".format(i, block.infer_shape(data=X_shape)[1][0]))

    avg_pool = mx.sym.Pooling(block, kernel=(1, 8), stride=(1, 1), pad=(0, 0), pool_type='avg', name='avg_pool')
    print("average pool output: ", avg_pool.infer_shape(data=X_shape)[1][0])

    dropout = mx.sym.Dropout(mx.sym.flatten(avg_pool), p=hyperparameters['dropout'], name='dropout')
    print("dropout output: ", dropout.infer_shape(data=X_shape)[1][0])

    output = mx.sym.FullyConnected(dropout, num_hidden=len(preprocessor.label_to_index), flatten=True, name='output')
    sm = mx.sym.SoftmaxOutput(output, softmax_label, hyperparameters['smooth_alpha'])
    print("softmax output: ", sm.infer_shape(data=X_shape)[1][0])

    return sm


def train(hyperparameters, channel_input_dirs, num_gpus, **kwargs):
    """
    This function can be called by Amazon Sagemaker.
    Trains an mxnet module.
    """
    # read pickled pandas df's from disk
    train_df = pd.read_pickle(channel_input_dirs['train'])
    test_df = pd.read_pickle(channel_input_dirs['test'])

    # Define vocab for lookup table
    alph = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} ~'
    char_to_index = {k: v for v, k in enumerate(list(alph))}

    # Build data iterators to feed our network
    preprocessor, train_iter, val_iter = build_iters(train_df, test_df, feature_col='utterance', label_col='intent',
                                                     alphabet=char_to_index, hyperparameters=hyperparameters)

    # Build network graph for computation
    symbol = build_symbol(train_iter, preprocessor, hyperparameters)

    # Build trainable module
    module = mx.mod.Module(symbol, context=mx.gpu() if num_gpus else mx.cpu())

    # Modify learning rate as we train
    schedule = mx.lr_scheduler.FactorScheduler(step=hyperparameters.get('lr_update_interval'),
                                               factor=hyperparameters.get('lr_update_factor'))

    # Initialize conv filter weights using MSRAPRelu so we can train deeper architectures
    init = mx.initializer.Mixed(patterns=['conv2d_weight', '.*'],
                                initializers=[mx.initializer.MSRAPrelu(factor_type='avg', slope=0.25),
                                              mx.initializer.Normal(sigma=0.02)])

    # Fit the model to the training data
    module.fit(train_data=train_iter,
               eval_data=val_iter,
               optimizer=hyperparameters['optimizer'],
               eval_metric=mx.metric.Accuracy(),
               optimizer_params={'learning_rate': hyperparameters.get('lr'),
                                 'wd': hyperparameters.get('l2'),
                                 'momentum': hyperparameters.get('momentum'),
                                 'lr_scheduler': schedule},
               initializer=init,
               num_epoch=hyperparameters.get('epochs'))

    return module  # return module so Sagemaker saves it


if __name__ == '__main__':
    args = parser.parse_args()
    args.blocks = ast.literal_eval(args.blocks)

    train(hyperparameters=vars(args), channel_input_dirs={'train': args.train_path, 'test': args.test_path},
          num_gpus=args.gpus)
