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

# We must pip install in code until an image with pandas is available :(
# Alternatively we could create our own custom image.
from pip._internal import main as pipmain
pipmain(['install', 'pandas'])
import pandas as pd
import mxnet as mx
import os
import numpy as np
from itertools import chain
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Deep inception inspired cnn for text classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train-dir', type=str, default='./data/ag_news',
                        help='path to pickled pandas train df')
parser.add_argument('--test-dir', type=str, default='./data/ag_news',
                        help='path to pickled pandas test df')
parser.add_argument('--gpus', type=int, default=None,
                    help='list of gpus to run, e.g. 0 or 0,2,5. negate to use cpu.')

parser.add_argument('--epochs', type=int, default=30,
                    help='how  many times to update the model parameters')
parser.add_argument('--batch-size', type=int, default=12,
                    help='the number of training records in each minibatch')
parser.add_argument('--sequence-length', type=int, default=64,
                    help='the number of characters in each training example')
parser.add_argument('--max-train-records', type=int, default=None,
                    help='the number of training examples')

# Optimizer
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimization algorithm to update model parameters with')
parser.add_argument('--lr', type=float, default=0.039801136409934885,
                    help='learning rate for chosen optimizer')
parser.add_argument('--momentum', type=float, default=0.9312060606446801,
                    help='momentum for optimizer')
parser.add_argument('--lr-update-factor', type=float, default=0.97,
                    help='factor to reduce lr by')
parser.add_argument('--lr-update-epoch', type=int, default=3,
                    help='reduce lr every n epochs')
parser.add_argument('--grad-clip', type=float, default=0.0,
                    help='Clips weights to +- this value')

# Regularization
parser.add_argument('--l2', type=float, default=0.0,
                    help='l2 regularization coefficient')
parser.add_argument('--dropout', type=float, default=0.21612639141339468,
                    help='dropout regularization probability for penultimate layer')
parser.add_argument('--smooth-alpha', type=float, default=0.04395768050161286,
                    help='label smoothing coefficient')

# Architecture
parser.add_argument('--char-embed', type=int, default=16,
                    help='character vector embedding size')
parser.add_argument('--temp-conv-filters', type=int, default=64,
                    help='character vector embedding size')
parser.add_argument('--block1_blocks', type=int, default=1,
                    help='number of blocks in section')
parser.add_argument('--block2_blocks', type=int, default=1,
                    help='number of blocks in section')
parser.add_argument('--block3_blocks', type=int, default=1,
                    help='number of blocks in section')
parser.add_argument('--block4_blocks', type=int, default=1,
                    help='number of blocks in section')
parser.add_argument('--pool-type', type=str, default='avg', help='type of pooling before fc layers')


class UtterancePreprocessor:
    """
    preprocessor that can be fit to data in order to preprocess it
    """
    def __init__(self, length, char_to_index=None, pad_char='🙀', unknown_char='☠', space_char='🤮'):
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
        # print("Space tokens represented as {}\n"
        #       "Unknown tokens represented as {}\n"
        #       "Padded tokens represented as {}".format(self.space_char, self.unknown_char, self.pad_char))

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

    # print("vocabulary used in lookup table: {}".format(preprocessor.char_to_index))

    # Get data as numpy array
    X_train, X_test = np.array(train_df['X'].values.tolist()), np.array(test_df['X'].values.tolist())
    Y_train, Y_test = np.array(train_df['Y'].values.tolist()), np.array(test_df['Y'].values.tolist())

    # Build MXNet data iterators
    train_iter = mx.io.NDArrayIter(data=X_train, label=Y_train, batch_size=hyperparameters['batch_size'], shuffle=True,
                                   last_batch_handle='pad')
    test_iter = mx.io.NDArrayIter(data=X_test, label=Y_test, batch_size=hyperparameters['batch_size'], shuffle=True,
                                  last_batch_handle='pad')
    return preprocessor, train_iter, test_iter


def build_symbol_vdcnn(iterator, preprocessor, hyperparameters):
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
    embedded_data = mx.sym.Embedding(data, input_dim=len(preprocessor.char_to_index), output_dim=hyperparameters['char_embed'])
    embedded_data = mx.sym.Reshape(mx.sym.transpose(embedded_data, axes=(0, 2, 1)), shape=(0, 0, 1, -1))
    print("embedded output: ", embedded_data.infer_shape(data=X_shape)[1][0])

    # Temporal Convolutional Layer (without activation)
    temp_conv_1 = mx.sym.Convolution(embedded_data, kernel=(1, 3), num_filter=hyperparameters['temp_conv_filters'], pad=(0, 1))
    print("temp conv output: ", temp_conv_1.infer_shape(data=X_shape)[1][0])

    # Create convolutional blocks with pooling in-between
    channels = (hyperparameters['temp_conv_filters'],
                2 * hyperparameters['temp_conv_filters'],
                4 * hyperparameters['temp_conv_filters'],
                8 * hyperparameters['temp_conv_filters'])

    blocks = (hyperparameters['block1_blocks'],
              hyperparameters['block2_blocks'],
              hyperparameters['block3_blocks'],
              hyperparameters['block4_blocks'])

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

    # pool output to one feature per kernel
    pool_k = block.infer_shape(data=X_shape)[1][0][3]
    print("{0} pool kernel size {1}, stride 1".format(hyperparameters['pool_type'], pool_k))
    block = mx.sym.flatten(mx.sym.Pooling(block, kernel=(1, pool_k), stride=(1, 1), pad=(0, 0), pool_type=hyperparameters['pool_type']))
    print("flattened pooling output: {1}".format(pool_k, block.infer_shape(data=X_shape)[1][0]))

    drop = mx.sym.Dropout(block, p=hyperparameters['dropout'])
    print("dropout output: ", block.infer_shape(data=X_shape)[1][0])

    output = mx.sym.FullyConnected(drop, num_hidden=len(preprocessor.label_to_index), flatten=True, name='output')
    sm = mx.sym.SoftmaxOutput(output, softmax_label, hyperparameters['smooth_alpha'])
    print("softmax output: ", sm.infer_shape(data=X_shape)[1][0])

    return sm


def best_val_score(val_iter, module, metric, scores):
    """
    callback function to print best validation score
    :return:
    """
    def _callback(epoch, sym=None, arg=None, aux=None):

        metric.reset()
        val_iter.reset()

        for batch in val_iter:
            module.forward(batch, is_train=False)
            module.update_metric(metric, batch.label)

        score = metric.get()[1]
        scores.append(score)
        metric.reset()
        val_iter.reset()
        best_score = max(scores)
        print("Epoch[{}] Validation-accuracy={:.6f}".format(epoch, score))
        print("Epoch[{}] Best Validation-accuracy={:.6f}".format(epoch, best_score))
    return _callback


def train(hyperparameters, channel_input_dirs, num_gpus, **kwargs):
    """
    This function can be called by Amazon Sagemaker.
    Trains an mxnet module.
    """
    # read pickled pandas df's from disk
    train_df = pd.read_pickle(os.path.join(channel_input_dirs['train'], 'train.pickle'))
    if hyperparameters['max_train_records']:
        train_df = train_df[:hyperparameters['max_train_records']]
    test_df = pd.read_pickle(os.path.join(channel_input_dirs['test'], 'test.pickle'))

    # Define vocab for lookup table
    alph = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} ~'
    char_to_index = {k: v for v, k in enumerate(list(alph))}

    # Build data iterators to feed our network
    preprocessor, train_iter, val_iter = build_iters(train_df, test_df, feature_col='utterance', label_col='intent',
                                                     alphabet=char_to_index, hyperparameters=hyperparameters)

    # Build network graph for computation
    symbol = build_symbol_vdcnn(train_iter, preprocessor, hyperparameters)

    # Build trainable module
    module = mx.mod.Module(symbol, context=mx.gpu() if hyperparameters['gpus'] else mx.cpu())

    # Modify learning rate as we train
    batches_per_epoch = (train_df.shape[0] // hyperparameters['batch_size']) + 1
    step = hyperparameters['lr_update_epoch'] * batches_per_epoch
    schedule = mx.lr_scheduler.FactorScheduler(step=step, factor=hyperparameters['lr_update_factor'])

    # Initialize conv filter weights using MSRAPRelu to help training deeper architectures
    init = mx.initializer.Mixed(patterns=['conv2d_weight', '.*'],
                                initializers=[mx.initializer.MSRAPrelu(factor_type='avg', slope=0.25),
                                              mx.initializer.Normal(sigma=0.02)])

    # Define a metric object to pass around
    metric = mx.metric.Accuracy()
    scores = []

    # Fit the model to the training data
    module.fit(train_data=train_iter,
               optimizer=hyperparameters['optimizer'],
               eval_metric=metric,
               optimizer_params={'learning_rate': hyperparameters.get('lr'),
                                 'wd': hyperparameters.get('l2'),
                                 'momentum': hyperparameters.get('momentum'),
                                 'lr_scheduler': schedule},
               initializer=init,
               num_epoch=hyperparameters.get('epochs'),
               epoch_end_callback=best_val_score(val_iter, module, metric, scores))

    return module  # return module so Sagemaker saves it


if __name__ == '__main__':
    args = parser.parse_args()

    module = train(hyperparameters=vars(args), channel_input_dirs={'train': args.train_dir, 'test': args.test_dir},
                   num_gpus=args.gpus)

    module.save_checkpoint(prefix='./checkpoint/transfer_model', epoch=args.epochs)
