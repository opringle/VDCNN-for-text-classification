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
import regex as re
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
parser.add_argument('--batch-size', type=int, default=512,
                    help='the number of training records in each minibatch')

parser.add_argument('--max-words', type=int, default=20,
                    help='the number of words in each utterance')
parser.add_argument('--max-word-length', type=int, default=20,
                    help='the number of characters in each word')

parser.add_argument('--embed-size', type=int, default=26,
                    help='character embeddding size')
parser.add_argument('--embed-dropout', type=float, default=0.5,
                    help='dropout rate for char embeddings')
parser.add_argument('--rnn-size', type=int, default=22,
                    help='size of each rnn hidden state')
parser.add_argument('--rnn-dropout', type=float, default=0.2,
                    help='dropout rate for rnn hidden states')
parser.add_argument('--cnn-filter-size', type=int, default=3,
                    help='height/width of cnn filters')
parser.add_argument('--filters', type=int, default=100,
                    help='number of cnn filters')
parser.add_argument('--pool-filter-size', type=int, default=2,
                    help='height/width of pooling filters')
parser.add_argument('--penultimate-dropout', type=float, default=0.4,
                    help='dropout rate for penultimate layer')


parser.add_argument('--optimizer', type=str, default='SGD',
                    help='optimization algorithm to update model parameters with')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate for chosen optimizer')
parser.add_argument('--l2', type=float, default=0.0,
                    help='l2 regularization coefficient')
parser.add_argument('--smooth-alpha', type=float, default=0.0,
                    help='label smoothing coefficient')
parser.add_argument('--max-train-utt-per-intent', type=int, default=None,
                    help='the number of training records in each minibatch')


class UtterancePreprocessor:
    """
    preprocessor that can be fit to data in order to preprocess it
    """
    def __init__(self, max_words, max_word_length, unknown_char_index, char_to_index=None, pad_char='~'):
        self.max_words = max_words
        self.max_word_length = max_word_length
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

    def split_utterance(self, string):
        """
        :param utterance: string
        :return: list of string, split using regex
        """
        string = re.sub(r"[[0-9]*(\\.[0-9]*)?]", "", string)
        string = re.sub(r"(')", "", string)
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
        return string.strip().lower().split(" ")

    def pad_list(self, list, length):
        """
        :param list:
        :param length:
        :return:
        """
        diff = length - len(list)
        if diff > 0:
            self.padded_data += 1
            list.extend([self.pad_char] * diff)
            return list
        else:
            self.sliced_data += 1
            return list[:length]

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
        tokenized_utterance = self.split_utterance(utterance)
        padded_utterance = self.pad_list(tokenized_utterance, self.max_words)
        char_tokenized_utterance = [list(token) for token in padded_utterance]
        padded_tokenized_utterance = [self.pad_list(charlist, self.max_word_length) for charlist in char_tokenized_utterance]
        indexed_utterance = [[self.char_to_index.get(char, self.unknown_char_index) for char in charlist]
                             for charlist in padded_tokenized_utterance]
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
    preprocessor = UtterancePreprocessor(args.max_words, args.max_word_length, unknown_char_index=len(alphabet),
                                         char_to_index=alphabet)
    preprocessor.fit(train_df[feature_col].values.tolist(), train_df[label_col].values.tolist())

    print(preprocessor.transform_utterance("I want to log in!"))

    print("index of unknown characters = {}\n"
          "padded characters represented as = {}\n"
          "index of pad char in dict: {}".format(preprocessor.unknown_char_index, preprocessor.pad_char,
                                                 alphabet[preprocessor.pad_char]))


    # Transform data
    train_df['X'] = train_df[feature_col].apply(preprocessor.transform_utterance)
    test_df['X'] = test_df[feature_col].apply(preprocessor.transform_utterance)
    train_df['Y'] = train_df[label_col].apply(preprocessor.transform_label)
    test_df['Y'] = test_df[label_col].apply(preprocessor.transform_label)
    print("{} utterances were padded & {} utterances were sliced to {} tokens and {} characters".format(preprocessor.padded_data,
                                                                                        preprocessor.sliced_data,
                                                                                        preprocessor.max_words,
                                                                                                        preprocessor.max_word_length))

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


def build_symbol(iterator, preprocessor, embed_dropout, rnn_hidden_units, rnn_dropout, cnn_filter_size, cnn_filters,
                 pool_filter, penultimate_dropout):
    """
    :return:  MXNet symbol object
    """
    X_shape, Y_shape = iterator.provide_data[0][1], iterator.provide_label[0][1]

    data = mx.sym.Variable(name="data")
    softmax_label = mx.sym.Variable(name="softmax_label")
    print("data_input: ", data.infer_shape(data=X_shape)[1][0])
    print("label input: ", softmax_label.infer_shape(softmax_label=Y_shape)[1][0])

    # Embed each character to 16 channels
    embedded_data = mx.sym.Embedding(data, input_dim=len(preprocessor.char_to_index), output_dim=args.embed_size)
    embedded_data = mx.sym.transpose(embedded_data, axes=(0, 1, 2, 3))
    embedded_data = mx.sym.Dropout(embedded_data, p=embed_dropout, name='embed_dropout')
    print("embedded output: ", embedded_data.infer_shape(data=X_shape)[1][0])

    # bidirectional lstm layer
    stacked_rnn_cells = mx.rnn.SequentialRNNCell()
    stacked_rnn_cells.add(mx.rnn.LSTMCell(num_hidden=rnn_hidden_units))
    stacked_rnn_cells.add(mx.rnn.DropoutCell(rnn_dropout))
    output, states = stacked_rnn_cells.unroll(length=args.max_words, inputs=embedded_data, merge_outputs=True)
    print("lstm output shape: {}".format(output.infer_shape(data=X_shape)[1][0]))

    cnn_input = mx.sym.Reshape(output, shape=(0, 1, args.max_words, -1), name='cnn_input')
    print("cnn input shape: {}".format(cnn_input.infer_shape(data=X_shape)[1][0]))

    # 2d convolutions
    convi = mx.sym.Convolution(data=cnn_input, kernel=(cnn_filter_size, cnn_filter_size), num_filter=cnn_filters, name="convi")
    relui = mx.sym.Activation(data=convi, act_type='relu', name="acti")
    print("cnn output shape: {}".format(relui.infer_shape(data=X_shape)[1][0]))

    # 2d pooling
    pooli = mx.sym.Pooling(data=relui, pool_type='max',
                           kernel=(pool_filter, pool_filter),
                           stride=(pool_filter, pool_filter),
                           pad=(0, 0),
                           name="pooli")
    print("pooling output shape: {}".format(pooli.infer_shape(data=X_shape)[1][0]))

    # flatten and dropout
    penultimate = mx.sym.Dropout(mx.sym.flatten(pooli), p=penultimate_dropout, name='penultimate')
    print("penultimate output shape: {}".format(penultimate.infer_shape(data=X_shape)[1][0]))

    # Pass to fully connected layer to map to neuron per class
    fc = mx.sym.FullyConnected(data=penultimate, num_hidden=len(preprocessor.label_to_index), name='output_')
    print("output shape: {}".format(fc.infer_shape(data=X_shape)[1][0]))

    # Softmax output
    sm = mx.sym.SoftmaxOutput(data=fc, label=softmax_label, name='softmax')
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
               optimizer_params={'learning_rate': args.lr, 'wd': args.l2},
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
    # Parse args
    args = parser.parse_args()

    # Setup dirs
    os.mkdir(args.output_dir) if not os.path.exists(args.output_dir) else None

    # Read training data into pandas data frames and sample if desired
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

    print(preprocessor.transform_utterance("I want to log in"))

    # Build network graph
    symbol = build_symbol(train_iter, preprocessor, args.embed_dropout, args.rnn_size, args.rnn_dropout,
                          args.cnn_filter_size, args.filters, args.pool_filter_size, args.penultimate_dropout)

    # Train the model
    trained_module = train(symbol, train_iter, val_iter)