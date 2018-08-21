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

import sagemaker
from sagemaker.mxnet import MXNet
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
import argparse
import boto3

# -*- coding: utf-8 -*-

parser = argparse.ArgumentParser(description="MXNet + Sagemaker hyperparameter optimization job",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--role_arn', type=str,
                    default='arn:aws:iam::430515702528:role/service-role/AmazonSageMaker-ExecutionRole-20180730T100605',
                    help='arn of sagemaker execution role')
parser.add_argument('--bucket_name', type=str, default='finn-dl-sandbox-atlas',
                    help='bucket to store code, data and artifacts')
parser.add_argument('--job-name', type=str, default='vdcnn', help='name of job')
parser.add_argument('--train-code', type=str, default='./vdcnn.py', help='python module containing train() function')
parser.add_argument('--train-instance-type', type=str, default='ml.m4.xlarge', help='instance type for training')
parser.add_argument('--train-instance-count', type=int, default=1, help='number of instances to distribute training')
parser.add_argument('--max-jobs', type=int, default=500, help='number of hyperparameter jobs to run')
parser.add_argument('--max-parallel-jobs', type=int, default=1, help='number of parallel jobs to run')
parser.add_argument('--data-dir', type=str, default='atb_model_46/strat_split/data', help='path to train/test pickle files')


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Assume permissions to provision services with Sagemaker role
    session = boto3.Session(profile_name='sagemaker_execution')
    sagemaker_session = sagemaker.Session(boto_session=session)
    local_session = sagemaker.local.local_session.LocalSession(boto_session=session)

    # Initialize variables
    custom_code_upload_location = 's3://{}/customcode/mxnet'.format(args.bucket_name)
    model_artifacts_location = 's3://{}/artifacts'.format(args.bucket_name)
    data_path = 's3://{}/{}'.format(args.bucket_name, args.data_dir)

    search_space = {'lr': ContinuousParameter(0.01, 0.1),
                    'momentum': ContinuousParameter(0.92, 0.999),
                    'batch_size': IntegerParameter(8, 128),
                    'dropout': ContinuousParameter(0.2, 0.9),
                    'smooth_alpha': ContinuousParameter(0.0, 0.1),
                    'char_embed': IntegerParameter(4, 16),
                    'temp_conv_filters': IntegerParameter(16, 64),
                    'block1_blocks': IntegerParameter(1, 5),
                    'block2_blocks': IntegerParameter(1, 5),
                    'block3_blocks': IntegerParameter(1, 2),
                    'block4_blocks': IntegerParameter(1, 2)
                    }

    hyperparameters = {'epochs': 30,
                       'batch_size': 13,
                       'optimizer': 'sgd',
                       'lr': 0.036859536068138014,
                       'lr_update_factor': 0.97,
                       'lr_update_epoch': 3,
                       'momentum': 0.9219489992332196,
                       'dropout': 0.4,
                       'smooth_alpha': 0.0402,
                       'grad_clip': 0.0, 'l2': 0.0,
                       'k': 256, 'l': 256, 'm': 384, 'n': 384, 'max_train_records': None,
                       'gpus': '1',
                       'sequence_length': 64,
                       'pool_type': 'max',
                       'block1_blocks': 1,
                       'block2_blocks': 1,
                       'block3_blocks': 1,
                       'block4_blocks': 1,
                       'char_embed': 16,
                       'temp_conv_filters': 64
                       }

    parser.add_argument('--char-embed', type=int, default=4,
                        help='character vector embedding size')
    parser.add_argument('--temp-conv-filters', type=int, default=8,
                        help='character vector embedding size')

    # Create an estimator
    estimator = MXNet(sagemaker_session=sagemaker_session if 'local' not in args.train_instance_type else local_session,
                      hyperparameters=hyperparameters,
                      entry_point=args.train_code,
                      role=args.role_arn,
                      output_path=model_artifacts_location,
                      code_location=custom_code_upload_location,
                      train_instance_count=args.train_instance_count,
                      train_instance_type=args.train_instance_type,
                      base_job_name=args.job_name,
                      py_version='py3',
                      framework_version='1.1.0',
                      train_volume_size=1)

    # Configure Hyperparameter Tuner
    my_tuner = HyperparameterTuner(estimator=estimator,
                                   objective_metric_name='Validation-accuracy',
                                   hyperparameter_ranges=search_space,
                                   metric_definitions=[
                                       {'Name': 'Validation-accuracy', 'Regex': 'Best Validation-accuracy=(\d\.\d+)'}],
                                   max_jobs=args.max_jobs,
                                   max_parallel_jobs=args.max_parallel_jobs)

    # Start hyperparameter tuning job
    my_tuner.fit({'train': data_path, 'test': data_path})
    # estimator.fit({'train': data_path, 'test': data_path})
