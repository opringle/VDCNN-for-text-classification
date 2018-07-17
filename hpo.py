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

parser.add_argument('--bucket_name', type=str, default='sagemaker-text-classification-data',
                    help='bucket to store code, data and artifacts')
parser.add_argument('--job-name', type=str, default='vdcnn', help='name of job')
parser.add_argument('--train-code', type=str, default='./vdcnn.py', help='python module containing train() function')
parser.add_argument('--train-instance-type', type=str, default='ml.p2.xlarge', help='instance type for training')
parser.add_argument('--train-instance-count', type=int, default=1, help='number of instances to distribute training')
parser.add_argument('--max-jobs', type=int, default=100, help='number of hyperparameter jobs to run')
parser.add_argument('--max-parallel-jobs', type=int, default=1, help='number of parallel jobs to run')
parser.add_argument('--train-dir', type=str, default='data/ag_news', help='path to train data from bucket root')
parser.add_argument('--test-dir', type=str, default='data/ag_news', help='path to test data from bucket root')


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Grant permissions to call services (S3, SageMaker etc)
    session = boto3.Session(profile_name='personal')
    s3 = session.resource('s3')
    sagemaker_session = sagemaker.Session(boto_session=session)
    # role = sagemaker.get_execution_role(sagemaker_session=sagemaker_session)
    # This retrieves the ARN string, if you are in an instance that has the role attached
    role = 'arn:aws:iam::...'

    # Initialize variables
    custom_code_upload_location = 's3://{}/customcode/mxnet'.format(args.bucket_name)
    model_artifacts_location = 's3://{}/artifacts'.format(args.bucket_name)
    train_path = 's3://{}/{}'.format(args.bucket_name, args.train_dir)
    test_path = 's3://{}/{}'.format(args.bucket_name, args.test_dir)

    search_space = {'lr': ContinuousParameter(0.00001, 0.5),
                    'momentum': ContinuousParameter(0.0, 0.999),
                    'l2': ContinuousParameter(0.0, 0.999),
                    'batch_size': IntegerParameter(8, 512),
                    'dropout': ContinuousParameter(0.0, 0.99),
                    'smooth_alpha': ContinuousParameter(0.0, 0.5),
                    'blocks': CategoricalParameter([[1, 1, 1], [2, 4, 2], [4, 8, 4], [5, 10, 5]]),
                    'pool_type': CategoricalParameter(['max', 'avg']),
                    'lr_reduce_factor': ContinuousParameter(0.5, 0.99),
                    'lr_reduce_epoch': ContinuousParameter(1, 10)
                    }

    hyperparameters = {'epochs': 30, 'batch_size': 128,
                       'sequence_length': 299, 'optimizer': 'sgd', 'lr': 0.01,
                       'momentum': 0.9, 'lr_update_factor': 0.97, 'lr_update_interval': 30000,
                       'grad_clip': 0.0, 'l2': 0.0, 'dropout': 0.2, 'smooth_alpha': 0.0,
                       'char_embed': 16, 'blocks': (5, 10, 5), 'k': 256, 'l': 256, 'm': 384, 'n': 384}

    # Create an estimator
    estimator = MXNet(sagemaker_session=sagemaker_session if 'local' not in args.train_instance_type else None,
                      hyperparameters=hyperparameters,
                      entry_point=args.train_code,
                      role=role,
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
                                   objective_metric_name='validation-accuracy',
                                   hyperparameter_ranges=search_space,
                                   metric_definitions=[
                                       {'Name': 'validation-accuracy', 'Regex': 'validation-accuracy=(\d\.\d+)'}],
                                   max_jobs=args.max_jobs,
                                   max_parallel_jobs=args.max_parallel_jobs)

    # Start hyperparameter tuning job
    # my_tuner.fit({'train': train_path, 'test': test_path})
    estimator.fit({'train': train_path, 'test': test_path})
