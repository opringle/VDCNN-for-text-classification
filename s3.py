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
import argparse
import boto3
import os

# -*- coding: utf-8 -*-

parser = argparse.ArgumentParser(description="MXNet + Sagemaker hyperparameter optimization job",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('bucket_name', type=str, help='bucket to store code, data and artifacts')
parser.add_argument('--s3-path', type=str, default='data/ag_news/', help='s3 pickle location')
parser.add_argument('--local-path', type=str, default='data/ag_news/', help='local pickle location')


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Authenticate to a specific set of shared credentials
    session = sagemaker.Session(boto_session=boto3.Session(profile_name='personal'))

    # Upload specified data to s3
    session.resource('s3').Bucket(args.bucket_name).Object(args.local_path + 'train.pickle').\
        upload_file(os.path.join(args.local_path, 'train.pickle'))
    session.resource('s3').Bucket(args.bucket_name).Object(args.local_path + 'test.pickle').\
        upload_file(os.path.join(args.local_path, 'test.pickle'))
