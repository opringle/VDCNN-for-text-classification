#!/usr/bin/env bash

# train the model with various max utt per intent
# assess how much data before performance degrades
#python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=128 --batch-size=128 \
#--lr=0.01 --channels=192,384,768 --blocks=3,5,2 --l2=0.01 --max-train-utt-per-intent=512

# should achieve 90.83 test score (13 minutes per epoch). achieves 88.7
# python vdcnn.py --gpus=0 --sequence-length=1024 --batch-size=128 --lr=0.01 --blocks='2,2,2,2' --channels='64,128,256,512' --final-pool
python -u vdcnn.py --gpus='' --max-words=20 --max-word-length=20 --batch-size=128 --lr=0.01 --channels=334,640,2048 --blocks=3,5,2 --max-train-utt-per-intent=30000 --smooth-alpha=0.1 --fc-dropout=0.75