#!/usr/bin/env bash

# hmmm
python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=32 --batch-size=512 --lr=0.005 --blocks='10,10,4,4' --channels='64,128,256,512' --fc-size=2048 --fc-dropout=0.7

# should achieve 90.83 test score (13 minutes per epoch). achieves 88.7
# python vdcnn.py --gpus=0 --sequence-length=1024 --batch-size=128 --lr=0.01 --blocks='2,2,2,2' --channels='64,128,256,512' --final-pool