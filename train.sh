#!/usr/bin/env bash

# hmmm
python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=64 --batch-size=512 --lr=0.001 --blocks='2,2,2,2' --channels='64,128,256,512'

# should achieve 90.83 test score (13 minutes per epoch in the paper)
python vdcnn.py --gpus=0 --sequence-length=1024 --batch-size=128 --lr=0.01 --blocks='2,2,2,2' --channels='64,128,256,512' --final-pool