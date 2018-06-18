#!/usr/bin/env bash

# hmmm
python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=64 --batch-size=512 --lr=0.001 --blocks='10,10,4,4' --channels='64,128,256,512' --hidden-dropout=0.5