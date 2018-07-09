#!/usr/bin/env bash

# hmmm
# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=32 --batch-size=512 --lr=0.005 --blocks='10,10,4,4' --channels='64,128,256,512' --fc-size=2048 --fc-dropout=0.7

# should achieve 73.43 test score (XXX minutes per epoch). achieves
#python -u vdcnn.py --data=./data/yahoo_answers --output-dir=./vdcnn_yahoo --gpus=0 \
#--lr=1.0 --momentum=0.99 --lr-reduce-epoch=100 --lr-reduce-factor=2 --batch-size=128 \
#--sequence-length=1024 --blocks='10,10,4,4' --channels='64,128,256,512' --final-pool --fc-size=2048 --fc-dropout=0.0

# should achieve 91.33 test score (XXX minutes per epoch). achieves XXX
python -u vdcnn.py --data=./data/ag_news --output-dir=./ag_news --gpus=0 --num-epoch=150 \
--lr=0.01 --momentum=0.9 --lr-reduce-epoch=3 --lr-reduce-factor=0.5 --batch-size=128 \
--sequence-length=1024 --blocks='10,10,4,4' --channels='64,128,256,512' --final-pool --fc-size=2048 --fc-dropout=0.0

# set momentum as close to 1 as possible, make alpha as large as possible without divergance