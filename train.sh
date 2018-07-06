#!/usr/bin/env bash

# train the model with various max utt per intent
# assess how much data before performance degrades
#python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=128 --batch-size=128 \
#--lr=0.01 --channels=192,384,768 --blocks=3,5,2 --l2=0.01 --max-train-utt-per-intent=512

# should achieve 90.83 test score (13 minutes per epoch). achieves 88.7
# python vdcnn.py --gpus=0 --sequence-length=1024 --batch-size=128 --lr=0.01 --blocks='2,2,2,2' --channels='64,128,256,512' --final-pool
# python -u vdcnn.py --gpus='' --max-words=20 --max-word-length=20 --batch-size=128 --lr=0.01 --channels=334,640,2048 --blocks=3,5,2 --max-train-utt-per-intent=30000 --smooth-alpha=0.1 --fc-dropout=0.75

# character level recurrent cnn
python -u vdcnn.py --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ \
--gpus=0 --max-words=13 --max-word-length=8 --batch-size=256 --num-epochs=256 \
--embed-size=16 --embed-dropout=0.1 --rnn-size=300 --rnn-dropout=0.2 --cnn-filter-size=3 \
--filters=100 --pool-filter-size=2 --penultimate-dropout=0.4
--lr=0.1