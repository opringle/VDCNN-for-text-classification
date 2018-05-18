#!/usr/bin/env bash

#
python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=128 --dropout=0.0 --final-pool --batch-size=512


# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=128 --dropout=0.0 --final-pool --batch-size=128