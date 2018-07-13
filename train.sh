#!/usr/bin/env bash

# ATB data (85.8 to beat)
#python -u vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --output-dir=model_atb \
#--num-epochs=256 --lr=0.075 --momentum=0.9 --dropout=0.2 --smooth-alpha=0.001 --lr-reduce-factor=1 --lr-reduce-epoch=3

# AGNEWs data (91.33 to beat)
python -u vdcnn.py --gpus=0 --data=./data/ag_news --output-dir=model_ag_news \
--num-epochs=256 --lr=0.075 --momentum=0.9 --dropout=0.2 --smooth-alpha=0.001 --lr-reduce-factor=1 --lr-reduce-epoch=3 \
--blocks=1,2,1

# YAHOO data (73.43 to beat)