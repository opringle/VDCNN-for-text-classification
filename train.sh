#!/usr/bin/env bash

# ATB data (85.8 to beat)
python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --output-dir=atb_model \
--checkpoint-freq=1 --load-prefix=yahoo_model/checkpoint --load-epoch=11 \
--num-epochs=256 --batch-size=512 --sequence-length=107 --char-embed=16 \
--lr=0.00006 \
--l2=0.0 --dropout=0.8 --smooth-alpha=0.001 \
--blocks=4,7,3

# AGNEWs data (91.33 to beat)
#python vdcnn.py --gpus=0 --data=./data/ \
#--num-epochs=256 --batch-size=128 --sequence-length=299 --char-embed=16 \
#--lr=0.00006 \
#--l2=0.0 --dropout=0.0 --smooth-alpha=0.001 \
#--blocks=4,7,3

# YAHOO data (73.43 to beat)
#python -u vdcnn.py --gpus=0 --data=./data/yahoo_answers/ --output-dir=yahoo_model --checkpoint-freq=1 \
#--checkpoint-freq=1 \
#--num-epochs=256 --batch-size=512 --sequence-length=107 --char-embed=16 \
#--lr=0.00006 \
#--l2=0.0 --dropout=0.8 --smooth-alpha=0.001 \
#--blocks=4,7,3