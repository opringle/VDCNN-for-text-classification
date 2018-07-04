#!/usr/bin/env bash

# matches inception architecture
python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ \
--num-epochs=256 --batch-size=512 --sequence-length=120 \
--lr=0.005 --epsilon=0.00000001 --grad-clip=0.0 \
--l2=0.0 --dropout=0.0 --smooth-alpha=0.0 \
--blocks=3,5,2 --channels=384,640,2048

python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ \
--num-epochs=256 --batch-size=512 --sequence-length=120 \
--lr=0.005 --epsilon=0.00000001 --grad-clip=0.0 \
--l2=0.0 --dropout=0.0 --smooth-alpha=0.0 \
--blocks=3,5,2 --channels=384,640,2048

# should achieve 90.83 test score (13 minutes per epoch). achieves 88.7
# python vdcnn.py --gpus=0 --sequence-length=1024 --batch-size=128 --lr=0.01 --blocks='2,2,2,2' --channels='64,128,256,512' --final-pool