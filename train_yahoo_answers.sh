#!/usr/bin/env bash
# should achieve 90.83 test score (6 minutes per epoch). achieves 87
python -u vdcnn.py --data=./data/yahoo_answers --output-dir=./ag_news --gpus=0 --num-epoch=15 \
--lr=0.01 --momentum=0.9 --lr-reduce-epoch=3 --lr-reduce-factor=0.5 --batch-size=128 \
--sequence-length=1024 --blocks='5,5,2,2' --channels='64,128,256,512' --fc-size=2048