#!/usr/bin/env bash
# should achieve 90.83 test score (6 minutes per epoch). achieves 87.35
python -u vdcnn.py --data=./data/ag_news --output-dir=./ag_news --gpus=0 --num-epoch=150 \
--lr=0.01 --momentum=0.9 --lr-reduce-epoch=3 --lr-reduce-factor=1.0 --batch-size=128 \
--sequence-length=1024 --blocks='1,1,1,1,1,1,1,1' --channels='32,64,128,256,512,1024,2048,4096' --dropout=0.5