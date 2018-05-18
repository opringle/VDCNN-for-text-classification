#!/usr/bin/env bash

# OK
# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=128 --dropout=0.0 --final-pool --batch-size=512

# > 76.8% validation score
# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=128 --final-pool --batch-size=512 --encode --lr=0.0005 --l2=0 --dropout=0.0

# > 76.2% validation score
# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=128 --batch-size=512 --encode --lr=0.0005 --l2=0.001 --dropout=0.0

# STABLE! > 76.3% validation score
# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=64 --batch-size=128 --encode --lr=0.001 --l2=0.0 --dropout=0.5

# STABLE! > 76.3% validation score
# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=64 --batch-size=512 --encode --lr=0.001 --l2=0.0 --dropout=0.0

# STABLE! > 76.3% validation score (infering vocab from training data)
# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=64 --batch-size=512 --encode --lr=0.001 --l2=0.0 --dropout=0.0



# STABLE! > 78% validation score (infering vocab from training data)
# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=64 --batch-size=128 --lr=0.001 --l2=0.0 --dropout=0.75


# Dropout massively slows training...
# python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=128 --batch-size=128 --lr=0.001 --blocks='2,2,2,2' --channels='64,128,256,512' --final-pool
python vdcnn.py --gpus=0 --data=../../finn-dl/MODEL/atb_model_41/strat_split/data/ --sequence-length=128 --batch-size=128 --lr=0.0001 --blocks='5,5,2,2' --channels='64,128,256,512' --encode



# This should achieve ~ 91.27% validation accuracy
# python vdcnn.py --gpus=0 --sequence-length=128 --batch-size=128 --lr=0.001 --blocks='5,5,2,2' --channels='64,128,256,512' --final-pool
