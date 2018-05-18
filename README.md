# VDCNN-for-text-classification
implementing this paper in MXNet: https://arxiv.org/pdf/1606.01781.pdf

- Do not use skip connections
- Do not use k-max pooling
- Added dropout between final 2 layers


## Insights

### Time to train on AG News dataset

- As the input sequence gets shorter the number of parameters in the network (with final layer max pooling) drastically reduces.
- This is because the output from the final max pooling layer (just before the fully connected layers) is smaller. The fully connected layers contribute most of the parameters in the network.
- As a result, on the AG News dataset, at sequence length *s=256* we use 2.067Gb of Gpu memory & take 59 seconds to compute a single epoch. At *s=1024* we use ~ 7.3Gb of Gpu memory & take minutes to compute an epoch. This is with a Tesla K80 GPU.

### Performance on AG news

- Easily reaches 85% validation accuracy with sequence length = 128 and 20% dropout probability

### Time to train on our data

- Sequence length 256 is sufficient
- This results in 2.1Gb of Gpu memory consumption
- 29s per epoch on atb data, consisting of 41854 training utterances

### Performance

- However, validation score sucks!