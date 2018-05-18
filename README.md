# VDCNN-for-text-classification
implementing this paper in MXNet: https://arxiv.org/pdf/1606.01781.pdf

- Do not use skip connections
- Do not use k-max pooling
- Added dropout between final 2 layers


## Insights

- As the input sequence gets shorter the number of parameters in the network (with final layer max pooling) drastically reduces.
- This is because the output from the final max pooling layer (just before the fully connected layers) is smaller. The fully connected layers contribute most of the parameters in the network.
- As a result at s=256 we ue 2.067Gb of GPU memory & at s=1024 we use ~ 7.3Gb  of GPU memory.
- Interestingly, this does not effect the speed the network is training. Makes sense. The GPU is just multiplying matrices. When multiplying large matrices the GPU will use lots of memory. But the speed at which the GPU multiplies matrices is the same in both cases.
