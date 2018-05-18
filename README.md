# VDCNN-for-text-classification
implementing this paper in MXNet: https://arxiv.org/pdf/1606.01781.pdf

- Do not use skip connections
- Do not use k-max pooling
- Added dropout between final 2 layers

## Insights

### Training compared to word-CNN model

- Validation score is much more unstable
- Model can quickly reach 60% training accuracy, but then learns slowly after that. Makes sense we have to learn  hierarchical features.


### Time to train

- As the input sequence gets shorter the number of parameters in the network (with final layer max pooling) drastically reduces.
- This is because the input to the fully connected layer changes in size. The fully connected layers contribute most of the parameters in the network.
- In the paper they use k-max pooling to ensure this input is always small. I'm using a pooling kernel, so will need to careful not to let this input get too large.
- The result is that on the Atb data, at sequence length *s=256* we use 2.57Gb of Gpu memory & take 32.6 seconds to compute a single epoch. At *s=128* we use 1.46Gb of Gpu memory & take 16.7 seconds to compute an epoch. This is with a Tesla K80 GPU.

### Performance on AG news

- Easily reaches 85% validation accuracy with sequence length = 128 and 20% dropout probability

### Performance on Atb model 41

- Using s=128 ensures ~ 0.05% of training data is sliced & trains at 17s per epoch
- Small learning rates (0.0001) cause the validation score to be drastically less than the training score. 0.001 works nicely.
- At training accuracy >80%, validation loss finally starts to approximate training loss (dropout = 0.2)
- Dropout=0.X ensures validation loss plateaus @Y epochs
- Initializing hyperparams with X ensures robust validation score during training
- [X] Validation accuracy > 70%
- [ ] Validation accuracy > 80%
- [ ] Validation accuracy > 85%
- [ ] Validation accuracy > 90%