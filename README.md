# VDCNN-for-text-classification
implementing this paper in MXNet: https://arxiv.org/pdf/1606.01781.pdf

## ToDo

- [ ] Convert k-max-pool to use ndarray
- [ ] Train effectively on AG news dataset (overnight will take 14 hours)
- [ ] Initialize weights according to paper
- [ ] Use their optimizer

## Insights

### Unable to match paper performance on public data

- batch norm could be wrong?

### Unstable validation score

- The validation score during training on all datasets fluctuates between the training score and -20%.
- It is possible the model is learning to classify using the patterns in padded data. This could provide enough signal distinguish some utterances between classes.
- Reducing the sequence length, and therefore the padding, stabilizes the model but forces us to truncate more utterances, loosing valuable signal.

### How to treat padded data and unknown data

- When an unknown character is encountered, we should return a random character embedding.
- Pad characters should have an embedding which is learned in training.

### Representing categorical text data

- Embedding characters makes less intuitive sense than representing them as categorical features.... Humans don't assign meaning to the letter a. We just look for patterns of a's.

### AGNews Dataset vs Finn Banking Data

- AGNews dataset has signficantly longer utterances (average chars = 190). This model is best trained with s=1024. This is equivolent to training an image detection network on higher resolution images. There is more signal in the data and so the model converges easily (13 epochs).
- Finn dataset has 34 characters on average with standard deviation 20.


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
- Small learning rates (0.0001) cause the validation score to be drastically less than the training score. 0.002 works nicely.
- Dropout 0.2 keeping things stable


- At training accuracy >X%, validation loss finally starts to approximate training loss (dropout = X)
- Dropout=0.X ensures validation loss plateaus @Y epochs
- Initializing hyperparams with X ensures robust validation score during training
- [X] Validation accuracy > 70%
- [ ] Validation accuracy > 80%
- [ ] Validation accuracy > 85%
- [ ] Validation accuracy > 90%