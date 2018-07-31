# VDCNN-for-text-classification
 
Following state of the art results using [deep cnn's for text classification](https://arxiv.org/pdf/1606.01781.pdf) this repo investigates drawing from the inception architecture for the same task.

## ToDo

- [ ] Remove pip install in code solution
- [ ] Retrieve ARN with boto instead of cli input
    
# Problems

- [x] When instance training, encoding errors occur in preprocessor

`UnicodeEncodeError: 'ascii' codec can't encode character '\U0001f383' in position 28: ordinal not in range(128)
`

- [ ] Local training works but hyperparameter jobs cannot be run locally..