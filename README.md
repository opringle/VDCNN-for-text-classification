# VDCNN-for-text-classification
 
- Reproduce state of the art results from [deep cnn's for text classification](https://arxiv.org/pdf/1606.01781.pdf).
- Investigate inception style architecture.
- Explore using large datasets referenced in the paper as transfer learning sources for a different target task.
- Use SageMaker distributed GPU training.

## ToDo

- [ ] Find a transfer learning dataset.. pretrain on ag_news?
- [ ] Improve hpo.py so it takes default hyperparameters from code

## Cleanup

- [ ] Remove pip install in code solution, by referencing a different sagemaker docker image
