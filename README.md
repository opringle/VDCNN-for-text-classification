# VDCNN-for-text-classification
 
- Reproduce state of the art results from [deep cnn's for text classification](https://arxiv.org/pdf/1606.01781.pdf).
- Investigate inception style architecture.
- Explore using large datasets referenced in the paper as transfer learning sources for a different target task.
- Use SageMaker distributed GPU training.

## ToDo

- [ ] Preprocess script: select dataset, downloads from gdrive, creates pandas pickle file and uploads to an s3 bucket of choice
- [ ] Distributed training on sagemaker with yahoo answers data
- [ ] Transfer learning on smaller datasets

## Cleanup

- [ ] Remove pip install in code solution, by referencing a different sagemaker docker image