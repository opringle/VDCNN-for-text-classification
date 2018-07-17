# VDCNN-for-text-classification
 
Following state of the art results using [deep cnn's for text classification](https://arxiv.org/pdf/1606.01781.pdf) this repo investigates drawing from the inception architecture for the same task.

## ToDo

- [ ] Get Sagemaker hyperopt job running locally, on cpu with small models
- [ ] Remove pip install in code solution
    
# Problems

- [ ] [Cannot retrieve role ARN string using sagemaker SDK locally on my machine](https://github.com/aws/sagemaker-python-sdk/issues/300):

`botocore.errorfactory.NoSuchEntityException: An error occurred (NoSuchEntity) when calling the GetRole operation: The user with name oliver_pringle cannot be found.
` 

    - This method may only be designed for use in instances...

- [ ] Cannot stage code in S3 when trying to run locally (likely related to above):

`botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the PutObject operation: Access Denied
`

- [ ] When instance training, encoding errors occur in preprocessor

`UnicodeEncodeError: 'ascii' codec can't encode character '\U0001f383' in position 28: ordinal not in range(128)
`
