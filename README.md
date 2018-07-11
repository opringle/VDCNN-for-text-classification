# VDCNN-for-text-classification

Implementing this paper in MXNet: https://arxiv.org/pdf/1606.01781.pdf

## Modifications

1. Average pooling with dropout before output layer replaces size 2048 fully connected layers
2. Alphabet distinguishes between unknown, padded and space characters
3. Depth increased

## Running the code

To retrain each model, run the corresponding shell script. eg:

`$ bash train_ag_news.sh`

## Results

Best reported in paper / my model

|                                         | imdb |       ag_news  |     yahoo_answer |
|:---------------------------------------:|:----:|:--------------:|:-----------------:|
|VDCNN (17 layers, avg-pooling + dropout) |      | 91.33 / -      | -/                |

## ToDo

- [ ] K-max-pool implementation screws with learning rate. Using mxnet's native pooling fixes issue.
- [ ] Add bucketing to massively reduce training time.
- [ ] Benchmark on AG news dataset