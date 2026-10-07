# Deep neural network

Cat vs dog again, this time with hidden layers. The forward pass, backprop and parameter
updates all work for any number of layers, so the same `dnn_model` runs logistic
regression, a shallow network and a deep one just by changing `layer_dims`.

Writeup: [Deep Learning Series: Magic Box is Moving the Deep Neural Network!](https://medium.easyread.co/deep-learning-series-magic-box-is-moving-the-deep-neural-network-3b736543bf89)

## Results

Three models, same data, 2,500 iterations of full batch gradient descent at the same
learning rate. The two networks hold roughly the same number of parameters, so what
differs between them is how many layers that budget is spread across.

| Model | `layer_dims` | Parameters | Train | Test |
|---|---|---|---|---|
| Logistic regression | `[12288, 1]` | 12,289 | 52.9% | 53.6% |
| Shallow network | `[12288, 30, 1]` | 368,701 | 68.8% | 63.2% |
| Deep network | `[12288, 30, 20, 10, 1]` | 369,511 | 70.6% | 62.8% |

The deep network is the best of the three at fitting the data it was shown and not the
best at anything else, which I'm fairly sure is overfitting. Diagnosing and fixing that
is the next thing on my list.

Five fresh pictures off Google Images, outside both sets: the deep network got all five,
the shallow network missed two cats, logistic regression missed a dog. Far too few
samples to conclude anything, but the ordering disagrees with the test column.

## Caveat on the logistic regression row

It isn't a fair rerun of [the first milestone](../logistic-regression/catvdog.ipynb),
which reached 61.1% test with a zero initialization and 5,000 smaller steps. Here it goes
through the same `initialize_parameters` as the networks, so it inherits He
initialization. With no hidden layer that starts it at a cost of 2.29 instead of 0.69,
saturated before the first update.

## Running it

Dataset download and environment setup are in the [root README](../../README.md).
