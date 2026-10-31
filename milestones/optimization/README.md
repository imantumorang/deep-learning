# Optimization

Same deep network as the [previous milestone](../deep-neural-network/)
(`[12288, 30, 20, 10, 1]`), first trained faster, then regularized. There's one notebook
per post, and each one only has the code that post had, so the model grows from one
notebook to the next.

| Notebook | Post | What's in it |
|---|---|---|
| [`01_mini_batch.ipynb`](01_mini_batch.ipynb) | [Speeding up the Learning of the Magic Box](https://medium.easyread.co/deep-learning-series-speeding-up-the-learning-of-the-magic-box-43ec5e13d6c3) | Full batch baseline, then mini-batch gradient descent: shuffle, split into batches of 128, update after every batch |
| [`02_optimizers.ipynb`](02_optimizers.ipynb) | [Optimization Algorithm, Reducing the Noise](https://medium.easyread.co/deep-learning-series-optimization-algorithm-reducing-the-noise-82082ed3b670) | Momentum, RMSprop and Adam on top of mini-batch, and the learning rate each one needs |
| [`03_regularization.ipynb`](03_regularization.ipynb) | [Bring Them Back to Normal with Regularization](https://medium.easyread.co/deep-learning-series-bring-them-back-to-normal-with-regularization-b5dc6db465ad) | L2, L1 and dropout to close the gap between train and test accuracy, with the lambda and keep_prob sweeps |
| [`minibatches_refactored.ipynb`](minibatches_refactored.ipynb) | | The final model from part 3, cleaned up. Dropout lives inside the normal forward and backward pass, L1/L2 are add-ons to the cost and gradients. Trains exactly like the part 3 model |

`dnn.py` has the building blocks from the previous milestone (layers, activations,
caches), `utils.py` loads the dataset and the test images.

## Running it

Dataset download and environment setup are in the [root README](../../README.md).

Each notebook runs on its own, top to bottom. They're slow, here's how long a full run
took on my laptop:

| Notebook | Time |
|---|---|
| `01_mini_batch.ipynb` | ~35 min |
| `02_optimizers.ipynb` | ~55 min |
| `03_regularization.ipynb` | ~2 h 45 min, most of it the sweeps |
| `minibatches_refactored.ipynb` | ~1 h 20 min |
