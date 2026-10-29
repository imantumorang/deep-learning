# Optimization

Same deep network as the [previous milestone](../deep-neural-network/)
(`[12288, 30, 20, 10, 1]`), first trained faster, then regularized. There's one notebook
per post, and each one only has the code that post had, so the model grows from one
notebook to the next.

| Notebook | Post | What's in it |
|---|---|---|
| [`01_mini_batch.ipynb`](01_mini_batch.ipynb) | Part 1 (link coming) | Full batch baseline, then mini-batch gradient descent: shuffle, split into batches of 128, update after every batch |
| [`02_optimizers.ipynb`](02_optimizers.ipynb) | Part 2 (link coming) | Momentum, RMSprop and Adam on top of mini-batch, and the learning rate each one needs |
| [`03_regularization.ipynb`](03_regularization.ipynb) | Part 3 (link coming) | L2, L1 and dropout to close the gap between train and test accuracy, with the lambda and keep_prob sweeps |
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
