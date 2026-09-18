# deep-learning

My public notebook for learning deep learning from the ground up. Every milestone here
is built with NumPy only: no PyTorch, no TensorFlow, no autograd. If the math shows up
in a writeup, it also shows up as code in this repo, and vice versa.

## The writeups

I'm publishing a series on Medium alongside the code. Each post explains the theory,
the repo holds the implementation.

1. **[Neural Networks: Backpropagation Concept with Manual Calculation](https://medium.easyread.co/neural-networks-backpropagation-concept-with-manual-calculation-748ac0f53783)**

   Working the forward and backward pass by hand, on paper, before writing any code.

2. **[Deep Learning Series: Building Magic Box with Logistic Regression](https://medium.easyread.co/814b880835d3)** *(scheduled for 19 Sep 2026)*

   Notes from building the cat-vs-dog classifier in `milestones/logistic-regression`.

More posts coming as I work through the milestones.

## Milestones

| Milestone | What it does | Where |
|---|---|---|
| Logistic regression | Cat vs dog binary classifier, single neuron, trained with plain gradient descent | [`milestones/logistic-regression/catvdog.ipynb`](milestones/logistic-regression/catvdog.ipynb) |

### Logistic regression, in numbers

8,005 training images and 2,023 test images, each resized to 64x64 RGB and flattened
into a 12,288-dimensional column vector. One neuron, sigmoid activation, binary
cross-entropy loss. 5,000 iterations at a learning rate of 0.001 gets to **63.6% train
accuracy and 61.1% test accuracy**.

That's barely better than a coin flip, and that's the point. A single linear boundary
over raw pixels can't separate cats from dogs. The next milestone is where the hidden
layers come in.

## Setup

Uses [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync
uv run jupyter lab
```

## Getting the dataset

`dataset/` is gitignored, so you need to fetch the images yourself. Grab the
[Cat and Dog dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog) from
Kaggle and unpack it so the tree looks like this:

```
dataset/catvdog/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/
```

The handful of images in `test-dataset/images/` are committed. Those are for poking at
the trained model with pictures it has never seen.

## License

[MIT](LICENSE)
