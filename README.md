# deep-learning

My public notebook for learning deep learning from the ground up. Everything so far is
plain NumPy, because I'm early enough in this that I'd rather write the gradients myself
than have a framework hand them to me. PyTorch and friends come later, once the math
stops being the hard part. If something shows up in a writeup, it shows up as code in
this repo, and vice versa.

## The writeups

I'm publishing a series on Medium alongside the code. Each post explains the theory,
the repo holds the implementation.

1. **[Neural Networks: Backpropagation Concept with Manual Calculation](https://medium.easyread.co/neural-networks-backpropagation-concept-with-manual-calculation-748ac0f53783)**

   Working the forward and backward pass by hand, on paper, before writing any code.

2. **[Deep Learning Series: Building Magic Box with Logistic Regression](https://medium.easyread.co/deep-learning-series-building-magic-box-with-logistic-regression-814b880835d3?sharedUserId=imantumorang)**

   Notes from building the cat-vs-dog classifier in `milestones/logistic-regression`.

More posts coming as I work through the milestones.

## Milestones

| Milestone | What it does | Where |
|---|---|---|
| Logistic regression | Cat vs dog binary classifier, single neuron, trained with plain gradient descent | [`milestones/logistic-regression/catvdog.ipynb`](milestones/logistic-regression/catvdog.ipynb) |

## Getting started

You need [uv](https://docs.astral.sh/uv/). It pulls the Python 3.14 toolchain and the
dependencies for you, so it's the only thing to install yourself.

```bash
git clone git@github.com:imantumorang/deep-learning.git
cd deep-learning
uv sync
```

### 1. Download the dataset

`dataset/` is gitignored, so the images don't come with the clone. Grab the
[Cat and Dog dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog) from
Kaggle. That same link sits in a comment in the notebook's first cell, in case you find
this repo before you find this README.

Unzip it under `dataset/catvdog/` so the paths match what the notebook expects:

```
dataset/catvdog/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/
```

Check the nesting after unzipping, since Kaggle archives often wrap everything in an
extra folder. The notebook walks these directories recursively and reads the class label
off the parent folder name, so `cats/` and `dogs/` have to be the direct parents of the
image files.

### 2. Run the notebook

```bash
uv run jupyter lab
```

Open [`milestones/logistic-regression/catvdog.ipynb`](milestones/logistic-regression/catvdog.ipynb)
and run the cells top to bottom. The first cell scans the dataset and prints a summary,
which doubles as a check that you got the folder layout right. The training cell runs
5,000 full-batch iterations over every image, so that's the slow one.

### 3. Point it at your own photos

The last cells classify a single file from `test-dataset/images/`, which is committed,
so there's something to run against immediately. Drop your own cat or dog picture in
that folder and change `my_image` to its filename.

## License

[MIT](LICENSE)
