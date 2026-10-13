"""Deterministic test networks, shared by every check suite.

Seeded, so a check that fails fails the same way twice. Add to this file when a new
suite needs a shape the existing fixtures do not cover — the point is that Adam,
dropout and L2 all check against the same networks, so a failure means the function
is wrong rather than the inputs being different.

Every fixture returns fresh arrays, so a check that mutates them cannot affect the
next one.
"""

import numpy as np


def tiny_network():
    """A 2-2-1 network with numbers picked so results are hand-checkable.

    sum of W^2 = (1 + 4 + 9 + 16) + (1 + 1) = 32, over m = 2 examples.
    """
    AL = np.array([[0.5, 0.5]])
    Y = np.array([[1, 0]])
    parameters = {
        "W1": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "b1": np.zeros((2, 1)),
        "W2": np.array([[1.0, 1.0]]),
        "b2": np.zeros((1, 1)),
    }
    return AL, Y, parameters


def deep_network():
    """A 4-3-2-1 network: every W has a different shape, m = 5 examples.

    The differing shapes are the point — they catch anything that assumes the
    weights can be stacked into one array.
    """
    rng = np.random.default_rng(42)
    AL = rng.uniform(0.05, 0.95, size=(1, 5))
    Y = np.array([[1, 0, 1, 1, 0]])
    parameters = {
        "W1": rng.normal(size=(3, 4)),
        "b1": np.zeros((3, 1)),
        "W2": rng.normal(size=(2, 3)),
        "b2": np.zeros((2, 1)),
        "W3": rng.normal(size=(1, 2)),
        "b3": np.zeros((1, 1)),
    }
    return AL, Y, parameters


def gradients_for(parameters, scale=1.0):
    """Deterministic `grads` matching a parameter dict, for update-rule checks.

    Shaped the way `backward_pass` returns them (`dW1`, `db1`, ...), so suites for
    update_parameter / momentum / Adam can feed a known gradient and assert on where
    the weights land.
    """
    rng = np.random.default_rng(7)
    grads = {}
    for l in range(1, len(parameters) // 2 + 1):
        grads["dW" + str(l)] = scale * rng.normal(size=parameters["W" + str(l)].shape)
        grads["db" + str(l)] = scale * rng.normal(size=parameters["b" + str(l)].shape)
    return grads
