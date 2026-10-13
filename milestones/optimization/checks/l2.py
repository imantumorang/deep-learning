"""Checks for the L2-regularized cost.

    J = cross_entropy(AL, Y) + (lambd / 2m) * sum over all layers of ||W||^2
"""

import numpy as np

from .fixtures import deep_network, tiny_network
from .runner import Suite, close

l2_cost = Suite(
    "calculate_cost_with_regularization",
    needs=("cost_with_l2", "plain_cost"),
    formula="J = cross_entropy(AL, Y) + (lambd / 2m) * sum over all layers of ||W||^2",
)


def expected_l2_term(parameters, lambd, m):
    """The oracle: (lambd / 2m) * sum of squares of every W, biases excluded.

    Deliberately a plain per-layer loop that sums each W on its own, so it agrees
    with the formula rather than with any particular way of coding it up.
    """
    L = len(parameters) // 2
    sum_of_squares = 0.0
    for l in range(1, L + 1):
        sum_of_squares += float(np.sum(parameters["W" + str(l)] ** 2))
    return (lambd / (2 * m)) * sum_of_squares


@l2_cost.check("equals the plain cross-entropy when lambd = 0")
def lambd_zero_is_plain_cost(cost_with_l2, plain_cost):
    AL, Y, parameters = deep_network()

    got = cost_with_l2(AL, Y, parameters, 0)
    expected = plain_cost(AL, Y)

    assert close(got, expected), f"expected {expected!r}, got {got!r}"
    return f"{got:.12f} == unregularized cost"


@l2_cost.check("matches a hand-computed 2-layer value")
def matches_hand_computed_value(cost_with_l2):
    AL, Y, parameters = tiny_network()

    # cross entropy: -(1/2) * (ln 0.5 + ln 0.5) = ln 2
    # L2 term:       (0.5 / (2 * 2)) * 32       = 4.0
    expected = np.log(2) + 4.0
    got = cost_with_l2(AL, Y, parameters, 0.5)

    assert close(got, expected), f"expected {expected!r}, got {got!r}"
    return f"{got:.12f} == ln(2) + 4.0"


@l2_cost.check("penalizes every layer, not just some")
def penalizes_every_layer(cost_with_l2, plain_cost):
    """Catches a penalty that silently covers only part of the network."""
    AL, Y, parameters = deep_network()
    lambd = 0.7
    m = Y.shape[1]

    got = cost_with_l2(AL, Y, parameters, lambd)
    expected = plain_cost(AL, Y) + expected_l2_term(parameters, lambd, m)

    assert close(got, expected), f"expected {expected!r}, got {got!r}"
    return f"{got:.12f} over all 3 layers"


@l2_cost.check("scales linearly with lambd")
def scales_linearly_with_lambd(cost_with_l2):
    AL, Y, parameters = deep_network()

    baseline = cost_with_l2(AL, Y, parameters, 0)
    single = cost_with_l2(AL, Y, parameters, 0.3)
    double = cost_with_l2(AL, Y, parameters, 0.6)

    assert close(double - baseline, 2 * (single - baseline)), (
        f"penalty at lambd=0.6 is {double - baseline!r}, "
        f"expected 2x the {single - baseline!r} at lambd=0.3"
    )
    return f"penalty {single - baseline:.9f} -> {double - baseline:.9f}"


@l2_cost.check("divides the penalty by m, the example count")
def divides_by_number_of_examples(cost_with_l2, plain_cost):
    """Same weights, twice the examples -> half the penalty.

    Pins m to the batch size, not to a layer width or a hard-coded constant.
    """
    AL, Y, parameters = deep_network()
    lambd = 0.7

    wide_AL = np.concatenate([AL, AL], axis=1)  # m = 10
    wide_Y = np.concatenate([Y, Y], axis=1)

    narrow = cost_with_l2(AL, Y, parameters, lambd) - plain_cost(AL, Y)
    wide = cost_with_l2(wide_AL, wide_Y, parameters, lambd) - plain_cost(wide_AL, wide_Y)

    assert close(wide, narrow / 2), (
        f"penalty at m=5 is {narrow!r}, at m=10 it is {wide!r}, expected {narrow / 2!r}"
    )
    return f"m=5 -> {narrow:.9f}, m=10 -> {wide:.9f}"


@l2_cost.check("leaves the biases out of the penalty")
def ignores_biases(cost_with_l2):
    AL, Y, parameters = deep_network()
    lambd = 0.7

    before = cost_with_l2(AL, Y, parameters, lambd)
    for l in range(1, len(parameters) // 2 + 1):
        parameters["b" + str(l)] = np.full_like(parameters["b" + str(l)], 100.0)
    after = cost_with_l2(AL, Y, parameters, lambd)

    assert close(after, before), (
        f"cost moved from {before!r} to {after!r} after changing only the biases"
    )
    return f"{after:.12f} unchanged by b = 100"


@l2_cost.check("does not mutate the parameters")
def does_not_mutate_parameters(cost_with_l2):
    AL, Y, parameters = deep_network()
    snapshot = {key: value.copy() for key, value in parameters.items()}

    cost_with_l2(AL, Y, parameters, 0.7)

    for key, original in snapshot.items():
        assert np.array_equal(parameters[key], original), f"{key} was modified in place"
    return "parameters untouched"


@l2_cost.check("returns a scalar")
def returns_a_scalar(cost_with_l2):
    """Keeps the cost printable and plottable, like calculate_cost already is."""
    AL, Y, parameters = deep_network()

    got = cost_with_l2(AL, Y, parameters, 0.7)

    assert np.ndim(got) == 0, f"expected a 0-d scalar, got shape {np.shape(got)}"
    return f"{type(got).__name__}, ndim 0"


def check_l2_cost(
    calculate_cost_with_regularization, calculate_cost, raise_on_fail=True
):
    """Check an L2-regularized cost function against the formula.

    `calculate_cost` is the unregularized reference, so the checks can isolate the
    penalty term.
    """
    return l2_cost.run(
        raise_on_fail=raise_on_fail,
        cost_with_l2=calculate_cost_with_regularization,
        plain_cost=calculate_cost,
    )
