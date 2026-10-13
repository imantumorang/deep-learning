"""Notebook-runnable checks for the hyperparameter-optimization experiments.

Each suite takes your notebook functions as arguments, so a check never goes stale
when you edit the cell. Every check runs even if an earlier one fails, each prints
PASS/FAIL with expected vs got, and an AssertionError is raised at the end if
anything failed (so a bad cell goes red). Pass `raise_on_fail=False` to just read
the table.

    from checks import check_l2_cost

    check_l2_cost(calculate_cost_with_regularization, calculate_cost)

Or check everything implemented so far in one cell — suites whose functions you
have not written yet are reported as skipped, not failed:

    from checks import run_all

    run_all(
        cost_with_l2=calculate_cost_with_regularization,
        plain_cost=calculate_cost,
    )

Layout:

    runner.py     Suite / run_all / close — the generic machinery, topic-agnostic
    fixtures.py   seeded test networks shared by every suite
    l2.py         L2-regularized cost

Adding a suite (adam, dropout, momentum, initialization, mini-batch):

 1. New module, e.g. `adam.py`. Declare the suite with the short argument names its
    checks will use:

        adam = Suite(
            "update_parameters_with_adam",
            needs=("update_with_adam", "initialize_adam"),
            formula="...",
        )

 2. Write each check as a function that takes the names it needs, asserts, and
    returns a one-line detail string for the report:

        @adam.check("leaves the parameters alone on a zero gradient")
        def zero_gradient_is_a_no_op(update_with_adam):
            _, _, parameters = deep_network()
            ...
            assert close(got, expected), f"expected {expected!r}, got {got!r}"
            return f"{got:.12f} unchanged"

    Reuse `fixtures.py` so every suite checks against the same networks; add a
    fixture there if you need a new shape.

 3. Add a `check_<topic>(...)` wrapper with the notebook's own long function names,
    then export it below.
"""

from .fixtures import deep_network, gradients_for, tiny_network
from .l2 import check_l2_cost, l2_cost
from .runner import SUITES, Suite, close, run_all

__all__ = [
    "check_l2_cost",
    "run_all",
    # for writing new suites
    "Suite",
    "close",
    "SUITES",
    "tiny_network",
    "deep_network",
    "gradients_for",
    "l2_cost",
]
