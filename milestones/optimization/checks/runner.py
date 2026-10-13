"""Generic machinery for notebook-runnable check suites.

Nothing in here knows about L2, Adam, dropout or any other topic — the sibling
modules own those. This file only knows how to register a group of checks and run
them against functions handed in from the notebook.

A `Suite` is a named group of checks plus the notebook functions it needs. Each
check declares the functions it uses as parameters, and the runner passes only
those, so a check never has to accept arguments it ignores.
"""

import inspect

import numpy as np

TOLERANCE = 1e-12

# Every Suite registers itself here at import time, so `run_all` can find them.
SUITES = []


def close(got, expected):
    """The float comparison every check uses."""
    return bool(np.isclose(got, expected, rtol=TOLERANCE, atol=0))


class Suite:
    """A named group of checks for one function (or one small family of them).

        adam = Suite(
            "update_parameters_with_adam",
            needs=("update_with_adam", "initialize_adam"),
            formula="m = b1*m + (1-b1)*dW ;  v = b2*v + (1-b2)*dW^2 ;  ...",
        )

        @adam.check("leaves the parameters untouched on a zero gradient")
        def zero_gradient_is_a_no_op(update_with_adam):
            ...
            assert close(got, expected), f"expected {expected!r}, got {got!r}"
            return f"{got:.12f} unchanged"
    """

    def __init__(self, name, needs, formula=None):
        self.name = name
        self.needs = tuple(needs)
        self.formula = formula
        self.checks = []
        SUITES.append(self)

    def check(self, description):
        """Register one check.

        The decorated function asserts what it wants to be true and returns a short
        detail string for the report. Its parameters must be names from this suite's
        `needs` — a typo raises here, at import time, rather than mid-run.
        """

        def register(fn):
            unknown = [p for p in inspect.signature(fn).parameters if p not in self.needs]
            if unknown:
                raise TypeError(
                    f"check `{fn.__name__}` asks for {', '.join(unknown)}, which "
                    f"{self.name} does not declare. Its needs are: {', '.join(self.needs)}."
                )
            self.checks.append((description, fn))
            return fn

        return register

    def run(self, raise_on_fail=True, **functions):
        """Run every check in this suite against the given notebook functions."""
        missing = [name for name in self.needs if name not in functions]
        if missing:
            raise TypeError(
                f"{self.name} needs {', '.join(missing)} — pass them in as keyword "
                f"arguments. Full list: {', '.join(self.needs)}."
            )

        print(f"{self.name} — {len(self.checks)} checks")
        if self.formula:
            print(f"  {self.formula}")
        print()

        failures = []
        for description, fn in self.checks:
            wanted = inspect.signature(fn).parameters
            arguments = {k: v for k, v in functions.items() if k in wanted}
            try:
                detail = fn(**arguments)
            except AssertionError as e:
                failures.append(description)
                print(f"  FAIL  {description}\n          {e}")
            except Exception as e:
                failures.append(description)
                print(f"  ERROR {description}\n          {type(e).__name__}: {e}")
            else:
                print(f"  PASS  {description}\n          {detail}")

        passed = len(self.checks) - len(failures)
        print(f"\n{passed} passed, {len(failures)} failed")

        if failures and raise_on_fail:
            raise AssertionError(
                f"{self.name}: {len(failures)} of {len(self.checks)} checks failed: "
                + "; ".join(failures)
            )
        return not failures


def run_all(raise_on_fail=True, **functions):
    """Run every suite whose needed functions are all present in `functions`.

    Lets one cell check everything implemented so far — suites whose functions you
    have not written yet are reported as skipped instead of failing:

        run_all(
            cost_with_l2=calculate_cost_with_regularization,
            plain_cost=calculate_cost,
        )
    """
    ran, skipped, failed = [], [], []

    for suite in SUITES:
        if any(name not in functions for name in suite.needs):
            skipped.append(suite)
            continue
        ran.append(suite)
        if not suite.run(raise_on_fail=False, **functions):
            failed.append(suite)
        print()

    for suite in skipped:
        missing = [name for name in suite.needs if name not in functions]
        print(f"{suite.name} — skipped, not given: {', '.join(missing)}")

    print(f"\n{len(ran) - len(failed)} of {len(ran)} suites passed, {len(skipped)} skipped")

    if failed and raise_on_fail:
        raise AssertionError("suites with failures: " + ", ".join(s.name for s in failed))
    return not failed
