"""The beartype claw is live on `lcm.solution` and `lcm.simulation`.

These packages sit *behind* the construction perimeter: by the time their
functions run, user input has already been validated by `Model.solve` /
`Model.simulate` and `validate_initial_conditions`. A type violation here
therefore signals an internal pylcm bug, so the claw is configured to raise
beartype's own `BeartypeCallHintViolation` rather than a project exception.

Each test calls an internal function with one argument of the wrong type,
chosen so the call would return cleanly if the function were *not*
instrumented — the violation is what proves the claw is installed.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from lcm import AgeGrid
from lcm.simulation.simulate import _compute_starting_periods
from lcm.solution.solve_brute import _log_per_period_stats
from lcm.utils.error_handling import validate_regime_transition_probs


def test_claw_checks_lcm_simulation() -> None:
    """An ill-typed argument to an `lcm.simulation` function is rejected.

    `_compute_starting_periods` annotates `initial_ages` as `Float1D` (a JAX
    array). A NumPy array would otherwise be accepted by `jnp.searchsorted`
    and the call would return cleanly; the claw turns it into a violation.
    """
    with pytest.raises(BeartypeCallHintViolation):
        _compute_starting_periods(
            initial_ages=np.array([25.0]),  # ty: ignore[invalid-argument-type]
            ages=AgeGrid(start=25, stop=75, step="Y"),
        )


def test_claw_checks_lcm_solution() -> None:
    """An ill-typed argument to an `lcm.solution` function is rejected.

    `_log_per_period_stats` annotates `logger` as `logging.Logger`. With an
    empty `diagnostic_rows` the body never runs, so an un-instrumented call
    would return `None`; the claw turns the bad `logger` into a violation.
    """
    with pytest.raises(BeartypeCallHintViolation):
        _log_per_period_stats(
            logger="not a logger",  # ty: ignore[invalid-argument-type]
            diagnostic_rows=[],
            mins=jnp.array([]),
            maxs=jnp.array([]),
            means=jnp.array([]),
        )


def test_claw_checks_lcm_utils_error_handling() -> None:
    """An ill-typed argument to an `lcm.utils.error_handling` function is rejected.

    `validate_regime_transition_probs` annotates `regime_transition_probs` as
    `MappingProxyType[RegimeName, FloatND]`. A plain `dict` whose values are JAX
    arrays would be accepted by `jnp.stack(list(...))` and the body would run to
    completion; the claw turns the wrong container type into a violation.
    """
    with pytest.raises(BeartypeCallHintViolation):
        validate_regime_transition_probs(
            regime_transition_probs={"working": jnp.array([1.0])},  # ty: ignore[invalid-argument-type]
            active_regimes_next_period=("working",),
            regime_name="working",
            age=50.0,
            next_age=51.0,
        )
