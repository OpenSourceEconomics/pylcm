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

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from lcm import AgeGrid, LinSpacedGrid, Model, Regime
from lcm.exceptions import (
    GridInitializationError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.interfaces import _build_regime_sharding
from lcm.model import _validate_log_args
from lcm.regime import _default_H
from lcm.simulation.simulate import _compute_starting_periods
from lcm.solution.solve_brute import _log_per_period_stats
from lcm.state_action_space import _validate_all_states_present
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


def test_claw_checks_lcm_state_action_space() -> None:
    """An ill-typed argument to an `lcm.state_action_space` function is rejected.

    `_validate_all_states_present` annotates `provided_states` as a
    `dict[StateName, FloatND | IntND]`. An empty `str` `provided_states`
    yields an empty `set(provided_states)`, which equals an empty
    `required_state_names`, so an un-instrumented call would return cleanly;
    the claw turns the wrong container type into a violation.
    """
    with pytest.raises(BeartypeCallHintViolation):
        _validate_all_states_present(
            provided_states="",  # ty: ignore[invalid-argument-type]
            required_state_names=set(),
        )


def test_claw_checks_lcm_interfaces() -> None:
    """An ill-typed argument to an `lcm.interfaces` function is rejected.

    `_build_regime_sharding` annotates `n_devices` as `int`. With an empty
    `grids` mapping the function returns `None` before `n_devices` is ever
    used, so an un-instrumented call would return cleanly; the claw turns the
    wrong `n_devices` type into a violation.
    """
    with pytest.raises(BeartypeCallHintViolation):
        _build_regime_sharding(
            grids=MappingProxyType({}),
            n_devices="not an int",  # ty: ignore[invalid-argument-type]
        )


def test_claw_checks_lcm_regime() -> None:
    """An ill-typed argument to an `lcm.regime` function is rejected.

    `_default_H` annotates `utility` as `FloatND` (a JAX array). A NumPy array
    would otherwise flow through `utility + discount_factor * E_next_V` and the
    call would return a NumPy array cleanly; the claw turns the wrong array
    library into a violation.
    """
    with pytest.raises(BeartypeCallHintViolation):
        _default_H(
            utility=np.array([1.0]),  # ty: ignore[invalid-argument-type]
            E_next_V=jnp.array([1.0]),
            discount_factor=jnp.array([0.95]),
        )


def test_regime_with_bad_arg_raises_project_exception() -> None:
    """A bad `Regime` argument still raises `RegimeInitializationError`.

    The package claw instruments `lcm.regime`'s private helpers with
    `INTERNAL_CONF`, but the explicit `@beartype(conf=REGIME_CONF)` decorator
    on the `Regime` constructor still wins: a type violation at construction
    surfaces as the project's `RegimeInitializationError`, not beartype's own
    `BeartypeCallHintViolation`.
    """
    with pytest.raises(RegimeInitializationError):
        Regime(
            transition=None,
            states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=3)},
            functions="not a mapping",  # ty: ignore[invalid-argument-type]
        )


def test_claw_checks_lcm_model() -> None:
    """An ill-typed argument to an `lcm.model` function is rejected.

    `_validate_log_args` annotates `log_path` as `str | Path | None`. With
    `log_level="progress"` the function returns before `log_path` is ever
    inspected, so an un-instrumented call would return cleanly; the claw turns
    the wrong `log_path` type into a violation.
    """
    with pytest.raises(BeartypeCallHintViolation):
        _validate_log_args(
            log_level="progress",
            log_path=123,  # ty: ignore[invalid-argument-type]
        )


def test_model_with_bad_arg_raises_project_exception() -> None:
    """A bad `Model` argument still raises `ModelInitializationError`.

    The package claw instruments `lcm.model`'s private helpers with
    `INTERNAL_CONF`, but the explicit `@beartype(conf=MODEL_CONF)` decorator on
    `Model.__init__` still wins: a type violation at construction surfaces as
    the project's `ModelInitializationError`, not beartype's own
    `BeartypeCallHintViolation`.
    """
    with pytest.raises(ModelInitializationError):
        Model(
            ages=AgeGrid(start=25, stop=75, step="Y"),
            regimes="not a mapping",  # ty: ignore[invalid-argument-type]
            regime_id_class=int,
        )


def test_linspaced_grid_with_bad_arg_raises_project_exception() -> None:
    """A bad `LinSpacedGrid` argument still raises `GridInitializationError`.

    The package claw instruments `lcm.grids`'s private helpers with
    `INTERNAL_CONF`, but the explicit `@beartype(conf=GRID_CONF)` decorator on
    `LinSpacedGrid.__init__` still wins: a type violation at construction
    surfaces as the project's `GridInitializationError`, not beartype's own
    `BeartypeCallHintViolation`.
    """
    with pytest.raises(GridInitializationError):
        LinSpacedGrid(
            start="not a number",  # ty: ignore[invalid-argument-type]
            stop=10.0,
            n_points=3,
        )
