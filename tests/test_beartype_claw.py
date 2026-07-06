"""The beartype claw is live on the entire `lcm` package.

The claw uses `INTERNAL_CONF`, so type violations in internal helpers
surface as beartype's own `BeartypeCallHintViolation`. User-facing
constructors (`Model`, `Regime`, `MarkovTransition`, every grid and shock,
`@categorical`, `as_leaf`) carry their own explicit `@beartype(conf=...)`
decorators that map violations to the relevant project exception
(`ModelInitializationError`, `RegimeInitializationError`,
`GridInitializationError`, `InvalidParamsError`); those decorators stack
on top of the claw and win at the user boundary.

Each `test_claw_checks_*` test calls an internal function with one argument
of the wrong type, chosen so the call would return cleanly if the function
were *not* instrumented — the violation is what proves the claw is
installed. Each `test_*_with_bad_arg_raises_project_exception` test
confirms that an ill-typed argument to a public constructor surfaces as
the project exception, not as `BeartypeCallHintViolation`.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from _lcm.engine import _build_regime_sharding
from _lcm.regime_building.max_Q_over_a import get_argmax_and_max_Q_over_a
from _lcm.simulation.simulate import _compute_starting_periods
from _lcm.solution.backward_induction import _log_per_period_stats
from _lcm.state_action_space import _validate_all_states_present
from _lcm.transition_checks import _validate_regime_transition_probs
from lcm import AgeGrid, LinSpacedGrid, Model
from lcm.exceptions import (
    GridInitializationError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.regime import Regime as UserRegime
from lcm.temporal_aggregation import H_linear


def test_claw_checks_lcm_simulation() -> None:
    """Type-violating arguments to internal `_lcm.simulation` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _compute_starting_periods(
            initial_ages=np.array([25.0]),  # ty: ignore[invalid-argument-type]
            ages=AgeGrid(start=25, stop=75, step="Y"),
        )


def test_claw_checks_lcm_solution() -> None:
    """Type-violating arguments to internal `_lcm.solution` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _log_per_period_stats(
            logger="not a logger",  # ty: ignore[invalid-argument-type]
            diagnostic_rows=[],
            mins=jnp.array([]),
            maxs=jnp.array([]),
            means=jnp.array([]),
        )


def test_claw_checks_lcm_transition_checks() -> None:
    """Type-violating arguments to `_lcm.transition_checks` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _validate_regime_transition_probs(
            regime_transition_probs={"working": jnp.array([1.0])},  # ty: ignore[invalid-argument-type]
            active_regimes_next_period=("working",),
            regime_name="working",
            age=50.0,
            next_age=51.0,
        )


def test_claw_checks_lcm_state_action_space() -> None:
    """Type-violating arguments to `_lcm.state_action_space` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _validate_all_states_present(
            provided_states="",  # ty: ignore[invalid-argument-type]
            required_state_names=set(),
        )


def test_claw_checks_lcm_engine() -> None:
    """Type-violating arguments to `_lcm.engine` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _build_regime_sharding(
            grids=MappingProxyType({}),
            n_devices="not an int",  # ty: ignore[invalid-argument-type]
        )


def test_claw_checks_lcm_regime() -> None:
    """Type-violating arguments to `lcm.regime` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        H_linear(
            utility=np.array([1.0]),  # ty: ignore[invalid-argument-type]
            E_next_V=jnp.array([1.0]),
            discount_factor=jnp.array([0.95]),
        )


def test_claw_allows_with_signature_wrapper_over_named_param_function() -> None:
    """A `with_signature` wrapper over a function with named parameters stays
    callable under the package claw.

    `get_argmax_and_max_Q_over_a` returns a `dags.with_signature` wrapper
    around a function whose parameters — `next_regime_to_V_arr` plus the
    `**states_actions_params` it expands — are named explicitly rather than
    being a bare `*args, **kwargs` forwarder. The claw decorates that
    wrapper. The wrapper advertises a permissive forwarder via its
    `__annotations__`, so the claw must enforce nothing against the
    synthetic, annotation-free `__signature__`; otherwise every call fails
    because each parameter is checked against the `inspect.Parameter.empty`
    sentinel.
    """

    def Q_and_F(
        next_regime_to_V_arr: MappingProxyType[str, jnp.ndarray],  # noqa: ARG001
        action: jnp.ndarray,
        state: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return action, action >= state

    argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("action",),
        state_names=("state",),
    )

    argmax, maximum = argmax_and_max_Q_over_a(
        next_regime_to_V_arr=MappingProxyType({"working": jnp.arange(3.0)}),
        action=jnp.array([0.0, 1.0, 2.0]),
        state=jnp.array(0.0),
    )

    assert int(argmax) == 2
    assert float(maximum) == 2.0


def test_regime_with_bad_arg_raises_project_exception() -> None:
    """A bad `Regime` argument surfaces as `RegimeInitializationError`."""
    with pytest.raises(RegimeInitializationError):
        UserRegime(
            transition=None,
            states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=3)},
            functions="not a mapping",  # ty: ignore[invalid-argument-type]
        )


def test_model_with_bad_arg_raises_project_exception() -> None:
    """A bad `Model` argument surfaces as `ModelInitializationError`."""
    with pytest.raises(ModelInitializationError):
        Model(
            ages=AgeGrid(start=25, stop=75, step="Y"),
            regimes="not a mapping",  # ty: ignore[invalid-argument-type]
            regime_id_class=int,
        )


def test_linspaced_grid_with_bad_arg_raises_project_exception() -> None:
    """A bad `LinSpacedGrid` argument surfaces as `GridInitializationError`."""
    with pytest.raises(GridInitializationError):
        LinSpacedGrid(
            start="not a number",  # ty: ignore[invalid-argument-type]
            stop=10.0,
            n_points=3,
        )
