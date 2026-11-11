"""The discrete next-state check reads a target's law only where the source routes.

A per-target law `next_<state>` for target `A` is consumed only at cells where
the regime transition gives `A` positive probability. The check therefore leaves
a code off `A`'s grid unchecked at a cell that sends no mass to `A`, and still
refuses one at a cell that does.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    Model,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import InvalidStateTransitionError
from lcm.regime import Regime as UserRegime
from lcm.typing import DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class _Status:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    therapy: ScalarInt
    no_therapy: ScalarInt


def _next_regime(*, status: DiscreteState) -> FloatND:
    """Send `high` entirely to `no_therapy`; split `low` evenly."""
    to_therapy = jnp.where(status == _Status.high, 0.0, 0.5)
    return jnp.stack([jnp.asarray(0.0), to_therapy, 1.0 - to_therapy])


def _off_grid_where_unrouted(*, status: DiscreteState) -> DiscreteState:
    """Leave the grid at `high`, the cell that sends no mass to `therapy`."""
    return jnp.where(status == _Status.high, -1, status)


def _off_grid_where_routed(*, status: DiscreteState) -> DiscreteState:
    """Leave the grid at `low`, a cell that sends half its mass to `therapy`."""
    return jnp.where(status == _Status.low, -1, status)


def _keep(*, status: DiscreteState) -> DiscreteState:
    return status


def _utility(*, status: DiscreteState) -> FloatND:
    return jnp.asarray(status, dtype=float)


def _model(therapy_law) -> Model:
    status = DiscreteGrid(category_class=_Status)
    source = UserRegime(
        states={"status": status},
        state_transitions={"status": {"therapy": therapy_law, "no_therapy": _keep}},
        functions={"utility": _utility},
    )
    target = UserRegime(states={"status": status}, functions={"utility": _utility})
    return Model(
        regimes={"source": source, "therapy": target, "no_therapy": target},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                targets={"therapy": (0,), "no_therapy": (0,)},
                law=StochasticTransition(func=_next_regime),
            )
        },
    )


_PARAMS = {"discount_factor": 0.95}


def test_solve_skips_off_grid_code_where_target_has_no_mass() -> None:
    """A law off its grid only where its target gets no mass solves."""
    _model(_off_grid_where_unrouted).solve(
        params=_PARAMS, log_level="off", runtime_checks=True
    )


def test_simulate_skips_off_grid_code_where_target_has_no_mass() -> None:
    """A law off its grid only where its target gets no mass simulates."""
    _model(_off_grid_where_unrouted).simulate(
        params=_PARAMS,
        log_level="off",
        runtime_checks=True,
        initial_conditions={
            "status": jnp.array([_Status.low, _Status.high]),
            "age": jnp.zeros(2),
            "regime_id": jnp.array([_RegimeId.source] * 2),
        },
    )


def test_solve_refuses_off_grid_code_where_target_has_mass() -> None:
    """A law off its grid where its target gets mass still raises."""
    with pytest.raises(InvalidStateTransitionError, match="next_status"):
        _model(_off_grid_where_routed).solve(
            params=_PARAMS, log_level="off", runtime_checks=True
        )
