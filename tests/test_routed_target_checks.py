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
    Phased,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    InvalidStateTransitionError,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import DiscreteState, FloatND, IntND, ScalarInt, UserFunction


@categorical(ordered=False)
class _Status:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    therapy: ScalarInt
    no_therapy: ScalarInt


def _therapy_probability(*, status: DiscreteState) -> FloatND:
    """Send `high` entirely to `no_therapy`; split `low` evenly."""
    return jnp.where(status == _Status.high, 0.0, 0.5)


def _to_therapy(*, therapy_probability: FloatND) -> FloatND:
    """Read the probability through an ordinary intermediate DAG helper."""
    return therapy_probability


def _next_regime(*, to_therapy: FloatND) -> FloatND:
    return jnp.stack([jnp.asarray(0.0), to_therapy, 1.0 - to_therapy])


def _realized_therapy_probability(*, status: DiscreteState) -> FloatND:
    return jnp.where(status == _Status.high, 0.5, 0.5)


def _invalid_realized_therapy_probability(*, status: DiscreteState) -> FloatND:
    return jnp.where(status == _Status.high, 1.2, 0.5)


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


def _model(
    *,
    therapy_law: UserFunction,
    therapy_probability: UserFunction | Phased[UserFunction, UserFunction] = (
        _therapy_probability
    ),
) -> Model:
    status = DiscreteGrid(category_class=_Status)
    source = UserRegime(
        states={"status": status},
        state_transitions={"status": {"therapy": therapy_law, "no_therapy": _keep}},
        functions={
            "utility": _utility,
            "therapy_probability": therapy_probability,
            "to_therapy": _to_therapy,
        },
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


def _initial_conditions() -> dict[str, FloatND | IntND]:
    return {
        "status": jnp.array([_Status.low, _Status.high], dtype=jnp.int32),
        "age": jnp.zeros(2),
        "regime_id": jnp.full(2, _RegimeId.source, dtype=jnp.int32),
    }


def test_solve_skips_off_grid_code_where_target_has_no_mass() -> None:
    """A law off its grid only where its target gets no mass solves."""
    _model(therapy_law=_off_grid_where_unrouted).solve(
        params=_PARAMS, log_level="off", runtime_checks=True
    )


def test_simulate_skips_off_grid_code_where_target_has_no_mass() -> None:
    """A law off its grid only where its target gets no mass simulates."""
    _model(therapy_law=_off_grid_where_unrouted).simulate(
        params=_PARAMS,
        log_level="off",
        runtime_checks=True,
        initial_conditions=_initial_conditions(),
    )


def test_solve_refuses_off_grid_code_where_target_has_mass() -> None:
    """A law off its grid where its target gets mass still raises."""
    with pytest.raises(InvalidStateTransitionError, match="next_status"):
        _model(therapy_law=_off_grid_where_routed).solve(
            params=_PARAMS, log_level="off", runtime_checks=True
        )


def test_simulate_validates_the_realized_routing_at_the_off_grid_cell() -> None:
    """A target live only under the realized helper reports its off-grid cell."""
    model = _model(
        therapy_law=_off_grid_where_unrouted,
        therapy_probability=Phased(
            solve=_therapy_probability, simulate=_realized_therapy_probability
        ),
    )

    with pytest.raises(InvalidStateTransitionError) as excinfo:
        model.validate_transitions(params=_PARAMS)

    assert "next_status" in str(excinfo.value)
    assert "{'status': 'high'}" in str(excinfo.value)


def test_simulate_refuses_invalid_probabilities_from_a_phased_helper() -> None:
    """A realized probability outside zero to one fails before simulation."""
    model = _model(
        therapy_law=_keep,
        therapy_probability=Phased(
            solve=_therapy_probability, simulate=_invalid_realized_therapy_probability
        ),
    )

    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.validate_transitions(params=_PARAMS)


def test_simulate_accepts_valid_probabilities_from_a_phased_helper() -> None:
    """Valid realized routing retains the complete invented population and ages."""
    result = _model(
        therapy_law=_keep,
        therapy_probability=Phased(
            solve=_therapy_probability, simulate=_realized_therapy_probability
        ),
    ).simulate(
        params=_PARAMS,
        initial_conditions=_initial_conditions(),
        log_level="off",
        runtime_checks=True,
    )

    assert result.n_subjects == 2
    assert result.n_periods == 2
