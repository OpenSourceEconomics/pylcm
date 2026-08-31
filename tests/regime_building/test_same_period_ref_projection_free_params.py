"""A regime-level reference projection may not introduce free parameters.

A `same_period_refs` projection resolves through the declaring regime's DAG and
receives that regime's `period` / `age`. An argument the regime does not supply
has nowhere to come from: it never reaches `get_params_template()`, so solving
without it reports a missing argument while supplying it is rejected as unknown,
leaving the model with no valid parameter assignment. Rejecting it where it is
declared names the reference, the projection, and the argument.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    LinSpacedGrid,
    ProjectedRegimeValue,
    Regime,
    ValueDependentConstraint,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt

_WEALTH = LinSpacedGrid(start=0.0, stop=10.0, n_points=3)


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    work: ScalarInt


def _next_regime() -> ScalarInt:
    return jnp.int32(0)


def _utility_f(wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return wealth * work


def _utility_m(wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return wealth * (1.0 - work)


def _next_wealth(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _participation_f(Q_f: FloatND, outside_option_f: FloatND) -> BoolND:
    return Q_f >= outside_option_f


def _split_wealth(wealth: ContinuousState) -> ContinuousState:
    """A legitimate projection: reads only a state the regime declares."""
    return 0.5 * wealth


def _halved_wealth(wealth: ContinuousState, divorce_cost: float) -> ContinuousState:
    """The defect: `divorce_cost` is supplied by nothing in the regime."""
    return 0.5 * wealth - divorce_cost


def _make_regime(*, projection) -> Regime:
    return Regime(
        transition=_next_regime,
        states={"wealth": _WEALTH},
        state_transitions={"wealth": _next_wealth},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
        constraints={
            "participation_f": ValueDependentConstraint(
                predicate=_participation_f,
                references={
                    "outside_option_f": ProjectedRegimeValue(
                        regime="single_f", projection={"wealth": projection}
                    )
                },
            )
        },
    )


def test_projection_with_an_unsupplied_argument_is_rejected() -> None:
    """The error names the reference, the projection, and the argument."""
    with pytest.raises(RegimeInitializationError) as excinfo:
        _make_regime(projection=_halved_wealth)

    message = str(excinfo.value)
    assert "outside_option_f" in message
    assert "divorce_cost" in message


def test_projection_reading_only_declared_names_is_accepted() -> None:
    """A projection over the regime's own states is the supported case."""
    regime = _make_regime(projection=_split_wealth)

    assert regime.same_period_refs["outside_option_f"].regime == "single_f"
