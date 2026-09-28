"""Parameters and laws are lowered only for the problems the starts require.

A `ByAge` case selected only at ages no required problem solves contributes no
parameter, and a regime without any demanded pair reads no parameter at all.
The values at the demanded pairs do not depend on which undemanded cases exist.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import ContinuousState, FloatND, ScalarInt

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _retirement_utility(*, wealth: ContinuousState, bonus: float) -> FloatND:
    return wealth + bonus


def _early_stay(early_rate: float) -> FloatND:
    return jnp.asarray(early_rate)


def _early_die(early_rate: float) -> FloatND:
    return 1 - jnp.asarray(early_rate)


def _late_stay(late_rate: float) -> FloatND:
    return jnp.asarray(late_rate)


def _late_die(late_rate: float) -> FloatND:
    return 1 - jnp.asarray(late_rate)


@categorical(ordered=False)
class LifeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def _model(initial_regimes: Any) -> Model:
    return Model(
        regimes={
            "working": Regime(
                regime_transitions=ByAge(
                    cases={
                        AgeRange(start=25, stop=45): {
                            "working": MarkovTransition(func=_early_stay),
                            "dead": MarkovTransition(func=_early_die),
                        },
                        AgeRange(start=45, stop=65): {
                            "working": MarkovTransition(func=_late_stay),
                            "dead": MarkovTransition(func=_late_die),
                        },
                    },
                    default="dead",
                ),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _utility},
            ),
            "retirement": Regime(
                regime_transitions=ByAge(cases={AgeRange(start=55, stop=75): "dead"}),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _retirement_utility},
            ),
            "dead": Regime(
                regime_transitions=None,
                states={"wealth": _WEALTH},
                functions={"utility": _utility},
            ),
        },
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=LifeId,
        initial_regimes=initial_regimes,
    )


def _leaves(*, tree: Any, prefix: str = "") -> frozenset[str]:
    if not isinstance(tree, dict):
        return frozenset({prefix})
    return frozenset(
        leaf
        for key, value in tree.items()
        for leaf in _leaves(tree=value, prefix=f"{prefix}/{key}")
    )


def _param_names(*, model: Model, regime: str) -> frozenset[str]:
    return frozenset(
        path.rsplit("/", 1)[-1]
        for path in _leaves(tree=model.get_params_template()[regime])
        if path
    )


@pytest.mark.parametrize(
    ("initial_regimes", "regime", "expected"),
    [
        ({25: "working"}, "working", {"discount_factor", "early_rate", "late_rate"}),
        ({55: "working"}, "working", {"discount_factor", "late_rate"}),
        ({55: "working"}, "retirement", set()),
        ({45: "dead"}, "working", set()),
        ({55: "retirement"}, "retirement", {"discount_factor", "bonus"}),
    ],
    ids=["both-cases", "late-case-only", "zero-node", "terminal-root", "late-root"],
)
def test_params_template_holds_only_the_parameters_demand_reads(
    *, initial_regimes: Any, regime: str, expected: set[str]
) -> None:
    """A regime's template lists exactly the parameters its demanded laws read."""
    assert _param_names(model=_model(initial_regimes), regime=regime) == expected


def test_late_root_values_equal_the_first_age_root_values_at_shared_pairs() -> None:
    """Dropping an undemanded case leaves the demanded values bit-identical."""
    full = _model({25: "working"}).solve(
        params={"discount_factor": 0.9, "early_rate": 0.7, "late_rate": 0.8},
        log_level="off",
    )
    late = _model({55: "working"}).solve(
        params={"discount_factor": 0.9, "late_rate": 0.8}, log_level="off"
    )
    for period in (3, 4):
        np.testing.assert_array_equal(
            np.asarray(late.values[period]["working"]),
            np.asarray(full.values[period]["working"]),
        )


def test_simulate_runs_on_a_late_root_with_only_the_demanded_parameters() -> None:
    """A late start simulates with the late case's parameter alone."""
    params = {"discount_factor": 0.9, "late_rate": 0.8}
    model = _model({55: "working"})
    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.asarray([0.0, 1.0]),
            "age": jnp.asarray([55.0, 55.0]),
            "regime_id": jnp.full(2, model.regime_names_to_ids["working"]),
        },
        solution=model.solve(params=params, log_level="off"),
        log_level="off",
        seed=0,
    )
    assert set(result.to_dataframe()["age"]) <= {55, 65, 75}
