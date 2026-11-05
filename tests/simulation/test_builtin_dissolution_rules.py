"""Built-in dissolution rules differ when exactly one member wants to leave."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    CollectiveUtility,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exits import ExitRule
from lcm.solver_api import ResultRetention
from lcm.typing import ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class _Household:
    couple: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


def _wife(wealth: ContinuousState) -> FloatND:
    return 3.0 * wealth


def _husband(wealth: ContinuousState) -> FloatND:
    return wealth


def _outside(wealth: ContinuousState) -> FloatND:
    return 2.0 * wealth


@pytest.mark.parametrize("rule", ["unilateral", "mutual"])
def test_builtin_dissolution_rule_prices_and_settles_original_members(
    *, rule: ExitRule
) -> None:
    """The husband can leave alone, but the wife vetoes a mutual dissolution."""
    wealth = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
    model = Model(
        regimes={
            "couple": Regime(
                states={"wealth": wealth},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _wife, "m": _husband})
                },
                exits={
                    "divorce": Exit(
                        rule=rule,
                        targets={
                            member: ExitTarget(
                                regime=f"single_{member}",
                                projection={"wealth": fixed_transition("wealth")},
                            )
                            for member in ("f", "m")
                        },
                    )
                },
            ),
            "single_f": Regime(
                states={"wealth": wealth}, functions={"utility": _outside}
            ),
            "single_m": Regime(
                states={"wealth": wealth}, functions={"utility": _outside}
            ),
        },
        n_periods=1,
        edges={},
        regime_id_class=_Household,
        initial_nodes=InitialNodes(by_period={0: "couple"}),
    )
    solution = model.solve(params={}, retention=ResultRetention.VALUES, log_level="off")
    expected_full = (
        [[2.0, 2.0], [4.0, 4.0]] if rule == "unilateral" else [[3.0, 1.0], [6.0, 2.0]]
    )
    np.testing.assert_array_equal(solution.values[0]["couple"], expected_full)
    roles = [model.stakeholder_names_to_ids[member] for member in ("f", "m", "f", "m")]
    frame = (
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(4, dtype=jnp.int32),
                "wealth": jnp.asarray([1.0, 1.0, 2.0, 2.0]),
                "regime_id": jnp.full(4, _Household.couple, dtype=jnp.int32),
                "own_stakeholder": jnp.asarray(roles, dtype=jnp.int32),
            },
            seed=0,
            log_level="off",
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("subject_id")
    )
    np.testing.assert_array_equal(frame["subject_id"], [0, 1, 2, 3])
    np.testing.assert_array_equal(frame["period"], [0, 0, 0, 0])
    np.testing.assert_array_equal(frame["wealth"], [1.0, 1.0, 2.0, 2.0])
    if rule == "unilateral":
        np.testing.assert_array_equal(
            frame["regime_name"], ["single_f", "single_m", "single_f", "single_m"]
        )
        np.testing.assert_array_equal(frame["value"], [2.0, 2.0, 4.0, 4.0])
    else:
        np.testing.assert_array_equal(frame["regime_name"], ["couple"] * 4)
        np.testing.assert_array_equal(frame["own_stakeholder"], roles)
        np.testing.assert_array_equal(frame["value_f"], [3.0, 3.0, 6.0, 6.0])
        np.testing.assert_array_equal(frame["value_m"], [1.0, 1.0, 2.0, 2.0])
        np.testing.assert_array_equal(
            [
                frame["value_f"].iloc[0],
                frame["value_m"].iloc[1],
                frame["value_f"].iloc[2],
                frame["value_m"].iloc[3],
            ],
            [3.0, 1.0, 6.0, 2.0],
        )
