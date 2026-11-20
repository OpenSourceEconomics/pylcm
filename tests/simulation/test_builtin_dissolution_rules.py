"""Built-in dissolution rules differ when exactly one member wants to leave."""

from collections.abc import Callable

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
    MutualConsent,
    ParetoObjective,
    Regime,
    Unilateral,
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


def _model(
    *,
    rule: ExitRule,
    wife: Callable[..., FloatND] = _wife,
    husband: Callable[..., FloatND] = _husband,
    objective: ParetoObjective | None = None,
) -> Model:
    wealth = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
    return Model(
        regimes={
            "couple": Regime(
                states={"wealth": wealth},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": wife, "m": husband}, objective=objective
                    )
                },
                exit=Exit(
                    rule=rule,
                    targets={
                        member: ExitTarget(
                            regime=f"single_{member}",
                            projection={"wealth": fixed_transition("wealth")},
                        )
                        for member in ("f", "m")
                    },
                ),
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


def _simulated_regimes(*, model: Model) -> list[str]:
    solution = model.solve(params={}, retention=ResultRetention.VALUES, log_level="off")
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "wealth": jnp.asarray([1.0, 2.0]),
            "regime_id": jnp.full(2, _Household.couple, dtype=jnp.int32),
            "own_stakeholder": jnp.full(
                2, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        },
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    return list(frame.sort_values("subject_id")["regime_name"])


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (Unilateral(), ["couple", "couple"]),
        (Unilateral(tie="leave"), ["single_f", "single_f"]),
        (MutualConsent(), ["couple", "couple"]),
        (MutualConsent(tie="leave"), ["single_f", "single_f"]),
    ],
)
def test_tie_setting_decides_a_member_indifferent_to_leaving(
    *, rule: ExitRule, expected: list[str]
) -> None:
    """With every member exactly indifferent, `tie` alone decides whether they stay."""
    model = _model(rule=rule, wife=_outside, husband=_outside)
    assert _simulated_regimes(model=model) == expected


def test_a_member_with_zero_pareto_weight_keeps_the_veto() -> None:
    """The wife has no weight in the household choice, yet her leaving ends it."""
    model = _model(
        rule=Unilateral(),
        wife=_husband,
        husband=_wife,
        objective=ParetoObjective(weights={"f": 0.0, "m": 1.0}),
    )
    assert _simulated_regimes(model=model) == ["single_f", "single_f"]


@pytest.mark.parametrize("rule", [Unilateral(), MutualConsent()])
def test_builtin_dissolution_rule_prices_and_settles_original_members(
    *, rule: ExitRule
) -> None:
    """The husband can leave alone, but the wife vetoes a mutual dissolution."""
    model = _model(rule=rule)
    solution = model.solve(params={}, retention=ResultRetention.VALUES, log_level="off")
    expected_full = (
        [[2.0, 2.0], [4.0, 4.0]]
        if isinstance(rule, Unilateral)
        else [[3.0, 1.0], [6.0, 2.0]]
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
    if isinstance(rule, Unilateral):
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
