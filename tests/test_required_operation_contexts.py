"""Required operations decide parameters and the domain of probability checks.

- A regime that is only valued (S) and never visited (H) owes the parameters of
  its backward problem, not those of its realized routing; promoting it to a
  physical visit adds them.
- The regime-selection check evaluates each required law on the rows its
  operation evaluates: the period's own grid, the true carried-state axes of a
  realized law, and only the economically feasible action rows.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    ByAge,
    Choose,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.phased import Phased
from lcm.typing import BoolND, ContinuousState, FloatND, IntND, ScalarInt


@categorical(ordered=False)
class _DemandId:
    source: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt
    end: ScalarInt
    other_end: ScalarInt


def _wealth_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _backward_utility(*, wealth: ContinuousState, backward_bonus: float) -> FloatND:
    return wealth + backward_bonus


def _realized_choice(*, realized_rate: float) -> IntND:
    return jnp.where(realized_rate >= 0.5, _DemandId.end, _DemandId.other_end)


def _wealth_regime(*, law: Any, utility: Any = _wealth_utility) -> Regime:
    return Regime(
        regime_transitions=law,
        states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={} if law is None else {"wealth": fixed_transition("wealth")},
        functions={"utility": utility},
    )


def _demand_model(
    *, promote: bool, redundant_root: bool = False, enable_jit: bool = True
) -> Model:
    """Source perceives `perceived` at age 1 but physically enters `realized`.

    Only `perceived`'s realized route reads `realized_rate`; its backward
    utility reads `backward_bonus`.
    """
    roots: dict[object, str] = {0: "source"}
    if promote:
        roots[1] = "perceived"
    if redundant_root:
        roots[2] = "end"
    return Model(
        enable_jit=enable_jit,
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_DemandId,
        initial_regimes=roots,
        regimes={
            "source": _wealth_regime(
                law=ByAge(cases={0: Phased(solve="perceived", simulate="realized")})
            ),
            "perceived": _wealth_regime(
                law=ByAge(
                    cases={
                        1: Phased(
                            solve="end",
                            simulate=Choose(
                                func=_realized_choice, targets=("end", "other_end")
                            ),
                        )
                    }
                ),
                utility=_backward_utility,
            ),
            "realized": _wealth_regime(law=ByAge(cases={1: "end"})),
            "end": _wealth_regime(law=None),
            "other_end": _wealth_regime(law=None),
        },
    )


def _leaf_names(*, tree: Any) -> set[str]:
    if not isinstance(tree, Mapping):
        return set()
    return {
        name
        for key, value in tree.items()
        for name in (_leaf_names(tree=value) if isinstance(value, Mapping) else {key})
    }


def _visited(*, promote: bool) -> frozenset[tuple[int, str]]:
    visited = {(0, "source"), (1, "realized"), (2, "end")}
    if promote:
        visited |= {(1, "perceived"), (2, "other_end")}
    return frozenset(visited)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("redundant_root", [False, True])
@pytest.mark.parametrize("promote", [False, True])
def test_demand_visits_exactly_the_physical_successors_of_the_roots(
    *, promote: bool, redundant_root: bool, enable_jit: bool
) -> None:
    """H is the realized closure of the roots; `perceived` is in it only as a root."""
    model = _demand_model(
        promote=promote, redundant_root=redundant_root, enable_jit=enable_jit
    )
    assert model.reachability.visited_nodes == _visited(promote=promote)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("redundant_root", [False, True])
@pytest.mark.parametrize("promote", [False, True])
def test_demand_values_the_visits_and_the_perceived_continuation(
    *, promote: bool, redundant_root: bool, enable_jit: bool
) -> None:
    """S is H plus the perceived continuation `(1, 'perceived')`."""
    model = _demand_model(
        promote=promote, redundant_root=redundant_root, enable_jit=enable_jit
    )
    assert model.reachability.nodes == _visited(promote=promote) | {(1, "perceived")}


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("redundant_root", [False, True])
def test_value_only_regime_has_no_realized_route_parameter(
    *, redundant_root: bool, enable_jit: bool
) -> None:
    """A valued, never visited regime owes only its backward parameters."""
    model = _demand_model(
        promote=False, redundant_root=redundant_root, enable_jit=enable_jit
    )
    assert _leaf_names(tree=model.get_params_template()["perceived"]) == {
        "discount_factor",
        "backward_bonus",
    }


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("redundant_root", [False, True])
def test_promoting_a_value_only_regime_adds_its_realized_route_parameter(
    *, redundant_root: bool, enable_jit: bool
) -> None:
    """The same declared law owes `realized_rate` once its regime is visited."""
    model = _demand_model(
        promote=True, redundant_root=redundant_root, enable_jit=enable_jit
    )
    assert _leaf_names(tree=model.get_params_template()["perceived"]) == {
        "discount_factor",
        "backward_bonus",
        "realized_rate",
    }


def test_value_only_regime_solves_without_its_realized_route_parameter() -> None:
    """`V_perceived(w) = 1 + 1.5 w`, so `V_source(w) = 0.5 + 1.75 w` on w in {0, 1}."""
    model = _demand_model(promote=False)
    values = model.solve(
        params={"discount_factor": 0.5, "backward_bonus": 1.0}, log_level="off"
    ).values
    np.testing.assert_array_equal(np.asarray(values[0]["source"]), [0.5, 2.25])


@categorical(ordered=False)
class _ProbabilityId:
    working: ScalarInt
    left: ScalarInt
    right: ScalarInt


def _ten() -> FloatND:
    return jnp.asarray(10.0)


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _zero_flow(*, wealth: ContinuousState) -> FloatND:
    return 0.0 * wealth


def _left_from_wealth(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _right_from_wealth(*, wealth: ContinuousState) -> FloatND:
    return 1.0 - wealth


def _moving_grid(age: float) -> LinSpacedGrid:
    return LinSpacedGrid(start=float(age), stop=float(age) + 1.0, n_points=2)


def _age_signature(age: float) -> float:
    return float(age)


def _age_grid_model(*, earlier_root: bool, enable_jit: bool = True) -> Model:
    """Wealth lives on {age, age + 1}; the age-1 law is `(wealth, 1 - wealth)`."""
    roots: dict[object, str] = {(0, 1): "working"} if earlier_root else {1: "working"}
    return Model(
        enable_jit=enable_jit,
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_ProbabilityId,
        initial_regimes=roots,
        regimes={
            "working": Regime(
                regime_transitions=ByAge(
                    cases={
                        0: "left",
                        1: {
                            "left": MarkovTransition(func=_left_from_wealth),
                            "right": MarkovTransition(func=_right_from_wealth),
                        },
                    }
                ),
                states={
                    "wealth": AgeSpecializedGrid(
                        build=_moving_grid, signature=_age_signature
                    )
                },
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _zero_flow},
            ),
            "left": Regime(regime_transitions=None, functions={"utility": _ten}),
            "right": Regime(regime_transitions=None, functions={"utility": _zero}),
        },
    )


@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
@pytest.mark.parametrize("earlier_root", [False, True])
def test_probability_validation_uses_each_required_age_grid(
    *, earlier_root: bool, log_level: LogLevel
) -> None:
    """At age 1 the grid point wealth=2 gives probabilities (2, -1)."""
    model = _age_grid_model(earlier_root=earlier_root)
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match="outside "):
        model.solve(params={"discount_factor": 0.5}, log_level=log_level)


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _carried_left(*, carried_share: ContinuousState) -> FloatND:
    return carried_share


def _carried_right(*, carried_share: ContinuousState) -> FloatND:
    return 1.0 - carried_share


def _carried_model(*, enable_jit: bool = True) -> Model:
    """The solve law is (1/2, 1/2); the realized law reads the carried share."""
    return Model(
        enable_jit=enable_jit,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_ProbabilityId,
        initial_regimes={0: "working"},
        regimes={
            "working": Regime(
                regime_transitions=Phased(
                    solve={
                        "left": MarkovTransition(func=_half),
                        "right": MarkovTransition(func=_half),
                    },
                    simulate={
                        "left": MarkovTransition(func=_carried_left),
                        "right": MarkovTransition(func=_carried_right),
                    },
                ),
                states={
                    "wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
                    "carried_share": Phased(
                        solve=_half,
                        simulate=LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
                    ),
                },
                state_transitions={
                    "wealth": fixed_transition("wealth"),
                    "carried_share": fixed_transition("carried_share"),
                },
                functions={"utility": _zero_flow},
            ),
            "left": Regime(regime_transitions=None, functions={"utility": _ten}),
            "right": Regime(regime_transitions=None, functions={"utility": _zero}),
        },
    )


@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
def test_realized_law_is_checked_on_the_carried_state_grid(
    *, log_level: LogLevel
) -> None:
    """Every carried-share row is a valid law, so `V = 0.5 * 5 = 2.5` everywhere."""
    solution = _carried_model().solve(
        params={"discount_factor": 0.5}, log_level=log_level
    )
    np.testing.assert_array_equal(np.asarray(solution.values[0]["working"]), [2.5, 2.5])


@pytest.mark.parametrize("seed", [0, 9])
@pytest.mark.parametrize("n_subjects", [2, 5])
def test_realized_routing_reads_the_actual_carried_state(
    *, n_subjects: int, seed: int
) -> None:
    """A carried share of 0 routes right and a share of 1 routes left, surely."""
    model = _carried_model()
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    panel = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.zeros(n_subjects),
            "regime_id": jnp.full(n_subjects, model.regime_names_to_ids["working"]),
            "wealth": jnp.ones(n_subjects),
            "carried_share": jnp.asarray(np.arange(n_subjects) % 2, dtype=float),
        },
        seed=seed,
        log_level="off",
    ).to_dataframe()
    last = panel.query("period == 1").sort_values("subject_id")
    assert list(last["regime_name"]) == [
        "right" if subject % 2 == 0 else "left" for subject in range(n_subjects)
    ]


def _feasible(*, wealth: ContinuousState, consumption: ContinuousState) -> BoolND:
    return consumption <= wealth


def _consumption_utility(*, consumption: ContinuousState) -> FloatND:
    return consumption


def _feasible_left(*, wealth: ContinuousState, consumption: ContinuousState) -> FloatND:
    return jnp.where(consumption <= wealth, 0.5, jnp.nan)


def _feasible_right(
    *, wealth: ContinuousState, consumption: ContinuousState
) -> FloatND:
    return 1.0 - _feasible_left(wealth=wealth, consumption=consumption)


def _bad_on_feasible_left(
    *, wealth: ContinuousState, consumption: ContinuousState
) -> FloatND:
    return jnp.where((wealth == 1.0) & (consumption == 0.0), 1.5, 0.5)


def _bad_on_feasible_right(
    *, wealth: ContinuousState, consumption: ContinuousState
) -> FloatND:
    return 1.0 - _bad_on_feasible_left(wealth=wealth, consumption=consumption)


def _feasibility_model(
    *, bad_feasible: bool, n_points: int = 2, enable_jit: bool = True
) -> Model:
    """Consumption above wealth is infeasible; the valid law is NaN only there."""
    left = _bad_on_feasible_left if bad_feasible else _feasible_left
    right = _bad_on_feasible_right if bad_feasible else _feasible_right
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=n_points)
    return Model(
        enable_jit=enable_jit,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_ProbabilityId,
        initial_regimes={0: "working"},
        regimes={
            "working": Regime(
                regime_transitions={
                    "left": MarkovTransition(func=left),
                    "right": MarkovTransition(func=right),
                },
                states={"wealth": grid},
                state_transitions={"wealth": fixed_transition("wealth")},
                actions={"consumption": grid},
                constraints={"feasible_consumption": _feasible},
                functions={"utility": _consumption_utility},
            ),
            "left": Regime(regime_transitions=None, functions={"utility": _ten}),
            "right": Regime(regime_transitions=None, functions={"utility": _zero}),
        },
    )


@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
def test_probability_check_skips_economically_infeasible_rows(
    *, log_level: LogLevel
) -> None:
    """Only (wealth=0, consumption=1) is NaN; `V(w) = w + 2.5` on w in {0, 1}."""
    values = (
        _feasibility_model(bad_feasible=False)
        .solve(params={"discount_factor": 0.5}, log_level=log_level)
        .values
    )
    np.testing.assert_array_equal(np.asarray(values[0]["working"]), [2.5, 3.5])


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("n_points", [2, 3])
def test_probability_check_skips_infeasible_rows_on_a_refined_grid(
    *, n_points: int, enable_jit: bool
) -> None:
    """With the NaN only at infeasible rows, `V(w) = w + 2.5` on every grid point."""
    values = (
        _feasibility_model(bad_feasible=False, n_points=n_points, enable_jit=enable_jit)
        .solve(params={"discount_factor": 0.5}, log_level="off")
        .values
    )
    np.testing.assert_array_equal(
        np.asarray(values[0]["working"]), np.linspace(0.0, 1.0, n_points) + 2.5
    )


@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
def test_probability_check_rejects_a_bad_law_at_a_feasible_row(
    *, log_level: LogLevel
) -> None:
    """(wealth=1, consumption=0) is feasible and gives probabilities (1.5, -0.5)."""
    model = _feasibility_model(bad_feasible=True)
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match="outside "):
        model.solve(params={"discount_factor": 0.5}, log_level=log_level)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("n_points", [2, 3])
def test_probability_check_rejects_a_bad_feasible_row_on_a_refined_grid(
    *, n_points: int, enable_jit: bool
) -> None:
    """The feasible row (wealth=1, consumption=0) stays invalid on every grid."""
    model = _feasibility_model(
        bad_feasible=True, n_points=n_points, enable_jit=enable_jit
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match="outside "):
        model.solve(params={"discount_factor": 0.5}, log_level="off")
