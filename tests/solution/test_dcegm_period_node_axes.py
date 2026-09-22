"""A DC-EGM period group owns the actual stochastic mesh it integrates."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import step
from _lcm.execution.core_program import core_program_graph
from _lcm.solution.dcegm import EGMStepBuild
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ExecutionPlanningError
from lcm.regime import Regime
from lcm.solver_api import SolutionResult
from lcm.solvers import DCEGM, STOCHASTIC_NODE_AXIS
from lcm.typing import FloatND, ScalarInt
from tests.conftest import assert_agrees_to_ulp
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.solution.test_dcegm_axis_width_policy import (
    Health,
    inverse_marginal_utility,
    next_wealth,
    savings,
    utility,
)
from tests.solution.test_dcegm_core_program import _run


@categorical(ordered=False)
class ShortHealth:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class DiagnosisRegimes:
    parent: ScalarInt
    young: ScalarInt
    old: ScalarInt
    dead: ScalarInt


def young_probability(age: int) -> FloatND:
    return jnp.asarray(age < 50, dtype=float)


def old_probability(age: int) -> FloatND:
    return jnp.asarray(age >= 50, dtype=float)


def three_health() -> FloatND:
    return jnp.asarray([0.2, 0.3, 0.5])


def two_health() -> FloatND:
    return jnp.asarray([0.4, 0.6])


def final_bequest(wealth: FloatND) -> FloatND:
    return jnp.log(wealth + 1.0)


def death_probability() -> FloatND:
    return jnp.asarray(1.0)


def _model(
    *,
    short_old_health: bool,
    parent_period: int | None = None,
    overlapping_children: bool = False,
) -> Model:
    grid = LinSpacedGrid(start=1.0, stop=20.0, n_points=4)
    old_domain = ShortHealth if short_old_health else Health
    parent = ConsumptionSavingsRegime(
        transition={
            "young": MarkovTransition(young_probability),
            "old": MarkovTransition(old_probability),
        },
        active=lambda age: (
            age < 60 if parent_period is None else age == 40 + 10 * parent_period
        ),
        states={"wealth": grid, "health": DiscreteGrid(Health)},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=5)},
        state_transitions={
            "wealth": next_wealth,
            "health": {
                "young": MarkovTransition(three_health),
                "old": MarkovTransition(
                    two_health if short_old_health else three_health
                ),
            },
        },
        functions={
            "utility": utility,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=5),
            n_constrained_points=4,
        ),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        ),
    )
    return Model(
        regimes={
            "parent": parent,
            "young": parent.replace(
                transition={"dead": MarkovTransition(death_probability)},
                active=lambda age: age == 50,
                state_transitions={"wealth": next_wealth, "health": {}},
            ),
            "old": parent.replace(
                transition={"dead": MarkovTransition(death_probability)},
                active=lambda age: age == 60 or (overlapping_children and age == 50),
                states={"wealth": grid, "health": DiscreteGrid(old_domain)},
                state_transitions={"wealth": next_wealth, "health": {}},
            ),
            "dead": Regime(
                transition=None,
                states={"wealth": grid},
                functions={"utility": final_bequest},
            ),
        },
        ages=AgeGrid(start=40, stop=70, step="10Y"),
        regime_id_class=DiagnosisRegimes,
        execution_config=ExecutionConfig(devices=(0,)),
    )


@pytest.mark.parametrize("short_old_health", [False, True])
def test_period_groups_keep_their_actual_child_health_extents(
    *, short_old_health: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = step.build_egm_step_functions
    recorded = {}

    def observe(**kwargs: Any) -> EGMStepBuild:
        build = original(**kwargs)
        recorded[kwargs["regime_name"]] = dict(build.stochastic_node_axes_by_period)
        return build

    monkeypatch.setattr(step, "build_egm_step_functions", observe)
    model = _model(short_old_health=short_old_health)
    assert recorded["parent"] == {
        0: (("health", 3),),
        1: (("health", 2 if short_old_health else 3),),
    }
    assert recorded["young"] == {1: ()}
    assert recorded["old"] == {2: ()}
    assert model._regimes["young"].solution.grids["health"].to_jax().size == 3
    assert model._regimes["old"].solution.grids["health"].to_jax().size == (
        2 if short_old_health else 3
    )
    kernels = model._regimes["parent"].solution.period_kernels
    assert set(kernels) == {0, 1}
    for period, kernel in kernels.items():
        assert set(core_program_graph(kernel=kernel)) == {"main", "replay"}
        for program in core_program_graph(kernel=kernel).values():
            axes = program.requirements.reduced_axes
            if short_old_health and period == 1:
                assert axes == ()
            else:
                assert len(axes) == 1
                assert axes[0].name == STOCHASTIC_NODE_AXIS
                assert axes[0].coordinate_names == ("health",)
                assert axes[0].coordinate_extents == (3,)


def test_conflicting_target_meshes_within_one_group_remain_refused() -> None:
    with pytest.raises(
        ExecutionPlanningError, match="under one target and 3 under 'young'"
    ):
        _model(short_old_health=True, overlapping_children=True)


@pytest.mark.parametrize("period", [0, 1])
def test_grouped_values_and_replay_match_independent_single_period_models(
    *, period: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_solve = Model.solve
    solved_values = []

    def record(self: Model, **kwargs: Any) -> SolutionResult:
        result = original_solve(self, **kwargs)
        solved_values.append(np.asarray(result.values[period]["parent"]))
        return result

    monkeypatch.setattr(Model, "solve", record)
    outputs = []
    for parent_period in (None, period):
        model = _model(short_old_health=True, parent_period=parent_period)
        kernel, context = ride_along_kernel(
            model=model,
            params={"discount_factor": 0.95},
            regime_name="parent",
            period=period,
        )
        outputs.append(_run(kernel=kernel, context=context, name="replay"))
    actual, expected = outputs
    assert len(solved_values) == 2
    assert_agrees_to_ulp(got=solved_values[0], expected=solved_values[1], n_ulp=8)
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    assert np.isfinite(np.asarray(actual[0])).all()
    for got, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(np.isfinite(got), np.isfinite(reference))
        assert_agrees_to_ulp(
            got=np.asarray(got), expected=np.asarray(reference), n_ulp=8
        )


def test_shorter_child_domain_changes_the_solved_parent_value() -> None:
    values = [
        np.asarray(
            _model(short_old_health=short)
            .solve(params={"discount_factor": 0.95}, log_level="off")
            .values[1]["parent"]
        )
        for short in (False, True)
    ]
    assert all(np.isfinite(value).all() for value in values)
    assert not np.allclose(values[0], values[1], rtol=0.0, atol=0.01)
