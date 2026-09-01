"""Production-path control for ordinary singleton action streaming."""

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from _lcm.solution import action_streaming
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.regime_building.test_collective_feasibility_is_shared import (
    _make_model as _build_collective_model,
)
from tests.test_models import taste_shocks_toy


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    working: ScalarInt


@categorical(ordered=False)
class RegimeId:
    acting: ScalarInt
    done: ScalarInt


def _next_regime() -> ScalarInt:
    """Move from the decision regime to the terminal regime."""
    return RegimeId.done


def _utility(
    *,
    wealth: ContinuousState,
    work: DiscreteAction,
    consumption: ContinuousAction,
) -> FloatND:
    """Give every C-order action cell a distinct observable value."""
    return wealth + 10.0 * work + consumption


def _only_target(
    *,
    work: DiscreteAction,
    consumption: ContinuousAction,
    target_work: float,
    target_consumption: float,
) -> BoolND:
    """Admit exactly the action cell named by the parameters."""
    return jnp.isclose(work, target_work) & jnp.isclose(consumption, target_consumption)


def _terminal_utility() -> FloatND:
    """Return an action-neutral terminal value."""
    return jnp.asarray(0.0)


def _build_model() -> Model:
    """Build the ordinary singleton model used by the production tracer."""
    acting = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={
            "work": DiscreteGrid(category_class=Work),
            "consumption": LinSpacedGrid(start=1.0, stop=3.0, n_points=3),
        },
        functions={"utility": _utility},
        constraints={"only_target": _only_target},
    )
    done = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"acting": acting, "done": done},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
    )


def _solve_target(*, model: Model, work: float, consumption: float) -> FloatND:
    """Solve with exactly one feasible target action cell."""
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"]["only_target"]["target_work"] = work
    params["acting"]["only_target"]["target_consumption"] = consumption
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    return model.solve(params=params, log_level="debug")[0]["acting"]


def test_public_singleton_solve_uses_streamed_action_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting global action five in the streamed core changes public solve output.

    The two action grids form the C-order product
    ``[(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]``. The injected defect masks
    only global identity five. A production solve targeting identity zero must remain
    unchanged, while a solve for identity five must publish an empty feasible set.
    """
    real_evaluate_block = cast(
        "Callable[..., tuple[jax.Array, jax.Array, jax.Array]]",
        action_streaming._evaluate_block,
    )

    def omit_global_action_five(
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        values, feasible, global_ids = real_evaluate_block(**kwargs)
        return values, feasible & (global_ids != 5), global_ids

    monkeypatch.setattr(
        action_streaming,
        "_evaluate_block",
        omit_global_action_five,
    )
    model = _build_model()

    untouched = _solve_target(model=model, work=0.0, consumption=1.0)
    omitted = _solve_target(model=model, work=1.0, consumption=3.0)

    assert_array_equal(untouched, jnp.asarray([2.0, 3.0]))
    assert bool(jnp.all(jnp.isneginf(omitted)))


def test_public_collective_solve_uses_one_streamed_household_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Masking global action one changes every stakeholder at the same winner."""
    real_evaluate_block = cast(
        "Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]]",
        action_streaming._evaluate_collective_block,
    )

    def omit_work_action(
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        objectives, stakeholder_values, feasible, global_ids = real_evaluate_block(
            **kwargs
        )
        return (
            objectives,
            stakeholder_values,
            feasible & (global_ids != 1),
            global_ids,
        )

    monkeypatch.setattr(
        action_streaming,
        "_evaluate_collective_block",
        omit_work_action,
    )
    solution, dissolution = _build_collective_model().solve(
        params={"discount_factor": 0.95},
        log_level="debug",
        return_dissolution_flags=True,
    )

    assert_array_equal(
        solution[1]["couple_terminal"],
        jnp.asarray([[-jnp.inf, -jnp.inf], [30.0, 0.0], [30.0, 0.0]]),
    )
    assert_array_equal(
        dissolution[1]["couple_terminal"],
        jnp.asarray([True, False, False]),
    )


def test_public_ev1_solve_maximizes_each_discrete_branch_before_logsum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping one branch leaves exactly the other branch's continuous maximum."""
    consumption = np.asarray(taste_shocks_toy.CONSUMPTION_GRID.to_jax())
    continuous_extent = consumption.size
    real_evaluate_block = cast(
        "Callable[..., tuple[jax.Array, jax.Array, jax.Array]]",
        action_streaming._evaluate_block,
    )
    observed = {"calls": 0}

    def omit_work_on_branch(
        **kwargs: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        values, feasible, global_ids = real_evaluate_block(**kwargs)
        observed["calls"] += 1
        return values, feasible & (global_ids < continuous_extent), global_ids

    monkeypatch.setattr(action_streaming, "_evaluate_block", omit_work_on_branch)

    discount_factor = 0.95
    solution = taste_shocks_toy.get_model().solve(
        params=taste_shocks_toy.get_params(
            scale=0.2,
            discount_factor=discount_factor,
        ),
        log_level="debug",
    )

    wealth = np.asarray(taste_shocks_toy.WEALTH_GRID.to_jax())
    terminal_wealth = np.asarray(taste_shocks_toy.TERMINAL_WEALTH_GRID.to_jax())
    continuation = np.interp(
        wealth[:, None] - consumption[None, :],
        terminal_wealth,
        np.log(terminal_wealth + 1.0),
    )
    work_off_Q = np.log(consumption)[None, :] + discount_factor * continuation
    expected = np.max(
        np.where(consumption[None, :] <= wealth[:, None], work_off_Q, -np.inf),
        axis=1,
    )

    assert observed["calls"] > 0
    assert_allclose(solution[0]["alive"], expected, rtol=1e-5, atol=1e-8)
