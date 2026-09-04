"""Production-path control for ordinary singleton action streaming."""

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    core_program_graph,
)
from _lcm.regime_building import max_Q_over_a
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


def _build_model(*, enable_jit: bool = True) -> Model:
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
        enable_jit=enable_jit,
    )


def _solve_target(*, model: Model, work: float, consumption: float) -> FloatND:
    """Solve with exactly one feasible target action cell."""
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"]["only_target"]["target_work"] = work
    params["acting"]["only_target"]["target_consumption"] = consumption
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    return model.solve(params=params, log_level="debug").values[0]["acting"]


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


def test_eager_singleton_hard_max_never_builds_the_dense_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JIT-disabled production resolves the same streamed native program."""
    real_get_max_Q_over_a = max_Q_over_a.get_max_Q_over_a

    def fail_dense_construction(**kwargs: Any) -> Callable[..., Any]:
        if kwargs["action_names"]:
            raise AssertionError("eligible eager GridSearch reached its dense oracle")
        return real_get_max_Q_over_a(**kwargs)

    monkeypatch.setattr(max_Q_over_a, "get_max_Q_over_a", fail_dense_construction)
    model = _build_model(enable_jit=False)
    actual = _solve_target(model=model, work=1.0, consumption=3.0)

    program = core_program_graph(
        kernel=model._regimes["acting"].solution.period_kernels[0]
    )["main"]
    assert program.disposition is CoreExecutionDisposition.PLANNED
    assert program.disposition_reason is None
    assert_array_equal(actual, jnp.asarray([14.0, 15.0]))


def test_public_collective_solve_does_not_call_streamed_household_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adverse resource route is an explicit dense native program."""

    def fail_streamed_collective(**_kwargs: Any) -> None:
        raise AssertionError("dense collective route called streamed reduction")

    monkeypatch.setattr(
        action_streaming,
        "_evaluate_collective_block",
        fail_streamed_collective,
    )
    model = _build_collective_model()
    program = core_program_graph(
        kernel=model._regimes["couple"].solution.period_kernels[0]
    )["main"]
    solution = model.solve(params={"discount_factor": 0.95}, log_level="debug")

    assert program.disposition is CoreExecutionDisposition.DENSE
    assert (
        program.disposition_reason
        == "deliberately_dense:collective_resource_regression"
    )
    assert jnp.all(jnp.isfinite(solution.values[0]["couple"]))


def test_public_ev1_solve_does_not_call_streamed_branch_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The noncanonical streamed reduction is excluded from production."""

    def fail_streamed_ev1(**_kwargs: Any) -> None:
        raise AssertionError("dense EV1 route called streamed reduction")

    monkeypatch.setattr(
        action_streaming,
        "_evaluate_ev1_branch_block",
        fail_streamed_ev1,
    )

    model = taste_shocks_toy.get_model()
    program = core_program_graph(
        kernel=model._regimes["alive"].solution.period_kernels[0]
    )["main"]
    solution = model.solve(
        params=taste_shocks_toy.get_params(
            scale=0.2,
            discount_factor=0.95,
        ),
        log_level="debug",
    )

    assert program.disposition is CoreExecutionDisposition.DENSE
    assert (
        program.disposition_reason == "deliberately_dense:ev1_canonical_reduction_order"
    )
    assert jnp.all(jnp.isfinite(solution.values[0]["alive"]))
