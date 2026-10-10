"""Production-path control for ordinary singleton action streaming."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Never

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
from _lcm.typing import MaxQOverAFunction, QAndFArg
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ReferenceName,
    RegimeName,
    ScalarInt,
    StateName,
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
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"acting": acting, "done": done},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        initial_nodes={0: "acting"},
        edges={"acting": {"done": 0}},
    )


type _MutableParam = str | float | dict[str, _MutableParam]
type _TemplateNode = str | Mapping[str, _TemplateNode]
type _FilledParam = float | dict[str, _FilledParam]


def _copy_template(node: _TemplateNode) -> _MutableParam:
    if isinstance(node, str):
        return node
    return {key: _copy_template(value) for key, value in node.items()}


def _require_filled(node: _MutableParam) -> _FilledParam:
    assert not isinstance(node, str)
    if isinstance(node, (int, float)):
        return node
    return {key: _require_filled(value) for key, value in node.items()}


def _solve_target(*, model: Model, work: float, consumption: float) -> FloatND:
    """Solve with exactly one feasible target action cell."""
    params = _copy_template(model.get_params_template())
    assert isinstance(params, dict)
    acting = params["acting"]
    assert isinstance(acting, dict)
    target = acting["only_target"]
    assert isinstance(target, dict)
    target["target_work"] = work
    target["target_consumption"] = consumption
    aggregator = acting["koopmans_aggregator"]
    assert isinstance(aggregator, dict)
    aggregator["discount_factor"] = 0.5
    filled = {key: _require_filled(value) for key, value in params.items()}
    return model.solve(params=filled, log_level="debug").values[0]["acting"]


def test_public_singleton_solve_uses_streamed_action_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting global action five in the streamed core changes public solve output.

    The two action grids form the C-order product
    ``[(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]``. The injected defect masks
    only global identity five. A production solve targeting identity zero must remain
    unchanged, while a solve for identity five must publish an empty feasible set.
    """
    real_evaluate_block = action_streaming._evaluate_block

    def omit_global_action_five(
        *,
        block_index: jax.Array,
        Q_and_F: Callable[..., tuple[FloatND, BoolND]],
        action_names: tuple[ActionName, ...],
        action_grids: tuple[jax.Array, ...],
        action_sizes: tuple[int, ...],
        fixed_kwargs: dict[ReferenceName, QAndFArg],
        n_actions: int,
        block_width: int,
        block_offsets: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        values, feasible, global_ids = real_evaluate_block(
            block_index=block_index,
            Q_and_F=Q_and_F,
            action_names=action_names,
            action_grids=action_grids,
            action_sizes=action_sizes,
            fixed_kwargs=fixed_kwargs,
            n_actions=n_actions,
            block_width=block_width,
            block_offsets=block_offsets,
        )
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

    def fail_dense_construction(
        *,
        Q_and_F: Callable[..., tuple[FloatND, BoolND]],
        batch_sizes: dict[StateName, int],
        action_names: tuple[ActionName, ...],
        state_names: tuple[StateName, ...],
        n_discrete_action_axes: int = 0,
        has_taste_shocks: bool = False,
        co_map_state_names: tuple[StateName, ...] = (),
        co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = (),
        stakeholders: tuple[str, ...] | None = None,
        pareto_weights: max_Q_over_a.ParetoWeights | None = None,
        fold_state_names: tuple[StateName, ...] = (),
        fold_weights: Mapping[StateName, FloatND] = MappingProxyType({}),
        fold_conditioning: Mapping[StateName, StateName] = MappingProxyType({}),
        cell_width_keyword: str | None = None,
        untiled_state_names: tuple[StateName, ...] = (),
        broadcast_state_names: tuple[StateName, ...] = (),
    ) -> MaxQOverAFunction:
        if action_names:
            raise AssertionError("eligible eager GridSearch reached its dense oracle")
        return real_get_max_Q_over_a(
            Q_and_F=Q_and_F,
            batch_sizes=batch_sizes,
            action_names=action_names,
            state_names=state_names,
            n_discrete_action_axes=n_discrete_action_axes,
            has_taste_shocks=has_taste_shocks,
            co_map_state_names=co_map_state_names,
            co_map_v_arr_in_axes=co_map_v_arr_in_axes,
            stakeholders=stakeholders,
            pareto_weights=pareto_weights,
            fold_state_names=fold_state_names,
            fold_weights=fold_weights,
            fold_conditioning=fold_conditioning,
            cell_width_keyword=cell_width_keyword,
            untiled_state_names=untiled_state_names,
            broadcast_state_names=broadcast_state_names,
        )

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
    """State tiling keeps the household's canonical dense action reduction."""

    def fail_streamed_collective[UnusedArgument](**_kwargs: UnusedArgument) -> Never:
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

    assert (
        program.disposition,
        program.disposition_reason,
        program.requirements.reduced_axes,
        bool(jnp.all(jnp.isfinite(solution.values[0]["couple"]))),
    ) == (CoreExecutionDisposition.PLANNED, None, (), True)


def test_public_ev1_solve_does_not_call_streamed_branch_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The noncanonical streamed reduction is excluded from production."""

    def fail_streamed_ev1[UnusedArgument](**_kwargs: UnusedArgument) -> Never:
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

    assert (
        program.disposition,
        program.disposition_reason,
        program.requirements.reduced_axes,
        bool(jnp.all(jnp.isfinite(solution.values[0]["alive"]))),
    ) == (CoreExecutionDisposition.PLANNED, None, (), True)
