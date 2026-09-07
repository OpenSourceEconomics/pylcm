"""What a solve releases, and what it leaves alone.

`log_level="debug"` logs every release with the artifact key and the closing
dispatch; a solve at any other level publishes the same values. A release only
ever frees a buffer a compiled executable produced: an eager solve releases
nothing, and an array the model declares — a plain grid or an age-specialized
per-period axis — stays readable even when a solver hands it through as a
continuation leaf.
"""

import dataclasses
import logging
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.scheduler import BufferRegistry
from _lcm.execution.value_transfer import ValueArtifactKind
from _lcm.grids.base import Grid
from _lcm.solution import backward_induction
from _lcm.solution.kernel_output import ConsumedKernelOutput
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from lcm import AgeGrid, AgeSpecializedGrid, LinSpacedGrid, Model, categorical
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    post_decision_lower_bound,
)
from lcm.regime import Regime
from lcm.solver_api import ArtifactKey, SolutionResult
from lcm.solvers import EGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
)


def _params() -> dict[str, float]:
    """Discount factor and consent-threshold parameters of the mini-EKL model."""
    return {"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2}


class _Records(logging.Handler):
    """Collect the release records the solve emits, in emission order."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if hasattr(record, "artifact_key"):
            self.records.append(record)


def _model() -> Model:
    return Model(
        regimes=_make_full_topology_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=EKLRegimeId,
    )


def _solve_with_release_log() -> tuple[Model, list[logging.LogRecord], SolutionResult]:
    model = _model()
    handler = _Records()
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        solution = model.solve(params=_params(), log_level="debug")
    finally:
        logger.removeHandler(handler)
    return model, handler.records, solution


def test_a_gated_continuation_is_released_by_the_source_that_reads_it() -> None:
    """Every released Wbar names the source's own dispatch as the closer."""
    _, records, _ = _solve_with_release_log()
    released = [
        (r.artifact_key, r.closing_dispatch)  # ty: ignore[unresolved-attribute]
        for r in records
        if r.artifact_key.kind  # ty: ignore[unresolved-attribute]
        is ValueArtifactKind.GATED_CONTINUATION
    ]

    assert released
    assert all(
        dispatch == (artifact.period - 1, artifact.regime)
        for artifact, dispatch in released
    )


def test_no_regime_value_is_ever_released() -> None:
    """Retained values stay on device for the whole solve."""
    _, records, solution = _solve_with_release_log()

    assert not any(
        r.artifact_key.kind  # ty: ignore[unresolved-attribute]
        is ValueArtifactKind.REGIME_VALUE
        for r in records
    )
    assert not any(
        value.is_deleted()
        for period in solution.values.values()
        for value in period.values()
    )


def test_releasing_changes_no_published_value() -> None:
    """A solve with releases publishes the values a silent solve publishes."""
    model, _, debug = _solve_with_release_log()
    silent = model.solve(params=_params(), log_level="off")

    np.testing.assert_array_equal(
        np.asarray(debug.values[1]["married"]), np.asarray(silent.values[1]["married"])
    )


_PASS_THROUGH_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=5.0, n_points=6)
_PASS_THROUGH_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=5.0, n_points=7)
_PASS_THROUGH_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=5.0, n_points=8)


@categorical(ordered=False)
class _PassThroughRegimeId:
    working: ScalarInt
    dead: ScalarInt


def _pass_through_next_regime() -> ScalarInt:
    """One decision period is followed by the terminal regime."""
    return _PassThroughRegimeId.dead


def _pass_through_utility(consumption: ContinuousAction) -> FloatND:
    """Log flow utility of consumption."""
    return jnp.log(consumption)


def _pass_through_terminal_utility(wealth: ContinuousState) -> FloatND:
    """The terminal period consumes the wealth on hand."""
    return jnp.log1p(wealth)


def _pass_through_savings(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    """Post-decision liquid balance."""
    return wealth - consumption


def _pass_through_next_wealth(savings: ContinuousState) -> ContinuousState:
    """Carry the liquid balance into the terminal period."""
    return savings


def _pass_through_margin() -> LiquidMargin:
    """The single liquid margin of the pass-through model."""
    return LiquidMargin(
        state="wealth",
        action="consumption",
        resources="wealth",
        post_decision_state="savings",
    )


def _pass_through_model(
    *,
    enable_jit: bool = False,
    wealth_grid: Grid | AgeSpecializedGrid = _PASS_THROUGH_WEALTH_GRID,
) -> Model:
    """A two-period EGM model whose terminal carry reuses the declared grid.

    The terminal regime carries no other state, so its endogenous grid is the
    wealth state's own axis, unbroadcast. Solving without compilation hands that
    very array to the working regime as a continuation leaf. `wealth_grid`
    varies the one declaration that decides which array the axis comes from —
    the model's materialized grid, or an `AgeSpecializedGrid`'s per-period node
    table.
    """
    margin = _pass_through_margin()
    working = ConsumptionSavingsRegime(
        transition=_pass_through_next_regime,
        states={"wealth": wealth_grid},
        actions={"consumption": _PASS_THROUGH_CONSUMPTION_GRID},
        state_transitions={"wealth": _pass_through_next_wealth},
        functions={
            "utility": _pass_through_utility,
            "savings": _pass_through_savings,
        },
        constraints={
            "borrowing_limit": post_decision_lower_bound(margin=margin, lower=0.0)
        },
        liquid=margin,
        solver=EGM(savings_grid=_PASS_THROUGH_SAVINGS_GRID),
        active=lambda age: age == 0,
    )
    dead = Regime(
        transition=None,
        states={"wealth": wealth_grid},
        functions={"utility": _pass_through_terminal_utility},
        active=lambda age: age == 1,
    )
    return Model(
        regimes={"working": working, "dead": dead},
        regime_id_class=_PassThroughRegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        enable_jit=enable_jit,
    )


def _age_specialized_wealth_grid() -> AgeSpecializedGrid:
    """A wealth axis whose floor rises with age, at a fixed number of points."""
    return AgeSpecializedGrid(
        build=lambda age: LinSpacedGrid(start=1.0 + 0.5 * age, stop=5.0, n_points=6),
        signature=lambda age: age,
    )


def _pass_through_params() -> dict[str, dict[str, dict[str, float]]]:
    """Discount factor of the pass-through model."""
    return {"working": {"koopmans_aggregator": {"discount_factor": 0.95}}, "dead": {}}


def test_a_model_whose_grid_is_a_continuation_leaf_solves_twice_alike() -> None:
    """A grid a solve hands through as a continuation leaf stays readable."""
    model = _pass_through_model()
    params = _pass_through_params()

    first = model.solve(params=params, log_level="debug")
    second = model.solve(params=params, log_level="debug")

    np.testing.assert_array_equal(
        np.asarray(second.values[0]["working"]),
        np.asarray(first.values[0]["working"]),
    )


def test_a_compiled_model_handing_its_grid_through_still_releases_a_leaf() -> None:
    """Handing a declared grid through leaves the engine's own leaves releasable."""
    model = _pass_through_model(enable_jit=True)
    handler = _Records()
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        model.solve(params=_pass_through_params(), log_level="debug")
    finally:
        logger.removeHandler(handler)

    assert any(
        record.artifact_key.kind  # ty: ignore[unresolved-attribute]
        is ValueArtifactKind.CONTINUATION_LEAF
        for record in handler.records
    )


def test_an_eager_solve_releases_nothing() -> None:
    """An eager dispatch's outputs may be its inputs, so none of them is freed."""
    model = _pass_through_model(enable_jit=False)
    handler = _Records()
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        model.solve(params=_pass_through_params(), log_level="debug")
    finally:
        logger.removeHandler(handler)

    assert handler.records == []


def test_an_age_specialized_axis_survives_an_eager_solve() -> None:
    """A per-period state axis stays readable, and the solve repeats its values."""
    model = _pass_through_model(wealth_grid=_age_specialized_wealth_grid())
    params = _pass_through_params()

    first = model.solve(params=params, log_level="debug")
    second = model.solve(params=params, log_level="debug")

    np.testing.assert_array_equal(
        np.asarray(second.values[0]["working"]),
        np.asarray(first.values[0]["working"]),
    )


def test_an_age_specialized_axis_is_still_on_device_after_a_solve() -> None:
    """The per-period node table the solve read is not deleted by it."""
    model = _pass_through_model(wealth_grid=_age_specialized_wealth_grid())
    model.solve(params=_pass_through_params(), log_level="debug")

    axes = model._regimes["dead"].solution.period_state_axes
    nodes = [
        cast("jax.Array", axis)
        for period_axes in (axes or {}).values()
        for axis in period_axes.values()
    ]

    assert [axis.is_deleted() for axis in nodes] == [False]


_ALIASED_REPLAY_KEY = ArtifactKey(type_id="tests.release_log.aliased", schema_version=1)


def test_a_replay_payload_sharing_a_continuation_leaf_survives_the_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One array published on two channels outlives a release of either."""
    published: list[FloatND] = []
    consume = backward_induction.consume_kernel_output

    def _also_publish_on_replay(**kwargs: object) -> ConsumedKernelOutput:
        result = consume(**kwargs)  # ty: ignore[invalid-argument-type]
        if result.continuation is None:
            return result
        leaf = jax.tree.leaves(result.continuation)[0]
        published.append(leaf)
        return dataclasses.replace(
            result,
            replay_artifacts=MappingProxyType({_ALIASED_REPLAY_KEY: leaf}),
        )

    monkeypatch.setattr(
        backward_induction, "consume_kernel_output", _also_publish_on_replay
    )
    _pass_through_model(enable_jit=True).solve(
        params=_pass_through_params(), log_level="debug"
    )

    assert [leaf.is_deleted() for leaf in published] == [False, False]


def _diagnostics_carrying(*, leaf: FloatND) -> SolverDiagnostics:
    """A diagnostic payload whose float fields all hold `leaf`."""
    flag = jnp.zeros((), dtype=jnp.bool_)
    count = jnp.asarray(0, dtype=jnp.int32)
    return SolverDiagnostics(
        max_outer_interpolation_error=leaf,
        max_outer_bracket_width=leaf,
        outer_nodes_used=count,
        outer_at_lower_bound=flag,
        outer_at_upper_bound=flag,
        keeper_adjuster_margin=leaf,
        best_second_best_margin=leaf,
        policy_fallback_mask=flag,
        unresolved_mask=flag,
        n_outer_all_invalid_cells=count,
    )


def test_a_retained_dissolution_flag_is_declared_on_its_own_buffer() -> None:
    """The per-regime flag mapping the result publishes protects its arrays."""
    registry = BufferRegistry()
    flag = jnp.zeros((3,), dtype=jnp.bool_)

    registry.declare_not_produced(tree=({"couple": flag},))

    assert registry.is_not_produced(array=flag)


def test_a_retained_diagnostic_payload_is_declared_on_its_own_buffers() -> None:
    """A diagnostic payload's arrays are reached even though it is no pytree."""
    registry = BufferRegistry()
    leaf = jnp.arange(3.0)
    payload = {"working": _diagnostics_carrying(leaf=leaf)}

    registry.declare_not_produced(
        tree=backward_induction._diagnostic_arrays(diagnostics=tuple(payload.values()))
    )

    assert registry.is_not_produced(array=leaf)


def test_a_diagnostic_payload_walked_as_a_tree_reaches_no_array() -> None:
    """Walking the payload directly is the mistake the flattening exists for."""
    registry = BufferRegistry()
    leaf = jnp.arange(3.0)

    registry.declare_not_produced(tree=({"working": _diagnostics_carrying(leaf=leaf)},))

    assert not registry.is_not_produced(array=leaf)


def test_a_diagnostic_payload_sharing_a_continuation_leaf_survives_the_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained diagnostic's array outlives the release of the leaf it shares."""
    published: list[FloatND] = []
    consume = backward_induction.consume_kernel_output

    def _also_publish_as_diagnostics(**kwargs: object) -> ConsumedKernelOutput:
        result = consume(**kwargs)  # ty: ignore[invalid-argument-type]
        if result.continuation is None:
            return result
        leaf = jax.tree.leaves(result.continuation)[0]
        published.append(leaf)
        return dataclasses.replace(result, diagnostics=_diagnostics_carrying(leaf=leaf))

    monkeypatch.setattr(
        backward_induction, "consume_kernel_output", _also_publish_as_diagnostics
    )
    _pass_through_model(enable_jit=True).solve(
        params=_pass_through_params(), log_level="debug"
    )

    assert [leaf.is_deleted() for leaf in published] == [False, False]


def test_a_replay_payload_sharing_a_continuation_leaf_keeps_its_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The array a second channel published still reads what it published."""
    published: list[tuple[FloatND, np.ndarray]] = []
    consume = backward_induction.consume_kernel_output

    def _also_publish_on_replay(**kwargs: object) -> ConsumedKernelOutput:
        result = consume(**kwargs)  # ty: ignore[invalid-argument-type]
        if result.continuation is None:
            return result
        leaf = jax.tree.leaves(result.continuation)[0]
        published.append((leaf, np.asarray(leaf).copy()))
        return dataclasses.replace(
            result,
            replay_artifacts=MappingProxyType({_ALIASED_REPLAY_KEY: leaf}),
        )

    monkeypatch.setattr(
        backward_induction, "consume_kernel_output", _also_publish_on_replay
    )
    _pass_through_model(enable_jit=True).solve(
        params=_pass_through_params(), log_level="debug"
    )

    np.testing.assert_array_equal(
        np.stack([np.asarray(leaf) for leaf, _ in published]),
        np.stack([snapshot for _, snapshot in published]),
    )


def _diagnostics_with_distinct_leaves() -> SolverDiagnostics:
    """A diagnostic payload whose every field holds a buffer of its own."""
    return SolverDiagnostics(
        max_outer_interpolation_error=jnp.arange(1.0),
        max_outer_bracket_width=jnp.arange(2.0),
        outer_nodes_used=jnp.arange(3, dtype=jnp.int32),
        outer_at_lower_bound=jnp.zeros((4,), dtype=jnp.bool_),
        outer_at_upper_bound=jnp.zeros((5,), dtype=jnp.bool_),
        keeper_adjuster_margin=jnp.arange(6.0),
        best_second_best_margin=jnp.arange(7.0),
        policy_fallback_mask=jnp.zeros((8,), dtype=jnp.bool_),
        unresolved_mask=jnp.zeros((9,), dtype=jnp.bool_),
        n_outer_all_invalid_cells=jnp.arange(10, dtype=jnp.int32),
        adjustment_probability=jnp.arange(11.0),
    )


def test_every_diagnostics_field_is_declared_before_a_release_or_donation() -> None:
    """No field of a retained diagnostic payload is left releasable."""
    registry = BufferRegistry()
    payload = _diagnostics_with_distinct_leaves()

    registry.declare_not_produced(
        tree=backward_induction._diagnostic_arrays(diagnostics=(payload,))
    )

    assert all(
        registry.is_not_produced(array=getattr(payload, field.name))
        for field in dataclasses.fields(payload)
    )
