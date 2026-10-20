"""Unexecuted public NNBEGM literal-parent / callable NB-child witness.

Integration: copy this file and oracle8_child_status.py into tests/solution/.
Native command after integration (not run by the reviewer):
  pytest tests/solution/test_round8_literal_parent_child_status.py -q

Only the parent's bound representation changes from the shipped resource
fixture: callable zero becomes literal zero.  Economics and the actual child
reader remain the same.  The finite route isolates this transport obligation
without introducing adaptive mesh-convergence assumptions.  The adaptive tests
use a converged mesh whose driver inserts nodes, solves validation probes it
leaves off the final mesh and serves repeated requests from its cache; each
new exact node solve must publish its child status and counts exactly once.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from _lcm.egm.outer_refinement import AdaptiveMeshResult
from _lcm.solution import nnbegm
from _lcm.solution.feasibility import (
    ChildStatusKernelOutput,
    ContinuationReadCounts,
    FeasibilityKernelOutput,
    QueryInputStatus,
)
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.solution.nnbegm import _InheritedChildSummary, _NNBEGMPeriodKernel
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    ExecutionConfig,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    Regime,
    SimulateSnapshot,
    SolveSnapshot,
    load_snapshot,
    outer_unchanged,
    post_decision_lower_bound,
)
from lcm.exceptions import InvalidSimulationInputError, InvalidValueFunctionError
from lcm.solver_api import KernelOutput, ResultRetention
from lcm.solvers import (
    NBEGM,
    NNBEGM,
    OUTER_CANDIDATE_AXIS,
    AdaptiveOuterMesh,
    FiniteOuterGrid,
)
from lcm.typing import ContinuousState, FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON, assert_agrees_to_ulp
from tests.solution import test_nbegm_query_actual_resources as fixture
from tests.solution._oracle8_child_status import (
    ExecutedRow,
    fold_executed,
    require_publication,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def make_literal_parent_model(
    *,
    resources,
    outer_search=None,
    enable_jit: bool = True,
    execution_config: ExecutionConfig | None = None,
) -> Model:
    """Use the shipped admitted model with a literal parent and DAG child floor."""
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    common = {
        "utility": fixture.utility,
        "savings": fixture.savings,
        "inverse_marginal_utility": fixture.inverse_marginal_utility,
    }
    parent = NestedConsumptionSavingsRegime(
        states={"liquid": fixture._LIQUID, "stock": fixture._STOCK},
        actions={
            "consumption": fixture._CONSUMPTION,
            "investment": fixture._INVESTMENT,
        },
        state_transitions={
            "liquid": fixture.next_liquid,
            "stock": fixture.next_parent_stock,
        },
        functions={
            **common,
            "resources": fixture.parent_resources,
            "new_stock": fixture.new_stock,
        },
        constraints={"borrowing": post_decision_lower_bound(margin=margin, lower=0.0)},
        liquid=margin,
        outer_continuous=OuterContinuousMargin(
            state="stock",
            action="investment",
            post_decision_state="new_stock",
            no_adjustment=outer_unchanged,
        ),
        solver=NNBEGM(
            inner=NBEGM(savings_grid=fixture._SAVINGS),
            outer_search=outer_search
            or FiniteOuterGrid(grid=IrregSpacedGrid(points=fixture._OUTER)),
        ),
    )
    child = ConsumptionSavingsRegime(
        koopmans_aggregator=parent.koopmans_aggregator,
        certainty_equivalent=parent.certainty_equivalent,
        states={"liquid": fixture._LIQUID, "stock": fixture._STOCK},
        actions={"consumption": fixture._CONSUMPTION},
        state_transitions={
            "liquid": fixture.next_liquid,
            "stock": fixture.next_child_stock,
        },
        functions={**common, "resources": resources},
        constraints={
            "borrowing": post_decision_lower_bound(margin=margin, lower=fixture.floor)
        },
        liquid=margin,
        solver=NBEGM(savings_grid=fixture._SAVINGS),
    )
    model = Model(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={"liquid": fixture._LIQUID, "stock": fixture._STOCK},
                functions={"utility": fixture.terminal},
            ),
        },
        ages=AgeGrid(exact_values=(0, 10, 20)),
        edges={"parent": {"child": (0,)}, "child": {"done": (10,)}},
        initial_nodes={0: "parent"},
        regime_id_class=fixture.RegimeId,
        enable_jit=enable_jit,
        execution_config=execution_config or ExecutionConfig(),
    )
    fixture.fail_if_not_actual_nb_query_path(model=model)
    kernel = model._regimes["parent"].solution.period_kernels[0]
    assert isinstance(kernel, _NNBEGMPeriodKernel)
    assert kernel.query_model is None
    authorities = (model._regimes["parent"].solution.query_input_authorities or {}).get(
        0, ()
    )
    assert authorities, "This witness requires the actual inherited child slots."
    return model


def _row(*, name, output) -> ExecutedRow:
    slots = tuple(
        tuple(bool(np.asarray(flag)) for flag in status)
        for status in (getattr(output, "query_status", None) or ())
    )
    native_counts = getattr(output, "continuation_read_counts", None)
    counts = (
        None
        if native_counts is None
        else tuple(int(np.asarray(x)) for x in native_counts)
    )
    return ExecutedRow(name=name, slots=slots, counts=counts)


@pytest.mark.parametrize("retention", list(ResultRetention))
@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("disable_jit", [False, True])
def test_public_literal_parent_preserves_executed_child_summaries(
    *, monkeypatch, runtime_checks, disable_jit, retention
) -> None:
    # The eager case also builds an eager model. A jitted model still lowers and
    # compiles every period program ahead of time, and under `jax.disable_jit`
    # each `lax.scan`/`lax.map` in that program is unrolled into the lowered
    # graph, so XLA compiles a vastly larger program instead of running eagerly.
    model = make_literal_parent_model(
        resources=fixture.child_positive_resources, enable_jit=not disable_jit
    )
    authorities = (model._regimes["parent"].solution.query_input_authorities or {})[0]
    native_inner = _RideAlongNBEGMPeriodKernel.__call__
    native_outer = _NNBEGMPeriodKernel.__call__
    stack = []
    observed = []

    def observe_inner(self, **kwargs):
        output = native_inner(self, **kwargs)
        if stack and self.regime_name == "parent":
            rows = stack[-1]
            rows.append(
                _row(
                    name="keeper" if not rows else f"adjuster-{len(rows)}",
                    output=output,
                )
            )
        return output

    def observe_outer(self, **kwargs):
        if self.regime_name != "parent":
            return native_outer(self, **kwargs)
        rows = []
        stack.append(rows)
        try:
            output = native_outer(self, **kwargs)
        finally:
            stack.pop()
        assert len(rows) == 1 + len(fixture._OUTER)
        # Model.solve runtime_checks selects the normal checked variant even
        # with log_level="off". Require real count collection, not an all-None
        # inner/outer pair that would vacuously agree with the optional oracle.
        if runtime_checks:
            assert all(row.counts is not None for row in rows)
            assert any(row.counts[1] > 0 for row in rows)
        else:
            assert all(row.counts is None for row in rows)
        expected = fold_executed(n_slots=len(authorities), rows=tuple(rows))
        require_publication(
            expected=expected, actual=_row(name="published", output=output)
        )
        observed.append(expected)
        return output

    monkeypatch.setattr(_RideAlongNBEGMPeriodKernel, "__call__", observe_inner)
    monkeypatch.setattr(_NNBEGMPeriodKernel, "__call__", observe_outer)
    with jax.disable_jit(disable_jit):
        model.solve(
            params=fixture.PARAMS,
            log_level="off",
            runtime_checks=runtime_checks,
            retention=retention,
        )
    assert observed


@pytest.mark.parametrize("retention", list(ResultRetention))
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_public_literal_parent_rejects_actual_nonfinite_child_budget(
    *, runtime_checks, retention
) -> None:
    """The host must report the actual child failure, with logging/checks irrelevant."""
    model = make_literal_parent_model(resources=fixture.child_nan_resources)
    with pytest.raises(InvalidValueFunctionError, match="query_coverage_missing"):
        model.solve(
            params=fixture.PARAMS,
            log_level="off",
            runtime_checks=runtime_checks,
            retention=retention,
        )


def _native_row(*, slots, counts):
    """Build one executed literal inner result with native scalar summaries."""
    return FeasibilityKernelOutput(
        value=jnp.zeros(2),
        feasibility_counts=(jnp.int32(0), jnp.int32(0)),
        query_status=tuple(
            QueryInputStatus(*(jnp.asarray(flag) for flag in slot)) for slot in slots
        ),
        continuation_read_counts=(
            None
            if counts is None
            else ContinuationReadCounts(*(jnp.int32(count) for count in counts))
        ),
    )


def test_inherited_child_fold_preserves_ordered_slots_and_sums_counts() -> None:
    """The production fold is the oracle fold on two ordered slots."""
    rows = (
        ExecutedRow("keeper", ((True, False, False), (False, False, False)), (1, 3)),
        ExecutedRow("winner", ((False, False, False), (False, True, False)), (2, 4)),
        ExecutedRow("excluded", ((False, False, False), (False, False, True)), (2, 4)),
    )
    native = tuple(_native_row(slots=row.slots, counts=row.counts) for row in rows)
    summary = _InheritedChildSummary.of(native[0])
    for row in native[1:]:
        summary = summary.add(row)
    published = summary.publish(KernelOutput(value=jnp.zeros(2)))
    assert type(published) is ChildStatusKernelOutput
    require_publication(
        expected=fold_executed(n_slots=2, rows=rows),
        actual=_row(name="published", output=published),
    )


def test_inherited_child_fold_rejects_mixed_count_schema() -> None:
    """A row without requested counts cannot be laundered into a partial sum."""
    slots = ((False, False, False),)
    summary = _InheritedChildSummary.of(_native_row(slots=slots, counts=(0, 1)))
    with pytest.raises(TypeError, match="read-count schema"):
        summary.add(_native_row(slots=slots, counts=None))


def test_inherited_child_fold_rejects_a_row_that_lost_its_slots() -> None:
    """Every executed solve carries the declared slots, even when all false."""
    summary = _InheritedChildSummary.of(
        _native_row(slots=((False, False, False),), counts=None)
    )
    with pytest.raises(TypeError, match="declaration slots"):
        summary.add(KernelOutput(value=jnp.zeros(2)))


# Tighter than a transport-only setting: on this kinked surface the mesh inserts
# nodes, leaves validated probes off its final mesh and serves repeated midpoint
# requests from its cache, then converges within the default node and round
# budgets. At the default tolerances it exhausts the round budget and fails
# closed, which is the mesh's own contract rather than a transport property.
_ADAPTIVE_MESH = AdaptiveOuterMesh(
    initial_grid=IrregSpacedGrid(points=fixture._OUTER),
    value_atol=1e-3,
    value_rtol=1e-3,
)

_SIMULATION_INITIAL = {
    "liquid": jnp.array([0.3, 1.7, 3.9, 6.1]),
    "stock": jnp.array([0.0, 0.4, 0.8, 1.0]),
    "age": jnp.zeros(4),
    "regime_id": jnp.zeros(4, dtype=jnp.int32),
}

# A failure radius far below the half-spacing of the nodes the mesh solves, so
# exactly one outer node reads a nonfinite child budget.
_NAN_RADIUS = 2.0**-8


@dataclass
class _AdaptiveExecution:
    """What one adaptive parent period executed, observed at native returns."""

    requests: list[tuple[float, ...]] = field(default_factory=list)
    """Each node batch the mesh driver asks its exact-solve callback for."""

    rows: list[tuple[float | None, ExecutedRow]] = field(default_factory=list)
    """Each executed inner solve: its outer node (`None` for the keeper)."""

    dispatches: list[tuple[int, ...]] = field(default_factory=list)
    """The chunk sizes of each host dispatch loop over pending nodes."""

    meshes: list[AdaptiveMeshResult] = field(default_factory=list)
    """The mesh the driver returns."""

    published: list[ExecutedRow] = field(default_factory=list)
    """The parent result's published child summary."""

    def requested(self) -> tuple[float, ...]:
        return tuple(node for request in self.requests for node in request)

    def first_solves(self) -> tuple[float, ...]:
        return tuple(dict.fromkeys(self.requested()))

    def cache_hits(self) -> tuple[float, ...]:
        seen: set[float] = set()
        hits = []
        for node in self.requested():
            if node in seen:
                hits.append(node)
            seen.add(node)
        return tuple(hits)

    def final_nodes(self) -> frozenset[float]:
        (mesh,) = self.meshes
        return frozenset(float(node) for node in np.asarray(mesh.nodes))

    def inserted(self) -> tuple[float, ...]:
        initial = frozenset(fixture._OUTER)
        return tuple(
            node
            for node in self.first_solves()
            if node in self.final_nodes() and node not in initial
        )

    def excluded_probes(self) -> tuple[float, ...]:
        return tuple(
            node for node in self.first_solves() if node not in self.final_nodes()
        )


def _observe_adaptive_parent(patch: pytest.MonkeyPatch) -> _AdaptiveExecution:
    """Record the adaptive parent's requests, executed solves, mesh and publication."""
    execution = _AdaptiveExecution()
    native_refine = nnbegm.refine_outer_mesh
    native_request = nnbegm._AdaptiveNodeSolver.__call__
    native_dispatch = nnbegm._dispatch_bounds
    native_node = _NNBEGMPeriodKernel._solve_adjuster_node
    native_inner = _RideAlongNBEGMPeriodKernel.__call__
    native_outer = _NNBEGMPeriodKernel.__call__
    nodes: list[float] = []
    parents: list[bool] = []

    def refine(**kwargs):
        mesh = native_refine(**kwargs)
        execution.meshes.append(mesh)
        return mesh

    # keyword-only-exempt: library-callback=_lcm.egm.outer_refinement.refine_outer_mesh
    def request(self, nodes_arr):
        execution.requests.append(tuple(float(node) for node in np.asarray(nodes_arr)))
        return native_request(self, nodes_arr)

    def dispatch(*, n_items, width):
        bounds = native_dispatch(n_items=n_items, width=width)
        execution.dispatches.append(tuple(stop - start for start, stop in bounds))
        return bounds

    def solve_node(self, **kwargs):
        nodes.append(float(kwargs["node"]))
        try:
            return native_node(self, **kwargs)
        finally:
            nodes.pop()

    def inner(self, **kwargs):
        output = native_inner(self, **kwargs)
        if parents and self.regime_name == "parent":
            node = nodes[-1] if nodes else None
            name = "keeper" if node is None else f"node-{node}"
            execution.rows.append((node, _row(name=name, output=output)))
        return output

    def outer(self, **kwargs):
        if self.regime_name != "parent":
            return native_outer(self, **kwargs)
        parents.append(True)
        try:
            output = native_outer(self, **kwargs)
        finally:
            parents.pop()
        execution.published.append(_row(name="published", output=output))
        return output

    patch.setattr(nnbegm, "refine_outer_mesh", refine)
    patch.setattr(nnbegm._AdaptiveNodeSolver, "__call__", request)
    patch.setattr(nnbegm, "_dispatch_bounds", dispatch)
    patch.setattr(_NNBEGMPeriodKernel, "_solve_adjuster_node", solve_node)
    patch.setattr(_RideAlongNBEGMPeriodKernel, "__call__", inner)
    patch.setattr(_NNBEGMPeriodKernel, "__call__", outer)
    return execution


def _n_child_slots(model: Model) -> int:
    return len((model._regimes["parent"].solution.query_input_authorities or {})[0])


def _require_adaptive_transport(
    *, execution: _AdaptiveExecution, n_slots: int, runtime_checks: bool
) -> None:
    """Each new exact node solve publishes once; cache hits and the mesh add nothing.

    The populations come from the driver's own requests and returned mesh, not
    from the production fold: a node first requested is one new exact solve, a
    repeated request is a cache hit, a solved node outside the final mesh is a
    validation probe and a final node outside the initial grid was inserted.
    """
    (mesh,) = execution.meshes
    assert mesh.rounds_used >= 1
    assert not mesh.unresolved
    assert execution.inserted(), "No solved node was inserted into the mesh."
    assert execution.excluded_probes(), "No solved probe was left off the mesh."
    assert execution.cache_hits(), "No repeated request was served from the cache."
    assert [node for node, _ in execution.rows] == [None, *execution.first_solves()]
    rows = tuple(row for _, row in execution.rows)
    if runtime_checks:
        assert all(row.counts is not None and row.counts[1] > 0 for row in rows)
    else:
        assert all(row.counts is None for row in rows)
    (published,) = execution.published
    require_publication(
        expected=fold_executed(n_slots=n_slots, rows=rows), actual=published
    )
    if runtime_checks:
        # Each excluded probe and each cache hit carries a positive eligible
        # read count, so a mesh-only fold and a fold that re-reads cache hits
        # both publish different counts from the executed one.
        excluded = frozenset(execution.excluded_probes())
        mesh_only = tuple(row for node, row in execution.rows if node not in excluded)
        assert fold_executed(n_slots=n_slots, rows=mesh_only).counts != (
            published.counts
        )
        by_node = dict(execution.rows)
        reread = rows + tuple(by_node[node] for node in execution.cache_hits())
        assert fold_executed(n_slots=n_slots, rows=reread).counts != published.counts


@pytest.mark.parametrize("retention", list(ResultRetention))
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_public_literal_adaptive_parent_preserves_every_executed_node(
    *, runtime_checks, retention
) -> None:
    """A converged adaptive mesh publishes the child slots of every new node solve."""
    model = make_literal_parent_model(
        resources=fixture.child_positive_resources, outer_search=_ADAPTIVE_MESH
    )
    with pytest.MonkeyPatch.context() as patch:
        execution = _observe_adaptive_parent(patch)
        model.solve(
            params=fixture.PARAMS,
            log_level="off",
            runtime_checks=runtime_checks,
            retention=retention,
        )
    _require_adaptive_transport(
        execution=execution,
        n_slots=_n_child_slots(model),
        runtime_checks=runtime_checks,
    )


# Each transform: whether the model jits, and its execution configuration.
_TRANSFORMS: dict[str, tuple[bool, ExecutionConfig | None]] = {
    "eager": (False, None),
    "outer_tile_2": (True, ExecutionConfig(axis_widths={OUTER_CANDIDATE_AXIS: 2})),
    "outer_covered": (True, ExecutionConfig(covered_axes=(OUTER_CANDIDATE_AXIS,))),
    "cell_tile_3": (True, ExecutionConfig(axis_widths={"cell": 3})),
}


@pytest.fixture(scope="module")
def adaptive_reference_values() -> dict[int, dict[str, np.ndarray]]:
    """The default-configuration adaptive solve's value functions."""
    model = make_literal_parent_model(
        resources=fixture.child_positive_resources, outer_search=_ADAPTIVE_MESH
    )
    solution = model.solve(params=fixture.PARAMS, log_level="off")
    return {
        period: {name: np.asarray(value) for name, value in regimes.items()}
        for period, regimes in solution.values.items()
    }


@pytest.mark.parametrize("transform", list(_TRANSFORMS))
def test_public_literal_adaptive_parent_transport_survives_execution_transforms(
    *, transform, adaptive_reference_values
) -> None:
    """Eager, tiled and covered execution publish the same law and the same values."""
    enable_jit, execution_config = _TRANSFORMS[transform]
    model = make_literal_parent_model(
        resources=fixture.child_positive_resources,
        outer_search=_ADAPTIVE_MESH,
        enable_jit=enable_jit,
        execution_config=execution_config,
    )
    with pytest.MonkeyPatch.context() as patch:
        execution = _observe_adaptive_parent(patch)
        solution = model.solve(params=fixture.PARAMS, log_level="off")
    _require_adaptive_transport(
        execution=execution, n_slots=_n_child_slots(model), runtime_checks=True
    )
    if transform == "outer_tile_2":
        assert any(chunks[-1] < 2 for chunks in execution.dispatches if chunks)
    for period, regimes in adaptive_reference_values.items():
        for name, expected in regimes.items():
            # Eager and tiled programs vectorize differently, which can move a
            # value to a representable neighbour; a changed reduction would not.
            assert_agrees_to_ulp(
                got=solution.values[period][name],
                expected=expected,
                n_ulp=4,
                err_msg=f"{transform}: period {period}, regime {name}",
            )


@pytest.fixture(scope="module")
def final_pass_probe() -> float:
    """A node first solved in the converging validation pass, left off the mesh."""
    model = make_literal_parent_model(
        resources=fixture.child_positive_resources, outer_search=_ADAPTIVE_MESH
    )
    with pytest.MonkeyPatch.context() as patch:
        execution = _observe_adaptive_parent(patch)
        model.solve(params=fixture.PARAMS, log_level="off")
    earlier = {node for request in execution.requests[:-1] for node in request}
    probes = [
        node
        for node in execution.requests[-1]
        if node not in earlier and node not in execution.final_nodes()
    ]
    assert probes, "The converging pass solved no new probe."
    return probes[0]


def _nan_child_budget_only_at(node: float):
    """Return the healthy child budget, NaN only within `_NAN_RADIUS` of `node`."""

    def resources(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
        return fixture.child_positive_resources(
            liquid=liquid, stock=stock
        ) + 0.0 * jnp.sqrt((stock - node) ** 2 - _NAN_RADIUS**2)

    return resources


@pytest.mark.parametrize("runtime_checks", [False, True])
def test_public_literal_adaptive_parent_reports_a_failed_excluded_probe(
    *, runtime_checks, final_pass_probe
) -> None:
    """A probe solved and then left off the mesh still reports its child failure."""
    model = make_literal_parent_model(
        resources=_nan_child_budget_only_at(final_pass_probe),
        outer_search=_ADAPTIVE_MESH,
    )
    with pytest.MonkeyPatch.context() as patch:
        execution = _observe_adaptive_parent(patch)
        with pytest.raises(InvalidValueFunctionError, match="query_coverage_missing"):
            model.solve(
                params=fixture.PARAMS,
                log_level="off",
                runtime_checks=runtime_checks,
            )
    assert final_pass_probe in execution.excluded_probes()
    flagged = [node for node, row in execution.rows if any(map(any, row.slots))]
    assert flagged == [final_pass_probe]


def test_public_literal_adaptive_parent_replays_with_both_debug_snapshots(
    *, tmp_path
) -> None:
    """Split replay matches automatic simulation; both debug snapshots persist."""
    model = make_literal_parent_model(
        resources=fixture.child_positive_resources, outer_search=_ADAPTIVE_MESH
    )
    with pytest.MonkeyPatch.context() as patch:
        execution = _observe_adaptive_parent(patch)
        solution = model.solve(
            params=fixture.PARAMS,
            log_level="debug",
            log_path=tmp_path / "solve",
            retention=ResultRetention.VALUES_AND_REPLAY,
        )
    _require_adaptive_transport(
        execution=execution, n_slots=_n_child_slots(model), runtime_checks=True
    )
    (solve_dir,) = sorted((tmp_path / "solve").glob("solve_snapshot_*/"))
    solve_snapshot = load_snapshot(solve_dir)
    assert isinstance(solve_snapshot, SolveSnapshot)
    snapshot_values = solve_snapshot.period_to_regime_to_V_arr
    assert snapshot_values is not None
    for period, regimes in solution.values.items():
        for name, value in regimes.items():
            np.testing.assert_array_equal(
                np.asarray(snapshot_values[period][name]),
                np.asarray(value),
            )
    replayed = model.simulate(
        params=fixture.PARAMS,
        initial_conditions=dict(_SIMULATION_INITIAL),
        solution=solution,
        log_level="debug",
        log_path=tmp_path / "simulate",
        seed=7,
    )
    (simulate_dir,) = sorted((tmp_path / "simulate").glob("simulate_snapshot_*/"))
    simulate_snapshot = load_snapshot(path=simulate_dir)
    assert isinstance(simulate_snapshot, SimulateSnapshot)
    assert simulate_snapshot.result is not None
    automatic = model.simulate(
        params=fixture.PARAMS,
        initial_conditions=dict(_SIMULATION_INITIAL),
        solution=None,
        log_level="off",
        seed=7,
    )
    assert_frame_equal(replayed.to_dataframe(), automatic.to_dataframe())


def test_public_literal_adaptive_values_only_result_refuses_replay() -> None:
    """A values-only adaptive result cannot stand in for retained replay policies."""
    model = make_literal_parent_model(
        resources=fixture.child_positive_resources, outer_search=_ADAPTIVE_MESH
    )
    solution = model.solve(
        params=fixture.PARAMS, log_level="off", retention=ResultRetention.VALUES
    )
    with pytest.raises(InvalidSimulationInputError, match=r"simulation\.policy"):
        model.simulate(
            params=fixture.PARAMS,
            initial_conditions=dict(_SIMULATION_INITIAL),
            solution=solution,
            log_level="off",
            seed=7,
        )
