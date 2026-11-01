"""Held genuine fixed/grid preparation; full cold Model admission remains closed."""

from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType
from typing import Literal, cast

import jax
import numpy as np
import pytest

from _lcm.egm.validation import (
    _fail_if_grid_hygiene_violated,
    fail_if_declared_lower_bound_disagrees_with_the_grid,
)
from _lcm.execution.workspace_planning import WorkspaceBudgetError
from _lcm.grids import (
    DiscreteGrid,
    Grid,
    GridBreakpoint,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
)
from _lcm.processes.iid import NormalIIDProcess, UniformIIDProcess
from _lcm.solution.dcegm import _BoundDCEGM
from _lcm.solution.generation import canonical_generation_roots
from lcm.exceptions import (
    ExecutionPlanningError,
    GridInitializationError,
    ModelInitializationError,
)
from lcm.typing import Float1D
from tests.egm.test_current_iid_generation_materialization import _owner
from tests.solution.test_post_decision_lower_bound import _model as _bounded_model

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LinSpacedGrid(start=0.125, stop=13.75, n_points=7),
        lambda: LogSpacedGrid(start=0.125, stop=13.75, n_points=7),
        lambda: IrregSpacedGrid(points=np.array([0.125, 0.75, 13.75])),
        lambda: DiscreteGrid(values=np.array([0.125, 0.75, 13.75])),
        lambda: PiecewiseLinSpacedGrid(
            start=0.125,
            stop=13.75,
            breakpoints=(GridBreakpoint(value=0.75, owner="left"),),
            points_per_segment=(3, 4),
        ),
        lambda: PiecewiseLogSpacedGrid(
            start=0.125,
            stop=13.75,
            breakpoints=(GridBreakpoint(value=0.75, owner="right"),),
            points_per_segment=(3, 4),
        ),
        lambda: UniformIIDProcess(start=0.125, stop=13.75, n_points=7),
        lambda: NormalIIDProcess(mu=0.125, sigma=1.75, n_points=7, gauss_hermite=False),
    ],
)
def test_original_grid_bits_and_original_operand_custody(
    *, factory: Callable[[], Grid]
) -> None:
    """Separate log/endpoint stages must preserve durable support coordinates."""
    spec = factory()
    expected = np.asarray(spec.to_jax())
    original_roots = canonical_generation_roots(declarations=(spec,))
    owner = _owner()
    try:
        actual = owner.materialize_grid(spec=spec)
        assert actual.dtype == expected.dtype
        assert np.asarray(actual).tobytes() == expected.tobytes()
        assert owner.materialize_grid(spec=spec) is actual
        held = jax.tree.leaves(owner.solve_input_roots())
        assert all(any(root is value for value in held) for root in original_roots)
        if isinstance(spec, LogSpacedGrid | PiecewiseLogSpacedGrid):
            assert np.asarray(actual)[0] == np.asarray(spec.start)
            assert np.asarray(actual)[-1] == np.asarray(spec.stop)
    finally:
        owner.close()


def test_grid_factory_refuses_real_compiler_reservation_before_output() -> None:
    spec = LinSpacedGrid(start=0.125, stop=13.75, n_points=4096)
    owner = _owner()
    owner.budget_bytes = 64
    try:
        with pytest.raises((WorkspaceBudgetError, ExecutionPlanningError)):
            owner.materialize_grid(spec=spec)
        assert not owner._grid_bindings
        assert not owner._pending
        assert np.asarray(spec.start) == 0.125
        assert np.asarray(spec.stop) == 13.75
    finally:
        owner.close()


class _UnprofiledGrid(Grid):
    def to_jax(self) -> Float1D:
        raise AssertionError("Unprofiled grid producer must never run.")


def test_unprofiled_grid_refuses_before_original_factory() -> None:
    owner = _owner()
    try:
        with pytest.raises(ExecutionPlanningError, match="construction route"):
            owner.materialize_grid(spec=_UnprofiledGrid())
        assert not owner.operations.cache
    finally:
        owner.close()


def test_runtime_irregular_placeholder_is_explicit_and_not_resolved() -> None:
    spec = IrregSpacedGrid(n_points=3)
    owner = _owner()
    try:
        actual = owner.materialize_grid(spec=spec, placeholder=True)
        assert actual.shape == (3,)
        assert np.isnan(np.asarray(actual)).all()
        with pytest.raises(GridInitializationError, match="runtime"):
            # The original missing-points error is not replaced by a fabricated bank.
            owner.materialize_grid(spec=spec)
    finally:
        owner.close()


def test_original_value_grid_cache_roots_are_charged_before_code_upload() -> None:
    spec = DiscreteGrid(values=np.array([0.125, 0.75, 13.75]))
    values = spec.values_to_jax()
    owner = _owner()
    owner.budget_bytes = values.nbytes - 1
    try:
        with pytest.raises(ExecutionPlanningError, match="before allocation"):
            owner.materialize_grid(spec=spec)
        assert not owner._pending
        assert any(root is values for root in owner._grid_roots)
        np.testing.assert_array_equal(values, np.array([0.125, 0.75, 13.75]))
    finally:
        owner.close()


@pytest.mark.parametrize("stage", ["params", "initial"])
def test_published_unrelated_tree_keeps_cached_grid_charged(
    *, stage: Literal["params", "initial"]
) -> None:
    """Publishing a different tree must not uncharge a still-owned support."""
    spec = LinSpacedGrid(start=0.125, stop=13.75, n_points=4096)
    owner = _owner()
    try:
        grid = owner.materialize_grid(spec=spec)
        expected = np.asarray(grid).copy()
        owner.publish(stage=stage, tree=MappingProxyType({}))
        assert not owner._pending
        # A cache hit does not need to republish the grid to be accountable.
        assert owner.materialize_grid(spec=spec) is grid
        roots = jax.tree.leaves(owner.solve_input_roots())
        assert any(root is grid for root in roots)
        live = owner.snapshot()
        device = owner.devices[0]
        assert sum(stop - start for start, stop in live.spans[device]) >= grid.nbytes
        owner.budget_bytes = grid.nbytes - 1
        with pytest.raises(ExecutionPlanningError, match="before allocation"):
            owner(
                value=np.ones(1),
                dtype=np.dtype(grid.dtype),
                name="unrelated_next_leaf",
            )
        assert not owner._pending
        np.testing.assert_array_equal(grid, expected)
    finally:
        owner.close()


def test_dcegm_grid_hygiene_keeps_original_coverage_error() -> None:
    """A savings grid ending below the Euler grid remains a named model error."""
    model = _bounded_model(savings_grid_start=-2.0, declared_lower_bound=-2.0)
    regime = model.user_regimes["saving"]
    solver = replace(
        cast("_BoundDCEGM", regime.solver),
        savings_grid=LinSpacedGrid(start=-2.0, stop=3.0, n_points=8),
    )
    with pytest.raises(ModelInitializationError) as original:
        _fail_if_grid_hygiene_violated(
            regime_name="saving", user_regime=regime, solver=solver
        )
    assert "3.0" in str(original.value)
    assert "4.0" in str(original.value)


@pytest.mark.parametrize("irregular", [False, True])
def test_declared_lower_bound_keeps_original_exact_grid_error(
    *, irregular: bool
) -> None:
    """The bound is compared at the actual grid dtype and retains its diagnostic."""
    model = _bounded_model(savings_grid_start=-2.0, declared_lower_bound=-2.0)
    regime = model.user_regimes["saving"]
    grid = (
        IrregSpacedGrid(points=np.array([-0.5, 0.0, 4.0]))
        if irregular
        else LinSpacedGrid(start=-0.5, stop=4.0, n_points=8)
    )
    solver = replace(cast("_BoundDCEGM", regime.solver), savings_grid=grid)
    with pytest.raises(ModelInitializationError) as original:
        fail_if_declared_lower_bound_disagrees_with_the_grid(
            regime_name="saving",
            user_regime=regime,
            solver=solver,
            solver_name="DCEGM",
        )
    assert "-2.0" in str(original.value)
    assert "-0.5" in str(original.value)
