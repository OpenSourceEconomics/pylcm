"""The outer post-decision sweep streams candidates under a planner axis.

The sweep keeps every candidate carry, so the width only reschedules the map
over the outer nodes: the maximum runs over the same values in the same order
whatever the blocks are. What the width does change is the vector width the
inner adjuster is compiled for, and XLA emits a differently vectorized kernel
per width, so the published values land on representable neighbours rather than
on the same bit pattern. The bound is therefore in units of the working
format's spacing, and set from what those neighbours actually are rather than
from a defensive round number.
"""

from typing import Any

import numpy as np
import pytest

import _lcm.solution.nnbegm as nnbegm_module
from _lcm.execution.core_program import core_program_graph
from lcm import ExecutionConfig
from lcm.solvers import (
    NEGM,
    OUTER_CANDIDATE_AXIS,
    AdaptiveOuterMesh,
    FiniteOuterGrid,
)
from tests.conftest import assert_agrees_to_ulp
from tests.simulation.test_nnbegm_split_workflow_parity import _MESH
from tests.test_models import n_nbegm_toy, negm_kinked_toy

_PARAMS: dict[str, Any] = {"discount_factor": 0.95, "alive": {}}
_N_OUTER = negm_kinked_toy.N_AZ


def test_negm_has_no_outer_batch_size_field() -> None:
    """The outer block width is an execution fact, not a solver field."""
    with pytest.raises(TypeError, match="outer_batch_size"):
        NEGM(
            inner=negm_kinked_toy.NEGM_SOLVER.inner,
            outer_grid=negm_kinked_toy.OUTER_GRID,
            outer_batch_size=4,  # ty: ignore[unknown-argument]
        )


def test_finite_outer_grid_has_no_batch_size_field() -> None:
    """A finite outer grid is a candidate set only."""
    with pytest.raises(TypeError, match="batch_size"):
        FiniteOuterGrid(
            grid=negm_kinked_toy.OUTER_GRID,
            batch_size=2,  # ty: ignore[unknown-argument]
        )


def test_adaptive_outer_mesh_has_no_batch_size_field() -> None:
    """An adaptive outer mesh describes refinement only."""
    with pytest.raises(TypeError, match="batch_size"):
        AdaptiveOuterMesh(
            initial_grid=negm_kinked_toy.OUTER_GRID,
            batch_size=2,  # ty: ignore[unknown-argument]
        )


def test_outer_sweep_declares_the_outer_candidate_axis() -> None:
    """The NEGM outer sweep program declares `outer_candidate` as a reduced axis."""
    model = negm_kinked_toy.build_model()
    kernel = next(iter(model._regimes["alive"].solution.period_kernels.values()))
    program = core_program_graph(kernel=kernel)["outer_sweep"]

    assert program.requirements.axis_names == (OUTER_CANDIDATE_AXIS,)


def test_outer_candidate_axis_spans_the_outer_grid() -> None:
    """The axis enumerates exactly the exogenous outer post-decision nodes."""
    model = negm_kinked_toy.build_model()
    kernel = next(iter(model._regimes["alive"].solution.period_kernels.values()))
    graph = core_program_graph(kernel=kernel)
    axis = graph["outer_sweep"].requirements.reduced_axes[0]

    assert axis.extent == _N_OUTER


def _solve(*, widths: dict[str, int]) -> Any:
    """Solve the kinked NEGM toy with the named axis widths fixed."""
    return negm_kinked_toy.build_model(
        execution_config=ExecutionConfig(axis_widths=widths)
    ).solve(params=_PARAMS, log_level="off")


@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_outer_sweep_value_is_identical_across_widths(*, width: int) -> None:
    """A hard max with lowest-id tie-break does not depend on the block schedule.

    The bound is twice the worst gap measured over these widths against the
    whole-axis solve: 1 ULP at float64 and 2 at float32, both at the narrowest
    widths. A partition-dependent reduction would miss by orders of magnitude
    more, so the doubling leaves the test discriminating.
    """
    reference = _solve(widths={OUTER_CANDIDATE_AXIS: _N_OUTER})
    streamed = _solve(widths={OUTER_CANDIDATE_AXIS: width})

    assert_agrees_to_ulp(
        got=np.asarray(streamed.values[1]["alive"]),
        expected=np.asarray(reference.values[1]["alive"]),
        n_ulp=4,
        err_msg=f"period-1 value at outer_candidate width {width}",
    )


@pytest.mark.parametrize("route", ["finite", "adaptive"])
@pytest.mark.parametrize("width", [None, 2])
def test_model_configuration_reaches_the_nested_host_dispatch(
    *, route: str, width: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured width partitions every finite or adaptive host dispatch."""
    observed: list[tuple[int, int | None, tuple[tuple[int, int], ...]]] = []
    dispatch_bounds = nnbegm_module._dispatch_bounds

    def record_bounds(
        *, n_items: int, width: int | None
    ) -> tuple[tuple[int, int], ...]:
        bounds = dispatch_bounds(n_items=n_items, width=width)
        observed.append((n_items, width, bounds))
        return bounds

    monkeypatch.setattr(nnbegm_module, "_dispatch_bounds", record_bounds)
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=_MESH if route == "adaptive" else None,
        execution_config=ExecutionConfig(
            axis_widths={} if width is None else {OUTER_CANDIDATE_AXIS: width}
        ),
    )
    model.solve(params={"discount_factor": 0.95}, log_level="off")

    expected = [
        (
            n_items,
            width,
            tuple(
                (start, min(start + (width or n_items), n_items))
                for start in range(0, n_items, width or n_items)
            ),
        )
        for n_items, _actual_width, _bounds in observed
    ]
    assert (bool(observed), observed) == (True, expected)


def test_host_dispatch_width_preserves_the_durable_model_identity() -> None:
    """Changing only a host dispatch width keeps the model fingerprint."""
    reference = n_nbegm_toy.build_model(variant="n_nbegm", n_periods=2)
    configured = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        execution_config=ExecutionConfig(axis_widths={OUTER_CANDIDATE_AXIS: 2}),
    )

    assert (
        configured._model_structure_fingerprint
        == reference._model_structure_fingerprint
    )
