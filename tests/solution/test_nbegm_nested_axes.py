"""A host-scheduled NNBEGM outer mesh preserves planning of its inner programs.

The fp32 conditional policy can vary with cell width at a certified candidate crossing.
Reproducer and exact owner/readout evidence:
https://github.com/OpenSourceEconomics/pylcm/issues/445.
"""

from dataclasses import replace
from functools import cache

import jax
import numpy as np
import pytest

from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.execution.core_program import CoreExecutionDisposition, core_program_graph
from lcm import ExecutionConfig, LinSpacedGrid
from lcm.solver_api import SolutionResult
from lcm.solvers import CELL_AXIS, OUTER_CANDIDATE_AXIS
from tests.conftest import EXACT_KERNEL_SKIP_REASON, X64_ENABLED
from tests.simulation.test_nnbegm_split_workflow_parity import _MESH
from tests.solution.test_nbegm_axes import _assert_arrays_agree, _assert_solutions_agree
from tests.test_models import n_nbegm_toy

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

# The reduced three-cell witness needs extra refinement rounds near its optimum;
# retain the existing accuracy tolerances and the fail-closed convergence check.
_OUTER_MESH = replace(_MESH, max_refinement_rounds=16)


@pytest.mark.parametrize("adaptive", [False, True])
def test_composite_keeps_inner_programs_planned(*, adaptive: bool) -> None:
    """Selecting outer nodes on the host preserves the inner cell width contract."""
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=_OUTER_MESH if adaptive else None,
        execution_config=ExecutionConfig(axis_widths={CELL_AXIS: 1}),
    )
    kernel = model._regimes["alive"].solution.period_kernels[0]
    programs = core_program_graph(kernel=kernel)
    assert programs
    for name, program in programs.items():
        assert program.disposition is CoreExecutionDisposition.PLANNED
        assert program.disposition_reason is None
        assert CELL_AXIS in program.requirements.axis_names
        assert program.requirements.host_axis_names == (
            (OUTER_CANDIDATE_AXIS,) if name.startswith("adjuster:") else ()
        )
        assert {read.target.regime for read in program.requirements.value_reads} == {
            "dead"
        }


@cache
def _solved_pair(*, adaptive: bool) -> tuple[SolutionResult, SolutionResult]:
    """Solve each immutable width configuration once for the independent assertions."""
    outer = _OUTER_MESH if adaptive else None
    reference = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=outer,
        illiquid_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=3),
        execution_config=ExecutionConfig(
            axis_widths={CELL_AXIS: 3, OUTER_CANDIDATE_AXIS: 2}
        ),
    )
    configured = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=outer,
        illiquid_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=3),
        execution_config=ExecutionConfig(
            axis_widths={CELL_AXIS: 1, OUTER_CANDIDATE_AXIS: 1}
        ),
    )
    assert (
        configured._model_structure_fingerprint
        == reference._model_structure_fingerprint
    )
    expected = reference.solve(params={"discount_factor": 0.95}, log_level="off")
    actual = configured.solve(params={"discount_factor": 0.95}, log_level="off")
    return actual, expected


@pytest.mark.parametrize("adaptive", [False, True])
def test_nested_widths_preserve_values_nodes_and_masks(*, adaptive: bool) -> None:
    """Real nested dispatch preserves values and the exact replay structure."""
    actual, expected = _solved_pair(adaptive=adaptive)
    assert actual.values.keys() == expected.values.keys()
    for period, regimes in expected.values.items():
        assert actual.values[period].keys() == regimes.keys()
        for regime, value in regimes.items():
            _assert_arrays_agree(actual=actual.values[period][regime], expected=value)
    assert actual.replay_artifacts.keys() == expected.replay_artifacts.keys()
    for ref, payload in expected.replay_artifacts.items():
        actual_payload = actual.replay_artifacts[ref]
        assert jax.tree.structure(actual_payload) == jax.tree.structure(payload)
        for got, want in zip(
            jax.tree.leaves(actual_payload), jax.tree.leaves(payload), strict=True
        ):
            got_array, want_array = np.asarray(got), np.asarray(want)
            assert got_array.shape == want_array.shape
            assert got_array.dtype == want_array.dtype
            if np.issubdtype(want_array.dtype, np.inexact):
                finite = np.isfinite(want_array)
                np.testing.assert_array_equal(np.isfinite(got_array), finite)
                np.testing.assert_array_equal(
                    np.where(finite, 0, got_array), np.where(finite, 0, want_array)
                )
            else:
                np.testing.assert_array_equal(got_array, want_array)
        if isinstance(payload, NestedEGMSimPolicy):
            assert isinstance(actual_payload, NestedEGMSimPolicy)
            np.testing.assert_array_equal(
                actual_payload.adjuster.outer_nodes, payload.adjuster.outer_nodes
            )


@pytest.mark.parametrize(
    "adaptive",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                condition=not X64_ENABLED,
                raises=AssertionError,
                strict=True,
                reason=(
                    "inherited NB-EGM cell-width sensitivity at a certified "
                    "candidate crossing; see "
                    "https://github.com/OpenSourceEconomics/pylcm/issues/445"
                ),
            ),
        ),
    ],
)
def test_nested_host_dispatch_preserves_inner_planning(*, adaptive: bool) -> None:
    """The full conditional policy has the same working-format parity obligation."""
    actual, expected = _solved_pair(adaptive=adaptive)
    _assert_solutions_agree(actual=actual, expected=expected)
