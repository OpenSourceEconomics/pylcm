"""Adaptive continuation rows retain their exact conditional inner solves."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Unpack, cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import EGMContinuationSpec
from _lcm.egm.carry import EGMCarry
from _lcm.egm.outer_candidates import OuterCandidateBank, OuterCandidateResult
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.solution import nnbegm as nested
from _lcm.solution.feasibility import FeasibilityKernelOutput
from lcm.solver_api import EGM_CONTINUATION, SIMULATION_POLICY, KernelOutput
from lcm.typing import Float1D, FloatND
from tests.conftest import DECIMAL_PRECISION
from tests.egm.test_adaptive_discrete_profiles import _adaptive_model
from tests.solution.test_nbegm_retention_specialization import _NativeKernelKwargs
from tests.test_models import n_nbegm_toy as smooth


@dataclass(frozen=True, kw_only=True)
class _ActualRecords:
    """References captured after genuine producer calls, without replacing results."""

    exact: dict[float, OuterCandidateResult] = field(default_factory=dict)
    """All nodes requested by the real adaptive driver, including validation probes."""
    banks: list[OuterCandidateBank] = field(default_factory=list)
    """Completed active banks in their production order."""
    keeper: dict[int, KernelOutput] = field(default_factory=dict)
    """Exact no-adjustment outputs by period."""
    outputs: dict[int, KernelOutput] = field(default_factory=dict)
    """Actual nested period outputs."""


def _record_producers(*, monkeypatch: pytest.MonkeyPatch) -> _ActualRecords:
    """Observe the existing callback, bank, keeper and period-output seams."""
    records = _ActualRecords()
    solve_nodes = nested._AdaptiveNodeSolver.__call__
    build_bank = nested.build_outer_candidate_bank
    solve_keeper = nested._NNBEGMPeriodKernel._solve_keeper
    solve_period = nested._NNBEGMPeriodKernel.__call__

    # keyword-only-exempt: library-callback=_lcm.egm.outer_refinement.refine_outer_mesh
    def record_nodes(self: nested._AdaptiveNodeSolver, nodes_arr: Float1D) -> FloatND:
        values = solve_nodes(self, nodes_arr)
        records.exact.update(self.cache)
        return values

    def record_bank(
        *, outer_nodes: FloatND, results: Sequence[OuterCandidateResult]
    ) -> OuterCandidateBank:
        bank = build_bank(outer_nodes=outer_nodes, results=results)
        records.banks.append(bank)
        return bank

    def record_keeper(
        self: nested._NNBEGMPeriodKernel,
        **kwargs: Unpack[_NativeKernelKwargs],
    ) -> KernelOutput:
        result = solve_keeper(self, **kwargs)
        records.keeper[kwargs["period"]] = result
        return result

    def record_period(
        self: nested._NNBEGMPeriodKernel,
        **kwargs: Unpack[_NativeKernelKwargs],
    ) -> KernelOutput:
        result = solve_period(self, **kwargs)
        records.outputs[kwargs["period"]] = result
        return result

    monkeypatch.setattr(nested._AdaptiveNodeSolver, "__call__", record_nodes)
    monkeypatch.setattr(nested, "build_outer_candidate_bank", record_bank)
    monkeypatch.setattr(nested._NNBEGMPeriodKernel, "_solve_keeper", record_keeper)
    monkeypatch.setattr(nested._NNBEGMPeriodKernel, "__call__", record_period)
    return records


def _assert_same_slot(*, published: EGMCarry, source: EGMCarry, slot: int) -> None:
    """A named outer slot retains each original conditional numeric leaf exactly."""
    for actual, expected in (
        (published.endog_grid, source.endog_grid),
        (published.value, source.value),
        (published.marginal_utility, source.marginal_utility),
    ):
        np.testing.assert_array_equal(actual[..., slot, :], expected)
    data = published.query_data
    original = source.query_data
    assert isinstance(data, EGMQueryData)
    assert isinstance(original, EGMQueryData)
    saved = data.leaves()
    raw = original.leaves()
    assert {path for path in saved if path[0] != "adaptive_outer_data"} == set(raw)
    for path, expected in raw.items():
        actual = saved[path]
        if path == ("floor_used",):
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_array_equal(actual[..., slot, :], expected)


def _assert_native_tuple(*, carry: EGMCarry, target: FloatND) -> None:
    """Native M, original saving and compact W share the same genuine D row."""
    data = carry.query_data
    assert isinstance(data, EGMQueryData)
    assert carry.policy is not None
    assert carry.value.shape == (smooth.N_ILLIQUID, 2, smooth.N_WEALTH)
    assert data.savings.shape == carry.value.shape
    assert data.floor_used.shape == (smooth.N_ILLIQUID,)
    np.testing.assert_array_equal(data.floor_used, np.zeros(smooth.N_ILLIQUID))
    wealth = np.asarray(smooth.WEALTH_GRID.to_jax())[None, None, :]
    current = np.asarray(smooth.ILLIQUID_GRID.to_jax())[:, None, None]
    codes = np.asarray((0, 1))[None, :, None]
    resources = wealth + smooth.LABOUR_INCOME - (np.asarray(target) - current) - codes
    consumption = np.asarray(carry.policy)
    finite = np.asarray(data.node_kind) == QueryNodeKind.FINITE
    assert np.any(finite[:, 0, :])
    assert np.any(finite[:, 1, :])
    np.testing.assert_array_almost_equal(
        np.asarray(data.savings)[finite],
        (resources - consumption)[finite],
        decimal=DECIMAL_PRECISION,
    )
    assert data.liquid_corner_data is None
    assert data.corner_continuation is not None
    assert data.corner_kind is not None
    assert data.corner_continuation.shape == (smooth.N_ILLIQUID, 2, 1)
    terminal_durable = -smooth.TERMINAL_SCALE / (
        np.asarray(smooth.ILLIQUID_GRID.to_jax()) + 1.0
    )
    expected_corner = -smooth.TERMINAL_SCALE + np.interp(
        np.asarray(target).reshape(-1),
        np.asarray(smooth.ILLIQUID_GRID.to_jax()),
        terminal_durable,
    )
    expected_corner = np.broadcast_to(
        expected_corner.reshape(-1, 1, 1), data.corner_continuation.shape
    )
    np.testing.assert_array_almost_equal(
        data.corner_continuation, expected_corner, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(data.corner_kind, QueryNodeKind.FINITE)


def test_adaptive_publication_keeps_the_actual_raw_action_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keeper and refined adjuster slots preserve conditional V/M/A and floor W."""
    model = _adaptive_model(callable_floor=True)
    records = _record_producers(monkeypatch=monkeypatch)
    model.solve(
        params={"discount_factor": 0.95, "alive": {"premium": 1.0}},
        log_level="off",
    )
    assert set(records.outputs) == {0}
    assert set(records.keeper) == {0}
    assert len(records.banks) == 1
    output = records.outputs[0]
    assert isinstance(output, FeasibilityKernelOutput)
    assert len(output.bound_status) == 1
    bank = records.banks[0]
    continuation = model._regimes["alive"].solution.continuation_spec
    assert isinstance(continuation, EGMContinuationSpec)
    authority = continuation.layout.query_spec
    assert authority is not None
    assert authority.outer_mode == "adaptive_continuous"
    assert authority.action_names == ("buy_private",)
    assert authority.state_names == ("illiquid",)
    assert authority.adaptive_outer is not None
    assert authority.adaptive_outer.adjuster_capacity == 513
    assert authority.adaptive_outer.golden_iterations == 40
    published = records.outputs[0].continuations[EGM_CONTINUATION]
    assert isinstance(published, EGMCarry)
    data = published.query_data
    assert isinstance(data, EGMQueryData)
    mesh = data.adaptive_outer_data
    assert mesh is not None
    count = int(mesh.outer_count)
    assert count == bank.n_candidates
    assert smooth.N_OUTER <= count <= 513
    assert published.value.shape == (smooth.N_ILLIQUID, 2, 514, smooth.N_WEALTH)
    np.testing.assert_array_equal(mesh.outer_nodes[:count], bank.outer_nodes)
    assert bool(jnp.all(jnp.diff(mesh.outer_nodes[:count]) > 0))
    np.testing.assert_array_equal(mesh.outer_node_kind[count:], QueryNodeKind.PAD)
    np.testing.assert_array_equal(
        data.node_kind[..., count + 1 :, :], QueryNodeKind.PAD
    )
    np.testing.assert_array_equal(data.link_valid[..., count + 1 :, :], desired=False)
    keeper = cast("EGMCarry", records.keeper[0].continuations[EGM_CONTINUATION])
    _assert_native_tuple(
        carry=keeper, target=smooth.ILLIQUID_GRID.to_jax()[:, None, None]
    )
    _assert_same_slot(published=published, source=keeper, slot=0)
    for index, node in enumerate(np.asarray(bank.outer_nodes)):
        source = records.exact[float(node)]
        np.testing.assert_array_equal(source.outer_node, node)
        _assert_native_tuple(carry=source.carry, target=jnp.asarray(node))
        _assert_same_slot(published=published, source=source.carry, slot=index + 1)
    assert SIMULATION_POLICY not in records.outputs[0].replay
