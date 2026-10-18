"""A literal nested parent consumes its actual callable-floor child reader."""

import numpy as np
import pytest

from _lcm.egm.dcegm_query import DCEGMQuery
from _lcm.egm.step import _EGMStep
from _lcm.execution.core_program import core_program_graph
from _lcm.solution.dcegm import _DCEGMPeriodKernel
from _lcm.solution.negm import _NEGMPeriodKernel
from lcm.solver_api import SolutionResult
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_germ_query_fixture import make_canonical_model

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_PARAMS = {"discount_factor": 1.0}


def _assert_child_terminal_values(
    *, solution: SolutionResult, literal_child: bool
) -> None:
    """The terminal-zero child consumes its budget at the winning outer target."""
    np.testing.assert_array_equal(solution.values[2]["done"], np.zeros((4, 2)))
    liquid = np.asarray([0.0, 0.5, 1.0, 2.0])[:, None]
    stock = np.asarray([0.0, 1.0])[None, :]
    floor = np.zeros_like(stock) if literal_child else stock / 4.0
    candidates = []
    for target in (stock, np.ones_like(stock), 2.0 * np.ones_like(stock)):
        consumption = liquid + 1.25 - (target - stock) ** 2 / 4.0 - floor
        delta = consumption - 1.0
        candidates.append(
            np.log(consumption)
            + (delta**3 / 6.0 + delta**4 / 2.0) * np.exp(-100.0 * delta**2)
            - (target - 1.0) ** 2
        )
    np.testing.assert_array_almost_equal(
        solution.values[1]["child"],
        np.maximum.reduce(candidates),
        decimal=DECIMAL_PRECISION,
    )


def test_literal_nested_chain_keeps_default_metadata_and_solves() -> None:
    """No installed query leaves both metadata mappings absent on a literal chain."""
    model = make_canonical_model(literal_child=True)
    for name in ("parent", "child"):
        solution = model._regimes[name].solution
        assert solution.query_authority is None
        assert solution.continuation_reader_capabilities is None
        assert solution.query_input_authorities is None
        for kernel in solution.period_kernels.values():
            assert isinstance(kernel, _NEGMPeriodKernel)
            assert kernel.floor_preparation is None
            for inner in (kernel.keeper_kernel, kernel.adjuster_kernel):
                assert isinstance(inner, _DCEGMPeriodKernel)
                assert inner.retain_nested_feasibility is False
                assert inner.own_bound_specs == ()
                assert inner.query_input_authorities is None
    actual = model.solve(params=_PARAMS, log_level="off", runtime_checks=False)
    _assert_child_terminal_values(solution=actual, literal_child=True)


def test_literal_parent_declares_both_real_child_readers_without_own_floor() -> None:
    """Each reached period uses the child's canonical authority on both cost paths."""
    model = make_canonical_model()
    parent = model._regimes["parent"]
    child = model._regimes["child"]
    assert parent.solution.query_authority is None
    assert child.solution.query_authority is not None
    assert tuple(parent.active_periods) == (0,)
    for period in parent.active_periods:
        kernel = parent.solution.period_kernels[period]
        assert isinstance(kernel, _NEGMPeriodKernel)
        assert kernel.floor_preparation is None
        expected_slots = (("child", period + 1, child.solution.query_authority.bound),)
        readers = []
        for inner in (kernel.keeper_kernel, kernel.adjuster_kernel):
            assert isinstance(inner, _DCEGMPeriodKernel)
            assert inner.retain_nested_feasibility is False
            assert inner.own_bound_specs == ()
            assert inner.query_input_authorities == expected_slots
            function = core_program_graph(kernel=inner)["replay"].function
            assert isinstance(function, _EGMStep)
            plan = function.pieces.continuation_plan
            assert plan.query_targets == ("child",)
            query = plan.child_reads["child"].query
            assert isinstance(query, DCEGMQuery)
            assert query.model.authority is child.solution.query_authority
            assert query.capabilities.value is True
            assert query.capabilities.marginal_states == frozenset({"resources"})
            readers.append(query.capabilities)
        assert readers[0] == readers[1]
        assert parent.solution.continuation_reader_capabilities is not None
        assert parent.solution.query_input_authorities is not None
        assert parent.solution.continuation_reader_capabilities[period] == {
            "child": readers[0]
        }
        assert parent.solution.query_input_authorities[period] == expected_slots


@pytest.mark.parametrize("runtime_checks", [False, True], ids=["mandatory", "checked"])
def test_literal_parent_public_solve_consumes_its_callable_child(
    *, runtime_checks: bool
) -> None:
    """Healthy child values cross the literal parent through the public solve."""
    # Complete the genuine all-literal control before reaching child-only transport.
    literal_model = make_canonical_model(literal_child=True)
    literal = literal_model.solve(
        params=_PARAMS, log_level="off", runtime_checks=runtime_checks
    )
    _assert_child_terminal_values(solution=literal, literal_child=True)
    model = make_canonical_model()
    actual = model.solve(params=_PARAMS, log_level="off", runtime_checks=runtime_checks)
    _assert_child_terminal_values(solution=actual, literal_child=False)
