"""Sampled scalar producers retain support cuts and excluded native topology."""

import functools
from dataclasses import replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.query_data import QueryNodeKind
from _lcm.execution.core_program import core_program_graph
from _lcm.solution import nbegm as nb
from _lcm.solution.egm import _RegimeMassGuardedCore
from lcm import Model, ref
from lcm.typing import RegimeName
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import RegimeId
from tests.egm.test_nb_scalar_sampled_run_producer import _model
from tests.test_models import nbegm_medicaid_toy

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _produce(
    *, kind: Literal["continuous_kink", "jump"]
) -> tuple[EGMCarry, nb._NBEGMContinuousCore]:
    """Observe actual raw rows immediately before their installed normalization."""
    observed = []
    original = nb._with_scalar_nbegm_sampled_run

    def capture(*, carry: EGMCarry, core: nb._RawNBEGMCore) -> EGMCarry:
        def save(actual: EGMCarry) -> None:
            observed.append((actual, core))

        jax.debug.callback(save, carry)
        return original(carry=carry, core=core)

    model = _model(kind=kind, callable_floor=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nb, "_with_scalar_nbegm_sampled_run", capture)
        solution = model.solve(
            params={"discount_factor": 1.0, "rate": 0.2, "threshold": 1.2},
            log_level="off",
        )
        jax.effects_barrier()
    assert len(observed) == 1
    carry, core = observed[0]
    carry = jax.tree.map(jnp.asarray, carry)
    assert isinstance(core, nb._NBEGMContinuousCore)
    assert core.schedule_spec.breakpoint_kinds == (kind,)
    assert core.feasibility_constraints == ()
    data = carry.query_data_for_transport()
    assert data is not None
    np.testing.assert_array_equal(carry.endog_grid, [0.75, 1.5, 3.0])
    np.testing.assert_array_equal(data.node_kind, [QueryNodeKind.FINITE] * 3)
    np.testing.assert_array_equal(data.owner_id, [0, 1, 1])
    np.testing.assert_array_equal(data.side_id, [0, 0, 0])
    np.testing.assert_array_equal(data.link_valid, [False, True])
    np.testing.assert_array_equal(data.floor_used, 0.5)
    np.testing.assert_array_equal(carry.value, solution.values[1]["child"])
    return carry, core


@pytest.fixture(scope="module")
def continuous_source() -> tuple[EGMCarry, nb._NBEGMContinuousCore]:
    return _produce(kind="continuous_kink")


@pytest.fixture(scope="module")
def jump_source() -> tuple[EGMCarry, nb._NBEGMContinuousCore]:
    return _produce(kind="jump")


def _normalize(*, carry: EGMCarry, core: nb._RawNBEGMCore, compiled: bool) -> EGMCarry:
    read = functools.partial(nb._with_scalar_nbegm_sampled_run, core=core)
    return (jax.jit(read) if compiled else read)(carry=carry)


def _assert_preserved(*, actual: EGMCarry, original: EGMCarry) -> None:
    """Every numeric/support leaf except the two declared outputs stays exact."""
    actual_leaves = actual.leaves()
    original_leaves = original.leaves()
    assert actual_leaves.keys() == original_leaves.keys()
    for path, expected in original_leaves.items():
        if path not in (("query_data", "owner_id"), ("query_data", "link_valid")):
            assert actual_leaves[path].dtype == expected.dtype
            np.testing.assert_array_equal(actual_leaves[path], expected)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("cut", ["none", "pad", "same_owner_false"])
def test_sampled_producer_preserves_independent_support_cuts(
    *, continuous_source, compiled: bool, cut: str
) -> None:
    """A continuous budget join cannot fill a missing node or an independent gap."""
    carry, core = continuous_source
    data = carry.query_data_for_transport()
    assert data is not None
    if cut == "pad":
        data = replace(
            data,
            node_kind=data.node_kind.at[1].set(QueryNodeKind.PAD),
            owner_id=data.owner_id.at[1].set(-1),
            link_valid=jnp.asarray([False, False]),
        )
        expected_owners = [0, -1, 0]
        expected_links = [False, False]
    elif cut == "same_owner_false":
        # Original nodes1/2 share real budget owner1; remove only their support.
        np.testing.assert_array_equal(data.owner_id[1:], [1, 1])
        data = replace(data, link_valid=data.link_valid.at[1].set(False))
        expected_owners = [0, 0, 0]
        expected_links = [True, False]
    else:
        expected_owners = [0, 0, 0]
        expected_links = [True, True]
    changed = replace(carry, query_data=data)
    result = _normalize(carry=changed, core=core, compiled=compiled)
    _assert_preserved(actual=result, original=changed)
    actual = result.query_data_for_transport()
    assert actual is not None
    np.testing.assert_array_equal(actual.owner_id, expected_owners)
    np.testing.assert_array_equal(actual.link_valid, expected_links)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("guard", ["one_sided", "negative_owner", "owner_overflow"])
def test_sampled_producer_preserves_whole_noncanonical_record(
    *, continuous_source, compiled: bool, guard: str
) -> None:
    """One represented noncanonical record prevents row-wide relabeling."""
    carry, core = continuous_source
    data = carry.query_data_for_transport()
    assert data is not None
    if guard == "one_sided":
        data = replace(data, side_id=data.side_id.at[1].set(1))
    else:
        owner = -1 if guard == "negative_owner" else 2
        assert owner < 0 or owner > len(core.schedule_spec.breakpoint_kinds)
        data = replace(data, owner_id=data.owner_id.at[1].set(owner))
    changed = replace(carry, query_data=data)
    result = _normalize(carry=changed, core=core, compiled=compiled)
    _assert_preserved(actual=result, original=changed)
    actual = result.query_data_for_transport()
    assert actual is not None
    np.testing.assert_array_equal(actual.owner_id, data.owner_id)
    np.testing.assert_array_equal(actual.link_valid, data.link_valid)


def _installed_raw_core(*, model: Model, regime_name: RegimeName, period: int):
    function = core_program_graph(
        kernel=model._regimes[regime_name].solution.period_kernels[period]
    )["main"].function
    seen = set()
    while not isinstance(function, nb._NBEGMContinuousCore | nb._NBEGMCaseCore):
        assert id(function) not in seen
        seen.add(id(function))
        if isinstance(function, nb._NBEGMOwnedSingleAxisCore):
            function = function.inner
        elif isinstance(function, _RegimeMassGuardedCore):
            function = function.core
        elif isinstance(function, functools.partial):
            function = function.func
        else:
            assert hasattr(function, "__wrapped__")
            function = function.__wrapped__
    return function


@pytest.fixture(scope="module")
def excluded_cores():
    base = _model(kind="continuous_kink", callable_floor=False)
    child = base.user_regimes["child"]
    child = replace(
        child, constraints={**child.constraints, "eligible": ref("liquid") >= 1.0}
    )
    boundary_model = Model(
        regimes={**base.user_regimes, "child": child},
        ages=base.ages,
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
    )
    boundary = _installed_raw_core(model=boundary_model, regime_name="child", period=1)
    assert isinstance(boundary, nb._NBEGMContinuousCore)
    assert boundary.feasibility_constraints
    case_model = nbegm_medicaid_toy.build_model(variant="nbegm")
    case = _installed_raw_core(model=case_model, regime_name="alive", period=0)
    assert isinstance(case, nb._NBEGMCaseCore)
    return {"compiled_boundary": boundary, "case": case}


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("route", ["jump", "case", "compiled_boundary"])
def test_sampled_producer_retains_excluded_native_routes(
    *, continuous_source, jump_source, excluded_cores, compiled: bool, route: str
) -> None:
    """Real jump, case and compiled-boundary cores retain the full input record."""
    if route == "jump":
        carry, core = jump_source
    else:
        carry = continuous_source[0]
        core = excluded_cores[route]
    result = _normalize(carry=carry, core=core, compiled=compiled)
    _assert_preserved(actual=result, original=carry)
    original = carry.query_data_for_transport()
    actual = result.query_data_for_transport()
    assert original is not None
    assert actual is not None
    np.testing.assert_array_equal(actual.owner_id, original.owner_id)
    np.testing.assert_array_equal(actual.link_valid, original.link_valid)
