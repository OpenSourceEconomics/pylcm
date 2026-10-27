"""Actual U future inputs survive original VALUES_AND_REPLAY retention/export."""

from collections.abc import Mapping

import jax
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_case_u_replay import (
    CASE_U_FUTURE_REPLAY,
    CASE_U_REPLAY_IDENTITY,
    CaseUReplayChildren,
    CaseUReplayFutureDate,
)
from _lcm.solution.current_iid_case_u_native import (
    NativeCaseUPeriodKernel,
    acquire_installed_case_u_date,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.solver_api import ArtifactRef, ResultRetention
from lcm.typing import RegimeName
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_installation import _model_u
from tests.egm.test_current_iid_case_u_owned_core import _params

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_genuine_u_future_retention_protects_original_inputs_until_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real kernel packages originals before registry-protected retirement."""
    observed: list[CaseUReplayChildren] = []
    original = CaseUReplayFutureDate.bind

    # keyword-only-exempt: library-callback=monkeypatch.setattr
    def observe(
        self: CaseUReplayFutureDate,
        *,
        children: Mapping[RegimeName, EGMCarry],
        period: int,
    ) -> CaseUReplayChildren:
        payload = original(self, children=children, period=period)
        for target, child in children.items():
            assert payload.children[target] is child
        observed.append(payload)
        return payload

    monkeypatch.setattr(CaseUReplayFutureDate, "bind", observe)
    model = _model_u()
    _, generation = model._prepare_call(params=_params())
    prepared = acquire_installed_case_u_date(
        generation=generation, regime_name="child", period=1
    )
    owner = generation.solver_preparation(regime_name="child", period=1)
    kernel = owner.installed_kernel
    assert isinstance(kernel, NativeCaseUPeriodKernel)
    assert kernel.prepared is prepared
    future = kernel.replay_future
    assert future is not None
    assert future.source == "child"
    assert future.period == 1
    assert tuple(future.specs) == ("done",)
    assert future.specs["done"] is (
        generation.execution._regimes["done"].solution.continuation_spec
    )
    simulation = generation.execution._regimes["child"].simulation
    assert simulation.case_u_replay_spec is not None
    assert simulation.replay_route is simulation.case_u_replay_spec
    assert simulation.case_u_replay_spec.identity == CASE_U_REPLAY_IDENTITY
    assert simulation.replay_route.consumer_route == "case_u_reconstruction"
    solution = model.solve(
        params=_params(), log_level="off", retention=ResultRetention.VALUES_AND_REPLAY
    )
    payload = solution.replay_artifacts[
        ArtifactRef(period=1, regime="child", key=CASE_U_FUTURE_REPLAY)
    ]
    assert isinstance(payload, CaseUReplayChildren)
    assert payload.source == "child"
    assert payload.period == 1
    assert observed
    actual = observed[-1]
    for target in future.specs:
        before = actual.children[target]
        after = payload.children[target]
        for source, exported in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        ):
            assert isinstance(source, jax.Array)
            assert not source.is_deleted()
            np.testing.assert_array_equal(source, exported)
    with pytest.raises(InvalidSimulationInputError, match="source/date/target"):
        original(future, children=actual.children, period=2)
    with pytest.raises(InvalidSimulationInputError, match="source/date/target"):
        original(future, children={}, period=1)
