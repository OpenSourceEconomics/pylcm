"""Actual U future inputs survive original VALUES_AND_REPLAY retention/export."""

import logging
from collections.abc import Callable, Mapping

import jax
import numpy as np
import pytest

from _lcm.continuation import ContinuationPayload
from _lcm.egm.carry import EGMCarry
from _lcm.egm.current_iid_case_u_replay import (
    CASE_U_FUTURE_REPLAY,
    CASE_U_REPLAY_IDENTITY,
    CaseUReplayChildren,
    CaseUReplayFutureDate,
)
from _lcm.engine import StateActionSpace
from _lcm.solution.current_iid_case_u_native import (
    NativeCaseUPeriodKernel,
    acquire_installed_case_u_date,
)
from _lcm.solution.feasibility import FeasibilityKernelOutput
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PytreeValue
from lcm.exceptions import InvalidSimulationInputError
from lcm.solver_api import ArtifactRef, ResultRetention
from lcm.typing import FloatND, RegimeName
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_case_u_installation import _model_u
from tests.egm.test_current_iid_case_u_owned_core import _params

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_genuine_u_future_retention_protects_original_inputs_until_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real kernel exports its original child inputs past their retirement."""
    observed: list[tuple[dict[RegimeName, EGMCarry], CaseUReplayChildren]] = []
    original_call = NativeCaseUPeriodKernel.__call__

    # keyword-only-exempt: library-callback=monkeypatch.setattr
    def observe(
        self: NativeCaseUPeriodKernel,
        *,
        compiled_cores: Mapping[str, Callable[..., PytreeValue]],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: TimeAxis,
        logger: logging.Logger,
    ) -> FeasibilityKernelOutput:
        # Host copies: the inputs retire after the period; the export must not.
        host = {
            name: jax.tree.map(np.asarray, child)
            for name, child in next_regime_to_continuation.items()
            if isinstance(child, EGMCarry)
        }
        output = original_call(
            self,
            compiled_cores=compiled_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
            logger=logger,
        )
        payload = output.replay.get(CASE_U_FUTURE_REPLAY)
        if payload is not None:
            assert isinstance(payload, CaseUReplayChildren)
            observed.append(
                ({name: host[name] for name in payload.child_names}, payload)
            )
        return output

    monkeypatch.setattr(NativeCaseUPeriodKernel, "__call__", observe)
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
    originals, actual = observed[-1]
    assert actual.child_names == payload.child_names
    for target in future.specs:
        before = originals[target]
        after = payload.children[target]
        for source, exported in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        ):
            assert isinstance(exported, jax.Array)
            assert not exported.is_deleted()
            np.testing.assert_array_equal(exported, source)
    with pytest.raises(InvalidSimulationInputError, match="source/date/target"):
        CaseUReplayFutureDate.bind(future, children=originals, period=2)
    with pytest.raises(InvalidSimulationInputError, match="source/date/target"):
        CaseUReplayFutureDate.bind(future, children={}, period=1)
