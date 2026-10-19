"""Finite policy routing authenticates the actual current-state borrowing DAG."""

from dataclasses import replace
from types import MappingProxyType

import pytest

from _lcm.continuation import EGMContinuationSpec
from _lcm.engine import NNBEGMPolicyRead
from _lcm.solution.model_authority import _validate_model_query_spec
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution.test_nested_dynamic_borrowing import _model, current_floor

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.mark.parametrize("tamper", ["missing", "bound", "inputs", "periods", "outer"])
def test_finite_policy_refuses_substituted_borrowing_authority(*, tamper: str) -> None:
    """Finite candidate ranking requires its model's exact floor and outer law."""
    model = _model(family="nnbegm", lower=current_floor)
    regime = model._regimes["alive"]
    read = regime.simulation.egm_policy_read
    assert isinstance(read, NNBEGMPolicyRead)
    authority = read.finite_query_authority
    assert authority is not None
    assert authority.finite_outer is not None
    state_action_space = regime.solution.state_action_space(
        regime_params=MappingProxyType({})
    )
    spec = regime.solution.continuation_spec
    assert isinstance(spec, EGMContinuationSpec)
    # Qualify this exact specimen before testing each independently changed field.
    _validate_model_query_spec(
        regime=regime, state_action_space=state_action_space, spec=spec
    )
    if tamper == "missing":
        changed = None
    elif tamper == "bound":
        changed = replace(
            authority, bound=replace(authority.bound, lower_target="wrong")
        )
    elif tamper == "inputs":
        changed = replace(authority, evaluator_inputs=())
    elif tamper == "periods":
        changed = replace(authority, target_periods=(999,))
    else:
        changed = replace(
            authority,
            finite_outer=replace(authority.finite_outer, outer_state="wrong"),
        )
    altered = replace(
        regime,
        simulation=replace(
            regime.simulation,
            egm_policy_read=replace(read, finite_query_authority=changed),
        ),
    )
    with pytest.raises(TypeError):
        _validate_model_query_spec(
            regime=altered,
            state_action_space=state_action_space,
            spec=spec,
        )
