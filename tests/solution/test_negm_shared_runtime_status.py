"""Genuine admitted DC runtime owners retain integrity beside empty current V.

These exercise the shared publication/status owner with its default policy.
Active NEGM runtime-floor admission remains an explicitly separate frontier.
"""

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution.dcegm import _BoundDCEGM
from _lcm.solution.feasibility import (
    FeasibilityKernelOutput,
    validate_query_input_status,
    validate_runtime_bound_status,
)
from _lcm.typing import FlatRegimeParams
from lcm import LiquidMargin, Model, post_decision_lower_bound
from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import ContinuousState, FloatND, UserParams
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_dcegm_activation import (
    RegimeId,
    _lower,
    activation_params,
    build_activation_model,
)
from tests.egm.test_dynamic_dcegm_consumers import (
    _capture_with_checks_off,
    _run_installed,
    _with_invalid_child_floor,
)
from tests.solution._nbegm_direct_oracle import OracleContext

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _empty_resources(*, liquid: ContinuousState) -> FloatND:
    return liquid - 0.5


def _public_empty_source() -> tuple[Model, UserParams]:
    """Keep the existing grids/laws and declare a genuine runtime parent floor."""
    base = build_activation_model(floor=0.5, literal=False, asset=False, parent=True)
    parent = base.user_regimes["parent"]
    solver = parent.solver
    assert isinstance(solver, _BoundDCEGM)
    margin = LiquidMargin(
        state=solver.continuous_state,
        action=solver.continuous_action,
        resources=solver.resources,
        post_decision_state=solver.post_decision_function,
    )
    parent = replace(
        parent,
        functions={**parent.functions, "resources": _empty_resources},
        constraints={
            "parent_borrowing": post_decision_lower_bound(margin=margin, lower=_lower)
        },
    )
    model = Model(
        regimes={**base.user_regimes, "parent": parent},
        ages=base.ages,
        edges={"parent": {"child": 0}, "child": {"done": 1}},
        initial_nodes={0: "parent"},
        regime_id_class=RegimeId,
    )
    # The public regime factory supplies this genuine bound solver.
    assert isinstance(model.user_regimes["parent"].solver, _BoundDCEGM)
    params = {
        **activation_params(floor=0.5, literal=False, parent=True),
        "parent": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "parent_borrowing_lower": {"floor": 0.2},
        },
    }
    return model, params


def test_actual_runtime_source_has_a_healthy_empty_control(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known negative resources are empty while genuine runtime status is clean."""
    model, params = _public_empty_source()
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert (
        cast(
            "_BoundDCEGM", model.user_regimes["parent"].solver
        )._negm_current_publication
        is False
    )
    output = _run_installed(kernel=kernel, context=context)
    assert isinstance(output, FeasibilityKernelOutput)
    np.testing.assert_array_equal(output.value, [-np.inf, -np.inf])
    np.testing.assert_array_equal(output.feasibility_counts[0], 2)
    assert len(output.bound_status) == 1
    np.testing.assert_array_equal(output.bound_status[0], [False, False])
    assert output.query_status is not None
    for status in output.query_status:
        np.testing.assert_array_equal(status, [False, False, False])


@pytest.mark.parametrize("own_failure", ["nonfinite", "collapse"])
@pytest.mark.parametrize("child_failure", ["nonfinite", "collapse"])
def test_actual_runtime_and_child_status_survive_empty_publication(
    *,
    own_failure: Literal["nonfinite", "collapse"],
    child_failure: Literal["nonfinite", "collapse"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safe arithmetic substitutions cannot erase either actual failing owner."""
    model, params = _public_empty_source()
    kernel, context = _capture_with_checks_off(
        model=model,
        params=params,
        regime_name="parent",
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )
    own = model._regimes["parent"].solution.query_authority
    assert own is not None
    assert own.bound.kind == "dag"
    context = _with_invalid_child_floor(
        model=model, context=context, failure=child_failure
    )
    actual = cast("FlatRegimeParams", context["flat_params"]["parent"])
    assert own.bound.lower_target is not None
    key = own.bound.lower_target + "__floor"
    assert key in actual
    dtype = jnp.asarray(actual[key]).dtype
    invalid = jnp.asarray(
        np.nan if own_failure == "nonfinite" else np.finfo(dtype).max / 2.0, dtype=dtype
    )
    changed_context: OracleContext = {
        **context,
        "flat_params": MappingProxyType(
            {
                **context["flat_params"],
                "parent": MappingProxyType({**actual, key: invalid}),
            }
        ),
    }
    output = _run_installed(kernel=kernel, context=changed_context)
    assert isinstance(output, FeasibilityKernelOutput)
    np.testing.assert_array_equal(output.value, [-np.inf, -np.inf])
    np.testing.assert_array_equal(output.feasibility_counts[0], 2)
    assert len(output.bound_status) == 1
    np.testing.assert_array_equal(
        output.bound_status[0], [own_failure == "nonfinite", True]
    )
    assert output.query_status is not None
    authorities = model._regimes["parent"].solution.query_input_authorities
    assert authorities is not None
    slots = authorities[0]
    assert slots[0][0] == "child"
    assert slots[0][1] == 1
    np.testing.assert_array_equal(
        jnp.stack(output.query_status[0])[:2], [child_failure == "nonfinite", True]
    )
    reason = (
        "nonfinite borrowing floor"
        if own_failure == "nonfinite"
        else "invalid shifted savings grid"
    )
    with pytest.raises(InvalidValueFunctionError, match=reason):
        validate_runtime_bound_status(
            statuses=output.bound_status,
            bound_specs=(own.bound,),
            regime_name="parent",
            period=0,
        )
    reason = (
        "nonfinite borrowing floor"
        if child_failure == "nonfinite"
        else "invalid shifted savings grid"
    )
    with pytest.raises(InvalidValueFunctionError, match=reason):
        validate_query_input_status(statuses=output.query_status, authorities=slots)
