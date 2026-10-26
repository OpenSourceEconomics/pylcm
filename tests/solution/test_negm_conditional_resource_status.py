"""Actual nested current resources retain mandatory status behind predicates."""

import logging
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution.feasibility import (
    FeasibilityKernelOutput,
    validate_query_input_status,
)
from _lcm.solution.negm import _NEGMPeriodKernel
from _lcm.solution.period_capture import computed_next_values
from _lcm.solution.period_replay import _load_capture_payload
from _lcm.typing import FlatRegimeParams
from lcm import Model
from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import ContinuousState, FloatND, ScalarFloat
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_germ_query_fixture import make_canonical_model
from tests.solution._nbegm_direct_oracle import OracleContext

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _gross_with_shift(
    *, liquid: ContinuousState, current_shift: ScalarFloat
) -> FloatND:
    return liquid + 1.25 + jnp.where(liquid == 0.0, current_shift, 0.0)


def _capture_empty_child(
    *, directory: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Model, _NEGMPeriodKernel, OracleContext]:
    model = make_canonical_model(false_predicate=True, child_gross=_gross_with_shift)
    with monkeypatch.context() as patch:
        patch.setenv("LCM_CAPTURE_PERIOD", "child@1")
        patch.setenv("LCM_CAPTURE_DIR", str(directory))
        model.solve(
            params={"discount_factor": 1.0, "current_shift": 0.0},
            log_level="off",
            runtime_checks=False,
        )
    inputs = _load_capture_payload(directory=directory / "child@1")["kernel_kwargs"]
    kernel = model._regimes["child"].solution.period_kernels[1]
    assert isinstance(kernel, _NEGMPeriodKernel)
    return (
        model,
        kernel,
        {
            "state_action_space": inputs["state_action_space"],
            "next_regime_to_V_arr": computed_next_values(
                next_regime_to_V_arr=inputs["next_regime_to_V_arr"]
            ),
            "next_regime_to_continuation": inputs["next_regime_to_continuation"],
            "flat_params": inputs["flat_params"],
            "period": inputs["period"],
            "ages": inputs["ages"],
        },
    )


def _run_child(
    *, kernel: _NEGMPeriodKernel, context: OracleContext
) -> FeasibilityKernelOutput:
    output = kernel(
        compiled_cores={
            name: partial(
                program.function,
                **{
                    axis.width_keyword: axis.extent
                    for axis in program.requirements.axes
                },
            )
            for name, program in kernel.core_programs().items()
        },
        **context,
        logger=logging.getLogger(__name__),
    )
    assert isinstance(output, FeasibilityKernelOutput)
    return output


@pytest.mark.parametrize("invalid_resource", [np.nan, np.inf, -np.inf])
def test_nested_nonfinite_resources_survive_false_current_predicate(
    *, invalid_resource: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty current choice cannot hide nonfinite actual resources."""
    model, kernel, context = _capture_empty_child(
        directory=tmp_path, monkeypatch=monkeypatch
    )
    healthy = _run_child(kernel=kernel, context=context)
    np.testing.assert_array_equal(healthy.value, np.full((4, 2), -np.inf))
    assert healthy.bound_status
    for status in healthy.bound_status:
        np.testing.assert_array_equal(status, [False, False])
    assert healthy.query_status is not None
    for status in healthy.query_status:
        np.testing.assert_array_equal(status, [False, False, False])

    actual = cast("FlatRegimeParams", context["flat_params"]["child"])
    shift_keys = tuple(key for key in actual if key.endswith("__current_shift"))
    assert len(shift_keys) == 1
    key = shift_keys[0]
    dtype = jnp.asarray(actual[key]).dtype
    liquid = jnp.asarray([0.0, 0.5, 1.0, 2.0], dtype=dtype)
    # The keeper's unchanged outer target has zero adjustment cost.
    resources = _gross_with_shift(
        liquid=liquid, current_shift=jnp.asarray(invalid_resource, dtype=dtype)
    )
    np.testing.assert_array_equal(jnp.isfinite(resources), [False, True, True, True])
    np.testing.assert_array_equal(resources[1:], [1.75, 2.25, 3.25])
    changed: OracleContext = {
        **context,
        "flat_params": MappingProxyType(
            {
                **context["flat_params"],
                "child": MappingProxyType(
                    {**actual, key: jnp.asarray(invalid_resource, dtype=dtype)}
                ),
            }
        ),
    }
    output = _run_child(kernel=kernel, context=changed)
    np.testing.assert_array_equal(output.value, np.full((4, 2), -np.inf))
    for status in output.bound_status:
        np.testing.assert_array_equal(status, [False, False])
    assert output.query_status is not None
    authorities = model._regimes["child"].solution.query_input_authorities
    assert authorities is not None
    slots = authorities[1]
    assert len(slots) == len(output.query_status) == 1
    assert slots[0][:2] == ("child", 1)
    np.testing.assert_array_equal(output.query_status[0], [False, False, True])
    with pytest.raises(InvalidValueFunctionError, match="query_coverage_missing"):
        validate_query_input_status(statuses=output.query_status, authorities=slots)
