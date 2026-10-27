"""Original host checks remain installed outside the genuine numerical core."""

from dataclasses import fields

import pytest

from _lcm.egm.upper_envelope.query import ComparisonArithmetic
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.solution import nbegm
from _lcm.solution.current_iid_nbegm import NativeNBCaseUPeriodKernel
from _lcm.solution.nbegm import (
    _DeferredProbe,
    _fail_if_budget_nonaffine_in_liquid,
    _NBEGMContinuousCore,
    _NBEGMFeasibilityConstraint,
    _NBEGMScheduleSpec,
)
from _lcm.typing import EconFunctionsMapping
from lcm.typing import ActionName, Float1D
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_nbegm_model import _model

pytestmark = [
    pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON),
    pytest.mark.coverage(backends=("cpu",), precisions="both"),
]


@pytest.mark.parametrize("device_memory_bytes", [None, 1 << 28])
def test_original_builder_keeps_numeric_field_and_host_check_identities(
    *, device_memory_bytes: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = nbegm._build_nbegm_continuous_core
    calls: list[tuple[_NBEGMScheduleSpec, _NBEGMContinuousCore]] = []

    def original_build(
        *,
        savings_grid: Float1D,
        functions: EconFunctionsMapping,
        consumption_action: ActionName,
        schedule_spec: _NBEGMScheduleSpec,
        feasibility_constraints: tuple[_NBEGMFeasibilityConstraint, ...] = (),
        envelope_arithmetic: ComparisonArithmetic = "certified",
        materializer: DatedGenerationMaterializer | None = None,
    ) -> _NBEGMContinuousCore:
        core = original(
            savings_grid=savings_grid,
            functions=functions,
            consumption_action=consumption_action,
            schedule_spec=schedule_spec,
            feasibility_constraints=feasibility_constraints,
            envelope_arithmetic=envelope_arithmetic,
            materializer=materializer,
        )
        assert core.schedule_spec.param_checks == ()
        for declaration in fields(schedule_spec):
            if declaration.name != "param_checks":
                assert getattr(core.schedule_spec, declaration.name) is getattr(
                    schedule_spec, declaration.name
                )
        calls.append((schedule_spec, core))
        return core

    monkeypatch.setattr(nbegm, "_build_nbegm_continuous_core", original_build)
    model = _model(device_memory_bytes=device_memory_bytes)
    solution = model._regimes["child"].solution
    kernel = solution.period_kernels[1]
    assert isinstance(kernel, NativeNBCaseUPeriodKernel)
    assert solution.param_checks is kernel.param_checks
    assert kernel.param_checks
    assert all(
        isinstance(check, _DeferredProbe)
        and check.probe is _fail_if_budget_nonaffine_in_liquid
        for check in kernel.param_checks
    )
    matches = [
        (spec, core)
        for spec, core in calls
        if kernel.point.core.schedule_spec is core.schedule_spec
    ]
    assert len(matches) == 1
    spec, core = matches[0]
    assert spec.param_checks is kernel.param_checks
    assert spec.param_checks != core.schedule_spec.param_checks
