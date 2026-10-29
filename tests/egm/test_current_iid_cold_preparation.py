"""Genuine original EGM builders under the same call-owned allocation boundary."""

import functools
from collections.abc import Callable

import jax
import numpy as np
import pytest

from _lcm.egm.current_iid_envelope import DatedCaseUStepAuthority
from _lcm.egm.step import _EGMStep
from _lcm.execution.core_program import CoreBuildContext
from _lcm.solution.current_iid_case_r import (
    CaseRArguments,
    CaseRProgramFunction,
    NativeCaseRPeriodKernel,
    case_r_native_inputs,
    trace_native_case_r,
)
from _lcm.solution.dcegm import _dcegm_output_roles, _dcegm_query_slots_and_capabilities
from _lcm.typing import PytreeValue, ShapeDtypePytree
from lcm import Model
from lcm.exceptions import ExecutionPlanningError
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_call_owned_generation import (
    _declarations,
)
from tests.egm.test_current_iid_call_owned_generation import (
    _params as _params_r,
)
from tests.egm.test_current_iid_case_u_installation import _model_u
from tests.egm.test_current_iid_case_u_owned_core import _params
from tests.egm.test_current_iid_generation_materialization import _owner

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def test_actual_dated_steps_admit_original_carry_without_changing_bits() -> None:
    """Use genuine installed context/solver/date, not a stand-alone fake schema."""
    model = _model_u()
    _, generation = model._prepare_call(params=_params())
    original = generation.solver_preparation(regime_name="child", period=1)
    assert original.plan is not None
    prepared = model._declaration_preparation
    assert prepared is not None
    # A U source's pieces exist only under its original source-date authority.
    authority = DatedCaseUStepAuthority(
        context=original.context,
        solver=original.solver,
        period=1,
        date=original.declarations["child"].date_at(period=1),
        plan=original.plan,
        state_utility_obligation=prepared.state_utility_obligations.get("child"),
    )
    expected = original.solver.prepare_period_steps(
        context=original.context, periods=(1,), case_u_source=authority
    )
    owner = _owner(roots=generation.roots)
    try:
        admitted = original.solver.prepare_period_steps(
            context=original.context,
            periods=(1,),
            materializer=owner,
            case_u_source=authority,
        )
        assert admitted.step_group_keys == expected.step_group_keys
        assert (
            admitted.stochastic_node_axes_by_period
            == expected.stochastic_node_axes_by_period
        )
        for actual, unchanged in zip(
            jax.tree.leaves(admitted.carry_template),
            jax.tree.leaves(expected.carry_template),
            strict=True,
        ):
            assert isinstance(actual, jax.Array)
            assert isinstance(unchanged, jax.Array)
            assert actual.dtype == unchanged.dtype
            assert jax.typeof(actual).weak_type == jax.typeof(unchanged).weak_type
            np.testing.assert_array_equal(actual, unchanged)
            assert any(
                actual is root for root in jax.tree.leaves(owner.solve_input_roots())
            )
        actual_step = admitted.steps[1]
        unchanged_step = expected.steps[1]
        assert isinstance(actual_step, _EGMStep)
        assert isinstance(unchanged_step, _EGMStep)
        assert (
            actual_step.pieces.continuation_plan.certainty_equivalent
            is unchanged_step.pieces.continuation_plan.certainty_equivalent
        )
    finally:
        owner.close()


def test_actual_dated_steps_refuse_before_retaining_bank_outputs() -> None:
    """The same genuine preparation must not allocate a bank past its budget."""
    model = _model_u()
    _, generation = model._prepare_call(params=_params())
    original = generation.solver_preparation(regime_name="child", period=1)
    owner = _owner(roots=generation.roots)
    owner.budget_bytes = 0
    try:
        with pytest.raises(ExecutionPlanningError):
            original.solver.prepare_period_steps(
                context=original.context, periods=(1,), materializer=owner
            )
        assert not owner._pending
    finally:
        owner.close()


def test_actual_native_r_argument_clocks_preserve_original_mapping() -> None:
    """Actual installed R source/parent mapping retains ordinary clock bits."""
    model = Model._from_call_owned_params(
        declarations=_declarations(), params=_params_r()
    )
    flat, generation = model._prepare_call(params=_params_r())
    original = generation.solver_preparation(regime_name="child", period=1)
    kernel = original.installed_kernel
    assert isinstance(kernel, NativeCaseRPeriodKernel)
    program = kernel.programs["main"]
    arguments = program.argument_builder
    function = program.function
    assert isinstance(arguments, CaseRArguments)
    assert isinstance(function, CaseRProgramFunction)
    child_templates = {}
    for target in arguments.base.stateful_targets:
        spec = generation.execution._regimes[target].solution.continuation_spec
        assert spec is not None
        child_templates[target] = spec.template_at(period=2)
    call = CoreBuildContext(
        state_action_space=original.context.state_action_space,
        next_regime_to_V_arr={},
        next_regime_to_continuation=child_templates,
        flat_params=flat,
        period=1,
        ages=original.context.ages,
    )
    unchanged = case_r_native_inputs(core=function.core, arguments=arguments(call))
    owner = _owner(roots=generation.roots)
    try:
        admitted = case_r_native_inputs(
            core=function.core,
            arguments=arguments.prepare(context=call, materializer=owner),
            materializer=owner,
        )
        assert jax.tree.structure(admitted) == jax.tree.structure(unchanged)
        for actual, expected in zip(
            jax.tree.leaves(admitted), jax.tree.leaves(unchanged), strict=True
        ):
            if isinstance(expected, jax.Array):
                assert isinstance(actual, jax.Array)
                assert actual.dtype == expected.dtype
                assert jax.typeof(actual).weak_type == jax.typeof(expected).weak_type
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
        original_widths = (
            admitted["savings_point_width"],
            admitted["stochastic_node_width"],
            admitted["envelope_cell_width"],
        )
        for savings_width, stochastic_width, cell_width in (original_widths, (2, 1, 2)):
            numeric = admitted.copy()
            numeric["savings_point_width"] = savings_width
            numeric["stochastic_node_width"] = stochastic_width
            numeric["envelope_cell_width"] = cell_width
            schema = trace_native_case_r(core=function.core, native_arguments=numeric)
            assert call.state_action_space is not None
            assert schema.value.shape == call.state_action_space.states["wealth"].shape
            frozen = kernel.spec.date_at(period=1).template
            for actual, expected in zip(
                jax.tree.leaves(schema.projected), jax.tree.leaves(frozen), strict=True
            ):
                assert actual.shape == expected.shape
                assert actual.dtype == expected.dtype
    finally:
        owner.close()


def test_actual_owned_output_roles_match_original_numeric_metadata() -> None:
    """Pure roles match a genuinely installed queried parent's numeric metadata."""
    model = Model._from_call_owned_params(
        declarations=_declarations(), params=_params_r()
    )
    _, generation = model._prepare_call(params=_params_r())
    owner = _owner(roots=generation.roots)
    try:
        parent = generation.solver_preparation(regime_name="parent", period=0)
        slots, _ = _dcegm_query_slots_and_capabilities(
            core=parent.step, regime_name="parent", period=0, own_bound_specs=()
        )
        assert slots is not None
        expected = _dcegm_output_roles(
            build=parent.build, publish_replay=False, query_slots=slots
        )
        actual = _dcegm_output_roles(
            build=parent.build,
            publish_replay=False,
            query_slots=slots,
            materializer=owner,
        )
        assert jax.tree.structure(actual) == jax.tree.structure(expected)
        assert jax.tree.leaves(actual) == jax.tree.leaves(expected)
        assert not owner._pending
    finally:
        owner.close()


def test_cold_case_r_build_traces_each_native_date_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demand closure takes the frozen schema; lowering checks the output roles."""
    from _lcm.egm.current_iid_native import NativeCaseRCore  # noqa: PLC0415

    traced: list[NativeCaseRCore] = []
    original = jax.eval_shape

    def counting(
        fun: Callable[..., PytreeValue], *args: PytreeValue, **kwargs: PytreeValue
    ) -> ShapeDtypePytree:
        if isinstance(fun, functools.partial) and isinstance(fun.func, NativeCaseRCore):
            traced.append(fun.func)
        return original(fun, *args, **kwargs)

    monkeypatch.setattr(jax, "eval_shape", counting)
    model = Model._from_call_owned_params(
        declarations=_declarations(), params=_params_r()
    )
    model._prepare_call(params=_params_r())
    # The child's single date installs its balanced schema by one shape trace.
    assert len(traced) == 1
