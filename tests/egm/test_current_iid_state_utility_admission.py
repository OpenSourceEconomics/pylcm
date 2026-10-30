"""Only actual owned U envelopes can discharge direct-state utility requirements."""

from dataclasses import replace
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.current_iid_envelope import (
    DatedCaseUStepAuthority,
    DatedStateUtilityObligation,
)
from _lcm.solution.current_iid_case_u_native import NativeCaseUPeriodKernel
from _lcm.solution.current_iid_dated import _validate_state_utility_sources
from _lcm.solution.dcegm import _BoundDCEGM
from lcm import ConsumptionSavingsRegime, Model
from lcm.exceptions import ModelInitializationError, UnsupportedOperationError
from lcm.typing import ContinuousAction, ContinuousState
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_call_owned_generation import _declarations, _params
from tests.egm.test_current_iid_case_u_installation import _model_u, _utility_u
from tests.egm.test_current_iid_case_u_owned_core import _actual_declarations
from tests.egm.test_current_iid_parent_demands import _resources
from tests.simulation.test_case_u_reconstruction import _model as _shock_utility_model

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _state_utility(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> jax.Array:
    return jnp.log(consumption) + wealth


def test_public_dcegm_retains_its_original_state_utility_refusal() -> None:
    declarations = _actual_declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    with pytest.raises(ModelInitializationError, match="DCEGM envelope"):
        Model(
            regimes={
                **declarations.regimes,
                "child": replace(
                    child,
                    functions={**child.functions, "utility": _utility_u},
                ),
            },
            ages=declarations.ages,
            initial_nodes=declarations.initial_nodes,
            edges=declarations.edges,
            regime_id_class=declarations.regime_id_class,
            shocks=declarations.shocks,
        )


def test_owned_state_utility_without_u_refuses_before_euler_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declarations = _actual_declarations()
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    declarations = replace(
        declarations,
        regimes={
            **declarations.regimes,
            "child": replace(
                child,
                functions={
                    **child.functions,
                    "utility": _state_utility,
                    "resources": _resources,
                },
                shocks=(),
            ),
        },
        shocks={},
    )

    def forbidden(
        *args: object,  # noqa: PAN001 - refusal sentinel never reads operands
        **kwargs: object,  # noqa: PAN001 - refusal sentinel never reads operands
    ) -> None:
        del args, kwargs
        pytest.fail("An ordinary Euler producer ran with an unresolved U obligation.")

    monkeypatch.setattr(_BoundDCEGM, "prepare_period_steps", forbidden)
    with pytest.raises(UnsupportedOperationError, match=r"actual.*producer"):
        Model._from_call_owned_params(declarations=declarations, params=_params())


@pytest.fixture(scope="module")
def installed_u_model() -> Model:
    return _model_u()


@pytest.mark.parametrize(
    "defect",
    ["wrong-date", "missing-date", "no-plan", "no-source", "incomplete-roster"],
)
def test_owned_u_obligation_refuses_incomplete_original_date_authority(
    *, defect: str, installed_u_model: Model
) -> None:
    prepared = installed_u_model._dated_generation
    assert prepared is not None
    numerical = prepared.preparation.numerical
    obligations = dict(numerical.state_utility_obligations)
    original = obligations["child"]
    assert isinstance(original, DatedStateUtilityObligation)
    contexts = dict(numerical.contexts)
    declarations = dict(contexts["child"].projected_continuation_declarations)
    if defect == "wrong-date":
        obligations["child"] = replace(original, periods=(0,))
    elif defect == "missing-date":
        obligations["child"] = replace(original, periods=())
    elif defect == "no-plan":
        contexts["child"] = replace(
            contexts["child"], current_iid_plans=MappingProxyType({})
        )
    elif defect == "no-source":
        declarations.pop("child")
    else:
        obligations["child"] = replace(original, periods=(0, 1))
        contexts["child"] = replace(
            contexts["child"],
            regimes_to_active_periods=MappingProxyType(
                {**contexts["child"].regimes_to_active_periods, "child": (0, 1)}
            ),
        )
    with pytest.raises(UnsupportedOperationError, match="Direct-state utility"):
        _validate_state_utility_sources(
            obligations=obligations,
            contexts=contexts,
            solvers=numerical.solvers,
            declarations=declarations,
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_owned_u_runtime_support_rebuild_recreates_its_original_obligation() -> None:
    declarations = _declarations(runtime_child_grid=True)
    child = declarations.regimes["child"]
    assert isinstance(child, ConsumptionSavingsRegime)
    declarations = replace(
        declarations,
        regimes={
            **declarations.regimes,
            "child": replace(
                child,
                functions={
                    **child.functions,
                    "utility": _utility_u,
                    "resources": _resources,
                },
            ),
        },
    )
    first_params = _params(child_points=np.array([1.0, 12.5]))
    second_params = _params(child_points=np.array([1.25, 12.75]))
    model = Model._from_call_owned_params(
        declarations=declarations, params=first_params
    )
    _, first = model._prepare_call(params=first_params)
    _, second = model._prepare_call(params=second_params)
    assert first is not second
    obligations = []
    for generation in (first, second):
        dated = generation.execution._dated_generation
        assert dated is not None
        numerical = dated.preparation.numerical
        numerical.validate_state_utility()
        obligation = numerical.state_utility_obligations["child"]
        assert obligation.periods == (1,)
        owner = generation.solver_preparation(regime_name="child", period=1)
        owner.validate_owner()
        assert isinstance(owner.installed_kernel, NativeCaseUPeriodKernel)
        assert owner.installed_kernel.prepared.build is owner.build
        obligations.append(obligation)
    assert obligations[0] is not obligations[1]
    solution = model.solve(params=second_params, log_level="off", runtime_checks=True)
    liquid = jnp.asarray([1.25, 12.75])
    np.testing.assert_array_almost_equal(
        solution.values[1]["child"],
        2 * jnp.log(liquid / 2) + liquid / 2,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("direct_state", [False, True])
def test_native_u_scope_binds_only_original_dated_utility_inputs(
    *,
    direct_state: bool,
) -> None:
    """Actual independent utility shocks retain their original native U publisher."""
    model = _model_u() if direct_state else _shock_utility_model()
    dated = model._dated_generation
    assert dated is not None
    numerical = dated.preparation.numerical
    build = numerical.steps["child"]
    authority = build.native_case_u_sources[1]
    assert isinstance(authority, DatedCaseUStepAuthority)
    assert authority.plan is authority.context.current_iid_plans[1]
    authority.require(
        context=replace(
            authority.context,
            functions=MappingProxyType(dict(authority.context.functions)),
        ),
        solver=authority.solver,
        period=1,
    )
    assert authority.date is authority.context.projected_continuation_declarations[
        "child"
    ].date_at(period=1)
    assert authority.utility_inputs(
        regime_name="child",
        solver=authority.solver,
        functions=authority.context.functions,
        periods=(1,),
    ) == (frozenset({"eps", "wealth"}) if direct_state else frozenset({"eps"}))
    with pytest.raises(UnsupportedOperationError, match="ordinary"):
        authority.solver.finish_period_kernels(context=authority.context, build=build)
    # Absent native authority, the original ordinary kernel scope still refuses.
    with pytest.raises(ModelInitializationError, match="utility function depends"):
        authority.solver.prepare_period_steps(context=authority.context, periods=(1,))


@pytest.mark.parametrize("defect", ["date", "plan", "solver", "functions"])
def test_native_u_piece_authority_refuses_rebound_original_sources(
    *,
    defect: str,
    installed_u_model: Model,
) -> None:
    """A retained declaration cannot authorize another source's numerical pieces."""
    dated = installed_u_model._dated_generation
    assert dated is not None
    build = dated.preparation.numerical.steps["child"]
    original = build.native_case_u_sources[1]
    authority = original
    context = original.context
    solver = original.solver
    if defect == "date":
        authority = replace(original, period=0)
    elif defect == "plan":
        context = replace(context, current_iid_plans=MappingProxyType({}))
    elif defect == "solver":
        solver = replace(solver)
    else:
        context = replace(
            context,
            functions=MappingProxyType(
                {**context.functions, "utility": context.functions["resources"]}
            ),
        )
    with pytest.raises(UnsupportedOperationError, match="original dated source"):
        authority.require(context=context, solver=solver, period=1)
