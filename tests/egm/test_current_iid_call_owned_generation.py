"""Held genuine Model-generation frontiers, not successful row substitutes."""

import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Never

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import lcm.model as model_module
from _lcm.continuation import ProjectedContinuationSpec
from _lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.solution.generation import canonical_parameter_schema
from _lcm.typing import FlatParams
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
)
from lcm.exceptions import ModelSealError, UnsupportedOperationError
from lcm.model import _ModelConstructorInputs
from lcm.persistence import load_solution
from lcm.shocks import Normal, ShockGroup, UserNodes
from lcm.solvers import DCEGM, ExactEnvelope
from lcm.typing import ContinuousState, FloatND, UserParams
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_current_iid_parent_demands import (
    _current_resources,
    _identity,
    _inverse,
    _landing,
    _RegimeId,
    _resources,
    _savings,
    _terminal,
    _utility,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _declarations(*, runtime_child_grid: bool = False) -> _ModelConstructorInputs:
    """Original age-reading economics with the analytically complete source domain.

    Parent saving at most four lands at 8.5. The child extends past that landing;
    log/log endpoint optima are resources/2, with income zero or one. Both
    initial and shifted runtime endpoints have those optima in the savings grid.
    """
    grid = IrregSpacedGrid(points=np.array([1.0, 4.0]))
    child_grid = (
        IrregSpacedGrid(n_points=2)
        if runtime_child_grid
        else IrregSpacedGrid(points=np.array([1.0, 12.5]))
    )
    savings = IrregSpacedGrid(
        points=np.array(
            [
                0.25,
                0.5,
                0.625,
                0.75,
                1.0,
                1.125,
                1.25,
                1.375,
                1.5,
                1.75,
                1.875,
                2.0,
                2.25,
                2.5,
                4.25,
                4.75,
                6.25,
                6.375,
                6.75,
                6.875,
                12.5,
                12.75,
            ]
        )
    )
    margin = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    functions = {
        "utility": _utility,
        "resources": _resources,
        "savings": _savings,
        "inverse_marginal_utility": _inverse,
    }
    parent = ConsumptionSavingsRegime(
        states={"wealth": grid},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.25, 2.0]))},
        functions=functions,
        state_transitions={"wealth": {"child": _landing}},
        liquid=margin,
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.5, 0.75, 1.0, 4.0])),
            envelope=ExactEnvelope(),
        ),
    )
    child = ConsumptionSavingsRegime(
        states={"wealth": child_grid},
        actions={"consumption": IrregSpacedGrid(points=np.array([0.25, 2.0]))},
        functions={**functions, "resources": _current_resources},
        state_transitions={"wealth": {"done": _identity}},
        liquid=margin,
        solver=DCEGM(savings_grid=savings, envelope=ExactEnvelope()),
        shocks=("income",),
    )
    return _ModelConstructorInputs(
        regimes={
            "parent": parent,
            "child": child,
            "done": Regime(
                states={"wealth": savings}, functions={"utility": _terminal}
            ),
        },
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "parent"}),
        edges={
            "parent": {"child": (0,)},
            "child": {"done": (1,)},
        },
        regime_id_class=_RegimeId,
        shocks={
            "income": ShockGroup(
                shocks={"eps": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.array([[0.0], [1.0]]),
                    weights=np.array([0.5, 0.5]),
                ),
            )
        },
    )


def _params(
    *,
    discount: float = 1.0,
    parent_slope: float = 2.0,
    child_points: np.ndarray | None = None,
) -> UserParams:
    child: UserParams = {"next_wealth": {"slope": 1.0}}
    if child_points is not None:
        child = {**child, "wealth": {"points": child_points}}
    return {
        "discount_factor": discount,
        "parent": {"next_wealth": {"slope": parent_slope}},
        "child": child,
    }


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_private_model_generation_uses_real_dates_and_dynamic_values() -> None:
    """Real constructor/programs; equal schemas do not freeze scientific values."""
    model = Model._from_call_owned_params(
        declarations=_declarations(), params=_params()
    )
    _, first = model._prepare_call(params=_params())
    _, changed = model._prepare_call(params=_params(discount=0.95, parent_slope=3.0))
    assert first is changed
    assert first.execution._regimes is model._regimes
    prepared = first.solver_preparation(regime_name="child", period=1)
    prepared.validate_owner()
    assert prepared.generation is first
    assert prepared.step is prepared.build.steps[1]
    assert prepared.plan is prepared.context.current_iid_plans[1]
    assert (
        prepared.installed_kernel is model._regimes["child"].solution.period_kernels[1]
    )
    child = model._regimes["child"].solution.continuation_spec
    assert isinstance(child, ProjectedContinuationSpec)
    date = child.date_at(period=1)
    assert date is child.dates_by_period[1]
    assert date.authority.period == 1
    assert model._regimes["parent"].solution.period_kernels[0] is not None
    # No authority-only replacement: the actual generation owns these programs.
    assert (
        first.execution._model_structure_fingerprint
        == model._model_structure_fingerprint
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_actual_runtime_support_rebuilds_real_program_generation() -> None:
    """Change genuine grid operands, retaining the prior model and result owner."""
    declarations = _declarations(runtime_child_grid=True)
    first_params = _params(child_points=np.array([1.0, 12.5]))
    second_params = _params(child_points=np.array([1.25, 12.75]))
    model = Model._from_call_owned_params(
        declarations=declarations, params=first_params
    )
    _, first = model._prepare_call(params=first_params)
    model._generation_cache.capacity = 1
    _, second = model._prepare_call(params=second_params)
    assert first is not second
    assert first.execution._regimes is not second.execution._regimes
    for generation, endpoint in ((first, 1.0), (second, 1.25)):
        spec = generation.execution._regimes["child"].solution.continuation_spec
        assert isinstance(spec, ProjectedContinuationSpec)
        assert float(spec.date_at(period=1).template.domain_lower) == endpoint
    # Eviction cannot detach a still-held producing generation.
    _, reacquired = model._prepare_call(params=first_params)
    assert reacquired is first
    assert model._regimes is first.execution._regimes


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_canonical_group_mapping_leaf_schema_is_not_an_object_scalar() -> None:
    """Raw namespaces lead to actual nested MappingLeaf dtype/shape signatures."""

    def parameters(*, size: int, value: float) -> FlatParams:
        return MappingProxyType(
            {
                "income": MappingProxyType(
                    {"law": MappingLeaf({"array": jnp.full((size,), value)})}
                ),
                "edges": MappingProxyType(
                    {"parent": MappingProxyType({"law__scale": jnp.asarray(1.0)})}
                ),
            }
        )

    assert canonical_parameter_schema(
        flat_params=parameters(size=2, value=1.0)
    ) == canonical_parameter_schema(flat_params=parameters(size=2, value=9.0))
    assert canonical_parameter_schema(
        flat_params=parameters(size=2, value=1.0)
    ) != canonical_parameter_schema(flat_params=parameters(size=3, value=1.0))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_private_model_solve_owned_forward_and_restore_use_same_generation(
    *,
    tmp_path: Path,
) -> None:
    """The actual public solve/forward bodies consume the private real Model."""
    params = _params()
    model = Model._from_call_owned_params(declarations=_declarations(), params=params)
    result = model.solve(params=params, log_level="warning")
    view = result._engine_view
    assert isinstance(view, OwnedSolutionView)
    assert view.generation is not None
    assert view.generation.execution._regimes is model._regimes
    frame = model.simulate(
        params=params,
        solution=result,
        seed=19,
        log_level="warning",
        initial_conditions={
            "regime_id": np.array([0]),
            "age": np.array([0]),
            "wealth": np.array([4.0]),
        },
    )
    assert not frame.to_dataframe().empty
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    state = restored.__getstate__()
    assert "_generation_cache" not in state
    assert "_declared_authority_lock" not in state
    assert state["_initial_generation_key"] == model._initial_generation_key
    # Persisted replay must cross the actual descriptor/native preflight.
    path = tmp_path / "call-owned-solution.lcm"
    result.save(path=path)
    foreign = load_solution(path=path)
    assert foreign._engine_view is None
    restored.simulate(
        params=params,
        solution=foreign,
        seed=19,
        log_level="warning",
        initial_conditions={
            "regime_id": np.array([0]),
            "age": np.array([0]),
            "wealth": np.array([4.0]),
        },
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_cold_budgeted_dated_specialization_refuses_before_native_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported eager lane refuses before either original dated factory."""
    declarations = replace(
        _declarations(runtime_child_grid=True),
        enable_jit=False,
        execution_config=ExecutionConfig(device_memory_bytes=1 << 29),
    )
    calls: list[bool] = []

    def observe(
        *args: object,  # noqa: PAN001 - refusal sentinel never reads operands
        **kwargs: object,  # noqa: PAN001 - refusal sentinel never reads operands
    ) -> Never:
        del args, kwargs
        calls.append(True)
        raise AssertionError("A refused generation reached its dated metadata factory.")

    monkeypatch.setattr(model_module, "prepare_dated_parameters", observe)
    monkeypatch.setattr(model_module, "build_private_dated_balanced_regimes", observe)
    with pytest.raises(
        UnsupportedOperationError,
        match="unsharded compiled scalar producers",
    ):
        Model._from_call_owned_params(
            declarations=declarations,
            params=_params(child_points=np.array([1.0, 12.5])),
        )
    assert calls == []


@pytest.mark.parametrize("nested", [False, True])
def test_retained_constructor_proxy_rebinding_refuses_before_preparation(
    *, nested: bool
) -> None:
    """A finalized runtime seal cannot authorize mutated raw constructor aliases."""
    original = _declarations()
    raw_regimes = dict(original.regimes)
    raw_functions = dict(raw_regimes["child"].functions)
    if nested:
        raw_regimes["child"] = replace(
            raw_regimes["child"], functions=MappingProxyType(raw_functions)
        )
    declarations = replace(original, regimes=MappingProxyType(raw_regimes))
    if nested:
        raw_functions["resources"] = _resources
    else:
        raw_regimes["child"] = raw_regimes["parent"]
    # The failure must precede the first declaration read, hence precedes native
    # metadata, support, template or compiler construction in a cold factory.
    with pytest.raises(RuntimeError, match="Retained Model constructor"):
        declarations.prepare(model=Model.__new__(Model))


_CUSTODY_RESOURCE_OFFSET = 0.0


def _resources_with_custody_global(*, wealth: ContinuousState, eps: FloatND) -> FloatND:
    return wealth + eps + _CUSTODY_RESOURCE_OFFSET


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_original_execution_seal_precedes_private_generation_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed callback global cannot reach normalization/support/cold rebuild."""
    original = _declarations(runtime_child_grid=True)
    regimes = dict(original.regimes)
    child = regimes["child"]
    regimes["child"] = replace(
        child,
        functions={**child.functions, "resources": _resources_with_custody_global},
    )
    params = _params(child_points=np.array([1.0, 12.5]))
    captured = jnp.asarray(0.0)
    monkeypatch.setattr(sys.modules[__name__], "_CUSTODY_RESOURCE_OFFSET", captured)
    model = Model._from_call_owned_params(
        declarations=replace(original, regimes=regimes), params=params
    )
    # The actual referenced-global array is retained by the same generation,
    # without walking unrelated module globals or reconstructing a copy.
    assert any(root is captured for root in model._generation_roots())
    assert isinstance(captured, jax.Array)
    flat, _ = model._prepare_call(params=params)
    monkeypatch.setattr(sys.modules[__name__], "_CUSTODY_RESOURCE_OFFSET", 0.125)
    with pytest.raises(ModelSealError, match="_CUSTODY_RESOURCE_OFFSET"):
        model._prepare_call(params=_params(child_points=np.array([1.25, 12.75])))
    with pytest.raises(ModelSealError, match="_CUSTODY_RESOURCE_OFFSET"):
        model._acquire_generation(flat_params=flat)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_raw_fixed_numpy_mutation_refuses_before_cold_declarations() -> None:
    """The normalized JAX fixed operand is not custody of its retained raw source."""
    fixed_discount = np.asarray(1.0)
    declarations = replace(
        _declarations(runtime_child_grid=True),
        fixed_params={"discount_factor": fixed_discount},
    )
    params = dict(_params(child_points=np.array([1.0, 12.5])))
    del params["discount_factor"]
    model = Model._from_call_owned_params(declarations=declarations, params=params)
    fixed_discount[...] = 0.95
    next_params = dict(_params(child_points=np.array([1.25, 12.75])))
    del next_params["discount_factor"]
    with pytest.raises(RuntimeError, match="constructor numeric contents"):
        model._prepare_call(params=next_params)


@pytest.mark.parametrize("grouped", [False, True])
def test_private_dated_retained_series_refuses_before_declaration_preparation(
    *, grouped: bool
) -> None:
    """A Series needs its full labels/clock contract, not a value-array checksum."""
    series = pd.Series([1.0, 1.0, 1.0], index=pd.Index([0, 1, 2], name="age"))
    leaf = UserMappingLeaf({"schedule": series}) if grouped else series
    declarations = replace(_declarations(), fixed_params={"discount_factor": leaf})
    with pytest.raises(UnsupportedOperationError, match="retained pandas Series"):
        Model._from_call_owned_params(declarations=declarations, params=_params())
