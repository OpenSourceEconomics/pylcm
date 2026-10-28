"""Held genuine Model-generation frontiers, not successful row substitutes."""

import inspect
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import FunctionType, MappingProxyType
from typing import Never, cast

import cloudpickle
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import lcm.model as model_module
from _lcm.continuation import ProjectedContinuationSpec
from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.current_iid_balanced import (
    BalancedProjectedEGMCarry,
    BalancedResourcesProgram,
)
from _lcm.egm.current_iid_projection import _read_projected
from _lcm.egm.projected_continuation import PROJECTED_EGM_CONTINUATION
from _lcm.params.mapping_leaf import MappingLeaf, UserMappingLeaf
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.solution.current_iid_case_r import CASE_R_RESOURCES_REPLAY
from _lcm.solution.generation import canonical_parameter_schema
from _lcm.solution.model_authority import _is_installed_balanced_r_continuation
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
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidParamsError,
    InvalidValueFunctionError,
    ModelSealError,
    ShockDeclarationError,
    SolutionIntegrityError,
    UnsupportedOperationError,
)
from lcm.model import _ModelConstructorInputs
from lcm.persistence import load_solution
from lcm.shocks import Normal, ShockGroup, UserNodes
from lcm.solver_api import ArtifactRef, PersistencePolicy, ResultRetention
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
from tests.solution.test_solution_persistence import (
    _MutableJSONValue,
    _read_manifest,
    _replace_manifest,
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
@pytest.mark.parametrize(
    ("nodes", "weights", "match"),
    [
        ([[0.0], [1.0]], [1.5, -0.5], "Negative concrete weights"),
        ([[0.0], [1.0]], [0.5, 0.25], "Weights must sum to unity"),
        ([[0.0], [np.nan]], [0.5, 0.5], "finite nonempty"),
        ([[0.0], [np.inf]], [0.5, 0.5], "finite nonempty"),
    ],
)
def test_current_iid_law_refuses_malformed_nodes_before_any_model(
    *, nodes: list[list[float]], weights: list[float], match: str
) -> None:
    """A malformed current-IID law is refused at its declaration."""
    with pytest.raises(ShockDeclarationError, match=match):
        UserNodes(nodes=np.array(nodes), weights=np.array(weights))


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
def test_public_model_solve_owned_forward_and_restore_use_same_generation(
    *,
    tmp_path: Path,
) -> None:
    """A warmed public facade restores genuine call-owned execution and replay."""
    params = _params()
    model = _public_model_from_declarations(declarations=_declarations())
    result = model.solve(params=params, log_level="warning")
    view = result._engine_view
    assert isinstance(view, OwnedSolutionView)
    assert view.generation is not None
    acquired = view.generation
    assert acquired.execution is not model
    assert model._generation_cache._cached[acquired.key] is acquired
    assert model._generation_cache.roots()
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
    assert isinstance(restored, Model)
    assert restored._is_prepared_dated
    assert restored._generation_cache.roots() == ()
    state = restored.__getstate__()
    assert "_generation_cache" not in state
    assert "_declared_authority_lock" not in state
    assert "_regimes" not in state
    # Persisted replay must cross the actual descriptor/native preflight.
    path = tmp_path / "call-owned-solution.lcm"
    result.save(path=path)
    foreign = load_solution(path=path)
    assert foreign._engine_view is None
    replay = restored.simulate(
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
    restored_generation = restored._generation_cache._cached[acquired.key]
    assert restored_generation.key == acquired.key
    assert restored_generation is not acquired
    assert restored_generation.execution is not restored
    state = restored.__getstate__()
    assert state["_initial_generation_key"] == model._initial_generation_key
    pd.testing.assert_frame_equal(
        replay.to_dataframe(), frame.to_dataframe(), check_exact=True
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


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_model_current_iid_balanced_solve_replay_and_archive(  # noqa: PLR0915 - complete public lifecycle and associated physical oracle
    *, tmp_path: Path
) -> None:
    """Public construction retains the same original economics through fresh replay."""
    declarations = _declarations()
    constructor_inputs = {
        field.name: getattr(declarations, field.name)
        for field in fields(declarations)
        if field.init
    }
    params = _params()
    model = Model(**constructor_inputs)
    template = model.get_params_template()
    child_template = template["parent"]["child"]
    assert isinstance(child_template, dict)
    assert "slope" in child_template["next_wealth"]
    assert "discount_factor" in template["parent"]["koopmans_aggregator"]
    result = model.solve(params=params, log_level="warning", max_compilation_workers=1)
    view = result._engine_view
    assert isinstance(view, OwnedSolutionView)
    generation = view.generation
    assert generation is not None
    prepared = generation.solver_preparation(regime_name="child", period=1)
    prepared.validate_owner()
    assert prepared.generation is generation
    assert prepared.step is prepared.build.steps[1]
    assert prepared.plan is prepared.context.current_iid_plans[1]
    assert (
        prepared.installed_kernel
        is generation.execution._regimes["child"].solution.period_kernels[1]
    )
    spec = generation.execution._regimes["child"].solution.continuation_spec
    assert isinstance(spec, ProjectedContinuationSpec)
    assert spec.date_at(period=1) is spec.dates_by_period[1]
    assert spec.date_at(period=1).authority.period == 1
    date = spec.date_at(period=1)
    assert type(date.template) is BalancedProjectedEGMCarry
    assert type(date.guard_program) is BalancedResourcesProgram
    child_regime = generation.execution._regimes["child"]
    assert _is_installed_balanced_r_continuation(
        regime=child_regime,
        regime_name="child",
        period=1,
        spec=spec,
        template=date.template,
    )
    assert not _is_installed_balanced_r_continuation(
        regime=child_regime,
        regime_name="child",
        period=1,
        spec=replace(spec),
        template=date.template,
    )
    reader = generation.execution._regimes[
        "parent"
    ].simulation.projected_forward_readers_by_period[0]["child"]
    assert reader.date is date
    retained = result.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(retained, BalancedProjectedEGMCarry)
    operands = reader.operands(carry=retained)
    assert tuple(operands) == reader.native_arguments
    for leaf_path, name in reader.argument_by_leaf.items():
        assert operands[name] is retained.leaves()[leaf_path]
    with pytest.raises(TypeError, match="target differs"):
        replace(reader, target="done").operands(carry=retained)
    child_value = np.asarray(view.values[1]["child"])
    wealth = np.asarray([1.0, 12.5], dtype=child_value.dtype)
    # Each quadrature outcome's original log/log optimum lies on the saving grid.
    expected = np.log(wealth / 2.0) + np.log((wealth + 1.0) / 2.0)
    tolerance = 64 * np.finfo(child_value.dtype).eps
    np.testing.assert_allclose(child_value, expected, rtol=tolerance, atol=tolerance)
    assert np.isfinite(np.asarray(view.values[0]["parent"])).all()
    ref = ArtifactRef(period=1, regime="child", key=CASE_R_RESOURCES_REPLAY)
    projected_ref = ArtifactRef(
        period=1, regime="child", key=PROJECTED_EGM_CONTINUATION
    )
    assert (
        result.metadata.artifact_descriptors[projected_ref].persistence
        is PersistencePolicy.MODEL_VERIFIABLE
    )
    assert (
        result.metadata.artifact_descriptors[ref].persistence
        is PersistencePolicy.MODEL_VERIFIABLE
    )
    assert result.metadata.artifact_descriptors[projected_ref].required_for
    assert result.metadata.artifact_descriptors[ref].required_for
    payload = result.replay_artifacts[ref]
    assert type(payload) is tuple
    assert payload
    assert all(isinstance(leaf, jax.Array) for leaf in payload)
    initial_conditions = {
        "regime_id": np.array([0]),
        "age": np.array([0]),
        "wealth": np.array([4.0]),
    }
    frame = model.simulate(
        params=params,
        solution=result,
        seed=19,
        log_level="warning",
        initial_conditions=initial_conditions,
    )
    # The original declared grid choices have a strict independent log/log
    # ranking. This checks their associated physical action and landing, not
    # equality of the projected PWA value to an analytic value interpolant.
    parent_actions = np.asarray([0.25, 2.0])
    child_landings = 2.0 * (4.0 - parent_actions) + 0.5
    grid_scores = (
        np.log(parent_actions)
        + np.log(child_landings / 2.0)
        + np.log((child_landings + 1.0) / 2.0)
    )
    winner = int(np.argmax(grid_scores))
    assert grid_scores[winner] > grid_scores[1 - winner]
    np.testing.assert_array_equal(
        frame.raw_results["parent"][0].actions["consumption"],
        parent_actions[winner : winner + 1],
    )
    np.testing.assert_array_equal(
        frame.raw_results["child"][1].states["wealth"],
        child_landings[winner : winner + 1],
    )
    expected_frame = frame.to_dataframe()
    assert not expected_frame.empty
    path = tmp_path / "public-balanced-current-iid.lcm"
    result.save(path=path)
    loaded = load_solution(path=path)
    assert loaded._engine_view is None
    fresh = Model(**constructor_inputs)
    replay = fresh.simulate(
        params=params,
        solution=loaded,
        seed=19,
        log_level="warning",
        initial_conditions=initial_conditions,
    )
    pd.testing.assert_frame_equal(
        replay.to_dataframe(), expected_frame, check_exact=True
    )


def _public_model_from_declarations(*, declarations: _ModelConstructorInputs) -> Model:
    """Pass actual constructor fields through the ordinary public entry point."""
    return Model(
        **{
            field.name: getattr(declarations, field.name)
            for field in fields(declarations)
            if field.init
        }
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_row_matches_the_literal_economics_off_the_child_grid() -> None:
    """Integrated W and economic M at off-grid wealth match the closed form.

    At w in {1.5, 2.5, 3.5} both conditional log/log optima (w/2 and (w+1)/2)
    lie on the declared saving grid, so the grid-conditional solution is
    W = log(w/2) + log((w+1)/2) and M = 1/w + 1/(w+1), independent of the
    production selection. The child grid is {1, 12.5}.
    """
    result = _public_model_from_declarations(declarations=_declarations()).solve(
        params=_params(), log_level="warning", max_compilation_workers=1
    )
    carry = result.retained_continuations[
        ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
    ]
    assert isinstance(carry, BalancedProjectedEGMCarry)
    dtype = carry.grid.dtype
    eps = float(jnp.finfo(dtype).eps)
    for wealth in (1.5, 2.5, 3.5):
        read = _read_projected(carry=carry, query=jnp.asarray(wealth, dtype=dtype))
        assert bool(read.valid)
        value = np.log(wealth / 2.0) + np.log((wealth + 1.0) / 2.0)
        marginal = 1.0 / wealth + 1.0 / (wealth + 1.0)
        assert abs(float(read.value) - value) / max(1.0, abs(value)) <= max(
            1e-6, 64 * eps
        )
        assert abs(float(read.economic_marginal) - marginal) / max(
            1.0, marginal
        ) <= max(1e-4, 8 * np.sqrt(eps))


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_replay_matches_the_literal_policy_at_off_node_shocks() -> None:
    """Physical replay at drawn, off-node shocks follows the closed-form policy.

    Every Euler point of this log/log child lies on c = R/2, so its exact
    envelope is c = (w + eps)/2 for any realized resources whose half lies on
    the saving-grid span. Drawn shocks are not quadrature nodes {0, 1}; the
    next state equals the saving R - c because the law's slope is one.
    """
    model = _public_model_from_declarations(declarations=_declarations())
    params = _params()
    result = model.solve(params=params, log_level="warning", max_compilation_workers=1)
    n_subjects = 8
    frame = model.simulate(
        params=params,
        solution=result,
        seed=19,
        log_level="warning",
        initial_conditions={
            "regime_id": np.zeros(n_subjects, dtype=int),
            "age": np.zeros(n_subjects, dtype=int),
            "wealth": np.linspace(2.5, 4.0, n_subjects, dtype=canonical_float_dtype()),
        },
    )
    child = frame.raw_results["child"][1]
    done = frame.raw_results["done"][2]
    wealth = np.asarray(child.states["wealth"])
    shock = np.asarray(child.shocks["eps"])
    consumption = np.asarray(child.actions["consumption"])
    assert not np.isin(shock, [0.0, 1.0]).any()
    resources = wealth + shock
    assert (resources / 2.0 >= 0.25).all()
    assert (resources / 2.0 <= 12.75).all()
    tolerance = max(1e-4, 8 * np.sqrt(np.finfo(consumption.dtype).eps))
    np.testing.assert_allclose(consumption, resources / 2.0, rtol=0, atol=tolerance)
    np.testing.assert_allclose(
        np.asarray(done.states["wealth"]),
        resources - consumption,
        rtol=0,
        atol=tolerance,
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_refuses_an_exhausted_projection_capacity() -> None:
    """A row that cannot meet its budgets within capacity refuses, never truncates."""
    declarations = _declarations()
    child = declarations.regimes["child"]
    assert isinstance(child.solver, DCEGM)
    child = replace(
        child, solver=replace(child.solver, current_iid_projection_capacity=16)
    )
    model = _public_model_from_declarations(
        declarations=replace(
            declarations, regimes={**declarations.regimes, "child": child}
        )
    )
    with pytest.raises(
        InvalidValueFunctionError,
        match=(
            r"Invalid projected continuation .* regime 'child', period 1.*"
            r"projection capacity of \d+ records.*current_iid_projection_capacity"
        ),
    ):
        model.solve(params=_params(), log_level="warning", max_compilation_workers=1)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    ("defect", "match"),
    [
        ("corrupted", "leaf 0 checksum does not match"),
        ("dropped", "do not exactly cover its artifact descriptors"),
    ],
)
def test_public_case_r_archive_refuses_a_damaged_replay_payload(
    *, defect: str, match: str, tmp_path: Path
) -> None:
    """A fresh Model never replays from a damaged or missing Case-R row."""
    declarations = _declarations()
    result = _public_model_from_declarations(declarations=declarations).solve(
        params=_params(), log_level="warning", max_compilation_workers=1
    )
    path = tmp_path / "public-case-r.lcm"
    result.save(path=path)
    with h5py.File(path, "r+") as archive:
        manifest = _read_manifest(archive)
        artifacts = cast("list[dict[str, _MutableJSONValue]]", manifest["artifacts"])
        (entry,) = (
            entry
            for entry in artifacts
            if cast(
                "dict[str, _MutableJSONValue]",
                cast("list[dict[str, _MutableJSONValue]]", entry["leaves"])[0][
                    "identity"
                ],
            )["type_id"]
            == CASE_R_RESOURCES_REPLAY.type_id
        )
        if defect == "corrupted":
            leaves = cast("list[dict[str, str]]", entry["leaves"])
            dataset = archive[leaves[0]["dataset"]]
            leaf = dataset[()]
            leaf[0] += 1.0
            dataset[...] = leaf
        else:
            artifacts.remove(entry)
            _replace_manifest(archive=archive, manifest=manifest)
    fresh = _public_model_from_declarations(declarations=declarations)
    with pytest.raises(SolutionIntegrityError, match=match):
        fresh.simulate(
            params=_params(),
            solution=load_solution(path=path),
            seed=19,
            log_level="warning",
            initial_conditions={
                "regime_id": np.array([0]),
                "age": np.array([0]),
                "wealth": np.array([4.0]),
            },
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_prepared_generations_reuse_schema_and_retain_real_support() -> None:
    """Call parameters choose real support while old held owners survive eviction."""
    model = _public_model_from_declarations(
        declarations=_declarations(runtime_child_grid=True)
    )
    assert model.state_names(regime_name="child") == ("wealth",)
    assert model.initial_state_names(regime_name="child") == ("wealth",)
    first_params = _params(child_points=np.array([1.0, 12.5]))
    _, first = model._prepare_call(params=first_params)
    _, same = model._prepare_call(
        params=_params(
            discount=0.95, parent_slope=3.0, child_points=np.array([1.0, 12.5])
        )
    )
    assert same is first
    first.solver_preparation(regime_name="child", period=1).validate_owner()
    model._generation_cache.capacity = 1
    _, second = model._prepare_call(
        params=_params(child_points=np.array([1.25, 12.75]))
    )
    assert second is not first
    assert second.execution._regimes is not first.execution._regimes
    for generation, endpoint in ((first, 1.0), (second, 1.25)):
        spec = generation.execution._regimes["child"].solution.continuation_spec
        assert isinstance(spec, ProjectedContinuationSpec)
        assert float(spec.date_at(period=1).template.domain_lower) == endpoint
        generation.solver_preparation(regime_name="child", period=1).validate_owner()
    _, reacquired = model._prepare_call(params=first_params)
    assert reacquired is first


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("budgeted", [False, True])
def test_public_prepared_pickle_rebuilds_actual_declarations(*, budgeted: bool) -> None:
    """A restored prepared Model derives its own programs from actual parameters."""
    declarations = _declarations()
    if budgeted:
        declarations = replace(
            declarations,
            execution_config=ExecutionConfig(device_memory_bytes=1 << 28),
        )
    model = _public_model_from_declarations(declarations=declarations)
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert isinstance(restored, Model)
    assert restored.get_params_template() == model.get_params_template()
    _, generation = restored._prepare_call(params=_params())
    actual = generation.solver_preparation(regime_name="child", period=1)
    actual.validate_owner()
    assert actual.context.regime_name == "child"
    assert actual.plan is actual.context.current_iid_plans[1]
    with pytest.raises(UnsupportedOperationError, match="state grid"):
        restored.state_grid(
            params=_params(), regime_name="child", state_name="wealth", period=1
        )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_prepared_pickle_keeps_saved_semantics_before_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regenerated identity witnesses cannot bless changed restored declarations."""
    model = _public_model_from_declarations(declarations=_declarations())
    state = model.__getstate__()
    declarations = cloudpickle.loads(cloudpickle.dumps(state["_constructor_inputs"]))
    assert isinstance(declarations, _ModelConstructorInputs)
    saved_seal = state["_prepared_declaration_seal"]
    state["_constructor_inputs"] = replace(
        declarations, fixed_params={"discount_factor": 0.95}
    )
    calls: list[bool] = []

    def observe(
        *args: object,  # noqa: PAN001 - refusal sentinel never reads operands
        **kwargs: object,  # noqa: PAN001 - refusal sentinel never reads operands
    ) -> Never:
        del args, kwargs
        calls.append(True)
        raise AssertionError(
            "Changed restored semantics reached a declaration factory."
        )

    monkeypatch.setattr(_ModelConstructorInputs, "prepare", observe)
    restored = Model.__new__(Model)
    with pytest.raises(ModelSealError, match="archive declaration semantics changed"):
        restored.__setstate__(state)
    assert state["_prepared_declaration_seal"] == saved_seal
    assert calls == []


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("mutation", ["global", "code", "defaults"])
def test_public_prepared_callback_mutation_refuses_before_support(
    *, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw callback code and globals cannot change before first-call preparation."""
    original = _declarations()
    regimes = dict(original.regimes)
    regimes["child"] = replace(
        regimes["child"],
        functions={
            **regimes["child"].functions,
            "resources": _resources_with_custody_global,
        },
    )
    monkeypatch.setattr(sys.modules[__name__], "_CUSTODY_RESOURCE_OFFSET", 0.0)
    model = _public_model_from_declarations(
        declarations=replace(original, regimes=regimes)
    )
    calls: list[bool] = []

    def observe(
        *args: object,  # noqa: PAN001 - refusal sentinel never reads operands
        **kwargs: object,  # noqa: PAN001 - refusal sentinel never reads operands
    ) -> Never:
        del args, kwargs
        calls.append(True)
        raise AssertionError("A mutated declaration reached its support factory.")

    monkeypatch.setattr(model_module, "prepare_dated_call_support", observe)
    if mutation == "global":
        monkeypatch.setattr(sys.modules[__name__], "_CUSTODY_RESOURCE_OFFSET", 0.125)
    else:

        def different_resources(*, wealth: ContinuousState, eps: FloatND) -> FloatND:
            return wealth + eps + 0.25

        raw = inspect.unwrap(_resources_with_custody_global)
        assert isinstance(raw, FunctionType)
        if mutation == "defaults":
            monkeypatch.setattr(raw, "__kwdefaults__", {"eps": 0.0})
        else:
            replacement = inspect.unwrap(different_resources)
            assert isinstance(replacement, FunctionType)
            assert raw.__code__.co_freevars == replacement.__code__.co_freevars
            monkeypatch.setattr(raw, "__code__", replacement.__code__)
    with pytest.raises(ModelSealError):
        model._prepare_call(params=_params())
    assert calls == []


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_prepared_numeric_mutation_refuses_before_support() -> None:
    """The normalized fixed operand does not replace custody of its raw source."""
    discount = np.asarray(1.0)
    model = _public_model_from_declarations(
        declarations=replace(
            _declarations(), fixed_params={"discount_factor": discount}
        )
    )
    params = dict(_params())
    del params["discount_factor"]
    discount[...] = 0.95
    with pytest.raises(RuntimeError, match="constructor numeric contents"):
        model._prepare_call(params=params)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_prepared_bad_params_and_capacity_refuse_before_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing real operands cannot publish a generation or enter dated factories."""
    model = _public_model_from_declarations(declarations=_declarations())
    calls: list[bool] = []

    def observe(
        *args: object,  # noqa: PAN001 - refusal sentinel never reads operands
        **kwargs: object,  # noqa: PAN001 - refusal sentinel never reads operands
    ) -> Never:
        del args, kwargs
        calls.append(True)
        raise AssertionError("A refused call reached a dated factory.")

    monkeypatch.setattr(model_module, "prepare_dated_call_support", observe)
    monkeypatch.setattr(model_module, "prepare_dated_parameters", observe)
    assert model._generation_cache.roots() == ()
    with pytest.raises(InvalidParamsError):
        model._prepare_call(params={})
    assert model._generation_cache.roots() == ()
    with pytest.raises(ExecutionPlanningError, match="budget before allocation"):
        _public_model_from_declarations(
            declarations=replace(
                _declarations(),
                execution_config=ExecutionConfig(device_memory_bytes=1),
            )
        )
    assert calls == []


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_case_r_archive_receiver_refuses_a_tight_budget_before_effect(
    *, tmp_path: Path
) -> None:
    """A receiver whose budget cannot hold the replay refuses before it retains."""
    declarations = _declarations()
    result = _public_model_from_declarations(declarations=declarations).solve(
        params=_params(), log_level="warning", max_compilation_workers=1
    )
    path = tmp_path / "tight-receiver.lcm"
    result.save(path=path)
    receiver = _public_model_from_declarations(
        declarations=replace(
            declarations,
            execution_config=ExecutionConfig(device_memory_bytes=1 << 16),
        )
    )
    with pytest.raises(
        ExecutionPlanningError,
        match="No workspace-width candidate fits the 65536-byte budget",
    ):
        receiver.simulate(
            params=_params(),
            solution=load_solution(path=path),
            seed=19,
            log_level="warning",
            initial_conditions={
                "regime_id": np.array([0]),
                "age": np.array([0]),
                "wealth": np.asarray([4.0], dtype=canonical_float_dtype()),
            },
        )
    assert not receiver._generation_cache._cached


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_public_budgeted_case_r_archive_receiver_preserves_original_delivery(
    *, tmp_path: Path
) -> None:
    """A fresh budgeted public receiver owns the original default Case-R archive."""
    declarations = _declarations()
    params = _params()
    producer = _public_model_from_declarations(declarations=declarations)
    result = producer.solve(
        params=params, log_level="warning", max_compilation_workers=1
    )
    assert result.metadata.retention is ResultRetention.VALUES_AND_REPLAY
    view = result._engine_view
    assert isinstance(view, OwnedSolutionView)
    producing_generation = view.generation
    assert producing_generation is not None
    child_value = np.asarray(view.values[1]["child"])
    wealth = np.asarray([1.0, 12.5], dtype=child_value.dtype)
    expected = np.log(wealth / 2.0) + np.log((wealth + 1.0) / 2.0)
    tolerance = 64 * np.finfo(child_value.dtype).eps
    np.testing.assert_allclose(child_value, expected, rtol=tolerance, atol=tolerance)
    initial_conditions = {
        "regime_id": np.array([0]),
        "age": np.array([0]),
        "wealth": np.asarray([4.0], dtype=canonical_float_dtype()),
    }
    frame = producer.simulate(
        params=params,
        solution=result,
        seed=19,
        log_level="warning",
        initial_conditions=initial_conditions,
    )
    parent_actions = np.asarray([0.25, 2.0], dtype=canonical_float_dtype())
    child_landings = 2.0 * (4.0 - parent_actions) + 0.5
    grid_scores = (
        np.log(parent_actions)
        + np.log(child_landings / 2.0)
        + np.log((child_landings + 1.0) / 2.0)
    )
    winner = int(np.argmax(grid_scores))
    assert grid_scores[winner] > grid_scores[1 - winner]
    np.testing.assert_array_equal(
        frame.raw_results["parent"][0].actions["consumption"],
        parent_actions[winner : winner + 1],
    )
    np.testing.assert_array_equal(
        frame.raw_results["child"][1].states["wealth"],
        child_landings[winner : winner + 1],
    )
    expected_frame = frame.to_dataframe()
    assert not expected_frame.empty
    path = tmp_path / "budgeted-case-r-delivery.lcm"
    result.save(path=path)
    loaded = load_solution(path=path)
    assert loaded._engine_view is None
    receiver = _public_model_from_declarations(
        declarations=replace(
            declarations,
            execution_config=ExecutionConfig(device_memory_bytes=1 << 28),
        )
    )
    replay = receiver.simulate(
        params=params,
        solution=loaded,
        seed=19,
        log_level="warning",
        initial_conditions=initial_conditions,
    )
    pd.testing.assert_frame_equal(
        replay.to_dataframe(), expected_frame, check_exact=True
    )
    (receiving_generation,) = tuple(receiver._generation_cache._cached.values())
    assert receiving_generation is not producing_generation
    assert receiving_generation.key == producing_generation.key
    assert receiving_generation.execution is not receiver
    actual = receiving_generation.solver_preparation(regime_name="child", period=1)
    actual.validate_owner()
    reader = receiving_generation.execution._regimes[
        "parent"
    ].simulation.projected_forward_readers_by_period[0]["child"]
    assert reader.date is actual.declarations["child"].date_at(period=1)
