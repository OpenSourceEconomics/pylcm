"""Canonical DC query authority is independent of numerical carry metadata."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from dags import concatenate_functions

from _lcm import continuation, model_processing
from _lcm.constraints.dynamic_bound import validate_declared_bounds
from _lcm.engine import Regime as EngineRegime
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.regime_building import processing
from _lcm.regime_building.age_normalization import (
    PeriodizedEconFunction,
    resolve_periodized_nodes,
)
from _lcm.regime_law import RegimeLaws
from _lcm.solution.contract import SolutionKernels
from _lcm.solution.dcegm import _BoundDCEGM
from _lcm.time import TimeAxis
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ConsumptionSavingsRegime,
    DiscreteGrid,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    categorical,
    fixed_transition,
    post_decision_lower_bound,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.solvers import DCEGM, ExactEnvelope, MSSEnvelope
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    FunctionName,
    RegimeName,
    ScalarInt,
    UserFunction,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm.test_dynamic_egm_floor import build_dynamic_egm_model

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class Income:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class Work:
    retired: ScalarInt
    working: ScalarInt


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class MixedRegimeId:
    saving: ScalarInt
    scalar: ScalarInt
    done: ScalarInt


def _resources(liquid: ContinuousState) -> FloatND:
    return 2.0 * liquid + 1.0


def _utility(
    *,
    consumption: ContinuousAction,
    passive_a: ContinuousState,
    passive_b: ContinuousState,
    income: DiscreteState,
    work: DiscreteAction,
    utility_scale: float,
) -> FloatND:
    return jnp.log(consumption) + utility_scale * (
        passive_a + passive_b + income + work
    )


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _floor_helper(*, passive_a: ContinuousState, floor_scale: float) -> FloatND:
    return passive_a * floor_scale


def _specialized_floor(age: float) -> UserFunction:
    def floor(*, passive_a: ContinuousState, floor_scale: float) -> FloatND:
        return passive_a * floor_scale + 0.25 * age

    return floor


def _age_signature(age: float) -> float:
    return age


def _lower(*, floor_helper: FloatND, period: ScalarInt) -> FloatND:
    return floor_helper + 0.1 * period


def _eligible(
    *, income: DiscreteState, work: DiscreteAction, eligibility: float
) -> BoolND:
    return income + work >= eligibility


def _next_liquid(savings: FloatND) -> ContinuousState:
    return savings + 4.0


def _next_liquid_direct(
    *, savings: FloatND, liquid: ContinuousState
) -> ContinuousState:
    return savings + 4.0 + liquid


def _terminal(liquid: ContinuousState) -> FloatND:
    return jnp.log(liquid)


def _make_model(
    *,
    asset: bool = False,
    dynamic: bool = True,
    specialized: bool = False,
    exact: bool = True,
) -> Model:
    """Declare unequal extents and interleaved roles for independent axis checks."""
    liquid = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    saving = ConsumptionSavingsRegime(
        liquid=liquid,
        states={
            "passive_b": IrregSpacedGrid(points=np.array([0.0, 0.5, 1.0])),
            "liquid": IrregSpacedGrid(points=np.array([0.5, 1.0, 2.0, 3.0, 4.0])),
            "income": DiscreteGrid(category_class=Income),
            "passive_a": IrregSpacedGrid(points=np.array([0.0, 1.0])),
        },
        actions={
            "consumption": IrregSpacedGrid(points=np.array([0.1, 0.5, 1.0])),
            "work": DiscreteGrid(category_class=Work),
        },
        functions={
            "utility": _utility,
            "resources": _resources,
            "savings": _savings,
            **(
                {
                    "floor_helper": AgeSpecializedFunction(
                        build=_specialized_floor, signature=_age_signature
                    )
                    if specialized
                    else _floor_helper
                }
                if dynamic
                else {}
            ),
        },
        constraints={
            "borrowing": post_decision_lower_bound(
                margin=liquid, lower=_lower if dynamic else 0.0
            ),
            "eligible": _eligible,
        },
        state_transitions={
            "liquid": _next_liquid_direct if asset else _next_liquid,
            "income": fixed_transition("income"),
            "passive_a": fixed_transition("passive_a"),
            "passive_b": fixed_transition("passive_b"),
        },
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=np.array([0.0, 0.25, 4.0])),
            n_constrained_points=4,
            refined_grid_factor=2.0,
            envelope=ExactEnvelope(max_runs=7) if exact else MSSEnvelope(),
        ),
    )
    return Model(
        regimes={
            "saving": saving,
            "done": Regime(
                states={
                    "liquid": IrregSpacedGrid(points=np.array([0.5, 1.0, 4.0, 8.0]))
                },
                functions={"utility": _terminal},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={"saving": {"saving": 0, "done": 1}},
        initial_nodes={0: "saving"},
        regime_id_class=RegimeId,
    )


class _CapturedBuilderError(Exception):
    """Stop a static fixture before any numerical kernel builder runs."""


def _capture_processor(
    *,
    monkeypatch: pytest.MonkeyPatch,
    asset: bool = False,
    dynamic: bool = True,
    scalar: bool = False,
    specialized: bool = False,
    exact: bool = True,
) -> processing._CanonicalRegimeBuilder:
    """Capture canonical inputs while bypassing only the unsupported DC floor route."""
    real_validate = model_processing._validate_borrowing_bound_routes
    captured: list[processing._CanonicalRegimeBuilder] = []

    def validate(
        *,
        user_regimes: Mapping[RegimeName, Regime],
        laws: RegimeLaws,
        ages: TimeAxis,
        active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
        materializer: DatedGenerationMaterializer | None = None,
    ) -> None:
        validation_error = None
        try:
            real_validate(
                user_regimes=user_regimes,
                laws=laws,
                ages=ages,
                active_periods_by_regime=active_periods_by_regime,
                materializer=materializer,
            )
        except ModelInitializationError as error:
            validation_error = error
        if validation_error is not None:
            expected_message = (
                "Regime 'saving' declares a callable borrowing bound; "
                "its EGM-family route does not implement dynamic floors."
            )
            assert str(validation_error) == expected_message
            assert isinstance(user_regimes["saving"].solver, _BoundDCEGM)
            installed = validate_declared_bounds(
                user_regime=user_regimes["saving"],
                regime_name="saving",
                law=laws["saving"],
                ages=ages,
                periods=active_periods_by_regime["saving"],
                materializer=materializer,
            )
            assert len(installed) == 1
            assert installed[0].kind == "dag"

    # keyword-only-exempt: library-callback=Python.descriptor_binding
    def capture(
        builder: processing._CanonicalRegimeBuilder,
        *,
        gated_continuations_by_source: Mapping[
            RegimeName, Mapping[RegimeName, processing.GatedContinuationSchedule]
        ],
    ) -> MappingProxyType[RegimeName, EngineRegime]:
        del gated_continuations_by_source
        captured.append(builder)
        raise _CapturedBuilderError

    with monkeypatch.context() as patch:
        patch.setattr(model_processing, "_validate_borrowing_bound_routes", validate)
        patch.setattr(processing._CanonicalRegimeBuilder, "__call__", capture)
        if scalar:
            with pytest.raises(_CapturedBuilderError):
                build_dynamic_egm_model()
        else:
            with pytest.raises(_CapturedBuilderError):
                _make_model(
                    asset=asset, dynamic=dynamic, specialized=specialized, exact=exact
                )
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("asset", [False, True])
def test_dc_query_preparation_uses_actual_row_axes_capacity_and_corner(
    *, monkeypatch: pytest.MonkeyPatch, asset: bool
) -> None:
    """Ordinary rows and liquid-row corners use their actual existing capacities."""
    processor = _capture_processor(monkeypatch=monkeypatch, asset=asset)
    prepare = getattr(processing, "_prepare_dcegm_query_models", None)
    assert prepare is not None, (
        "DC canonical query pools must be prepared before kernel builders."
    )
    models = prepare(processor=processor)
    assert tuple(models) == ("saving",)
    model = models["saving"]
    spec = model.authority
    assert spec.state_names == ("income", "passive_b", "passive_a")
    assert spec.action_names == ("work",)
    assert spec.row_capacity == (5 if asset else 14)
    assert spec.corner_coordinate_kind == (
        "liquid_row" if asset else "compact_interval"
    )
    assert spec.coordinate_kind == "resources"
    assert spec.marginal_convention == "ordinary_linear"
    assert spec.interval_ids == (0,)
    assert spec.interval_side_ids == (0,)
    assert spec.target_periods == (0, 1)
    assert spec.schema_version == 2
    assert spec.row_topology.kind == "exact_monotone_run"
    assert spec.row_topology.owner_capacity == 7
    assert spec.row_topology.side_ids == (-1, 0, 1)
    assert spec.row_topology.schema_version == 1
    np.testing.assert_array_equal(model.offsets, [0.0, 0.25, 4.0])
    inputs = dict(spec.evaluator_inputs)
    assert inputs["resources"] == ("liquid",)
    assert "passive_a" in inputs["lower_bound"]
    assert "floor_helper__floor_scale" in inputs["lower_bound"]
    assert "period" in inputs["lower_bound"]
    assert "utility__utility_scale" in inputs["utility"]
    assert "eligible__eligibility" in inputs["predicate:eligible"]
    assert dict(spec.evaluator_targets)["predicate:borrowing"] == "borrowing"
    assert model.functions["koopmans_aggregator"] is model.koopmans_aggregator
    assert (
        model.authority.bound.lower_target
        == processor.post_decision_bounds["saving"][0].lower_target
    )


def test_literal_dc_does_not_allocate_feature_query_model(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary literal DC model retains its optional-none representation."""
    processor = _capture_processor(monkeypatch=monkeypatch, dynamic=False)
    prepare = getattr(processing, "_prepare_dcegm_query_models", None)
    assert prepare is not None, (
        "DC canonical query preparation needs a no-feature path."
    )
    assert prepare(processor=processor) is None


def test_dc_query_preparation_preserves_target_period_specialization(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every target period authenticates its own canonical callable closure."""
    processor = _capture_processor(monkeypatch=monkeypatch, specialized=True)
    models = processing._prepare_dcegm_query_models(processor=processor)
    assert models is not None
    model = models["saving"]
    helper = model.functions["floor_helper"]
    assert isinstance(helper, PeriodizedEconFunction)
    assert helper.resolve(0) is not helper.resolve(1)
    pools = {
        period: cast(
            "Mapping[FunctionName, UserFunction]",
            resolve_periodized_nodes(mapping=model.functions, period=period),
        )
        for period in (0, 1)
    }
    values = []
    for period in (0, 1):
        lower = concatenate_functions(
            pools[period], targets=model.authority.bound.lower_target
        )
        values.append(
            lower(
                passive_a=jnp.asarray(2.0),
                floor_helper__floor_scale=0.5,
                period=period,
            )
        )
    np.testing.assert_allclose(values, [1.0, 1.35], atol=2e-7, rtol=0.0)
    continuation.validate_query_evaluator_authority(
        spec=model.authority,
        functions=pools[0],
        target_periods=(0, 1),
        functions_by_period=pools,
    )
    substituted = {
        **pools,
        1: {**pools[1], "floor_helper": pools[0]["floor_helper"]},
    }
    with pytest.raises(TypeError, match="dependencies"):
        continuation.validate_query_evaluator_authority(
            spec=model.authority,
            functions=pools[0],
            target_periods=(0, 1),
            functions_by_period=substituted,
        )


def test_feature_dc_query_preparation_refuses_non_exact_envelope(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run provenance requires the declared exact-envelope compiler route."""
    processor = _capture_processor(monkeypatch=monkeypatch, exact=False)
    with pytest.raises(RegimeInitializationError, match="requires ExactEnvelope"):
        processing._prepare_dcegm_query_models(processor=processor)


def test_scalar_query_pool_remains_independent_of_dc_pool(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical maps preserve scalar feature models without fabricating DC records."""
    processor = _capture_processor(monkeypatch=monkeypatch, scalar=True)
    prepare = getattr(processing, "_prepare_dcegm_query_models", None)
    assert prepare is not None, "DC pools must coexist with scalar pools."
    assert prepare(processor=processor) is None
    scalar = processing._prepare_scalar_egm_query_models(processor=processor)
    assert scalar is not None
    assert tuple(scalar) == ("saving",)
    assert scalar["saving"].authority.row_topology is None


def _bound() -> PostDecisionBoundSpec:
    return PostDecisionBoundSpec(
        post_decision="savings",
        liquid_state="liquid",
        action="consumption",
        resources_target="resources",
        kind="dag",
        literal_value=None,
        lower_target="floor",
        lower_inputs=("floor_scale",),
        dependency_identity=(),
        declaration_names=("borrowing",),
    )


@pytest.mark.parametrize(
    "field", ["owner_capacity", "side_ids", "schema_version", "kind", "absent"]
)
def test_row_topology_is_compared_fieldwise_against_independent_model(
    *, field: Literal["owner_capacity", "side_ids", "schema_version", "kind", "absent"]
) -> None:
    """Copied descriptors cannot substitute different run ownership semantics."""
    topology_type = getattr(continuation, "EGMRowTopologySpec", None)
    assert topology_type is not None, (
        "DC run ownership needs an independent topology descriptor."
    )
    topology = topology_type(kind="exact_monotone_run", owner_capacity=7)
    spec = continuation.EGMQuerySpec(
        bound=_bound(),
        state_names=(),
        action_names=(),
        coordinate_kind="resources",
        marginal_convention="ordinary_linear",
        corner_coordinate_kind="compact_interval",
        row_capacity=14,
        interval_ids=(0,),
        interval_side_ids=(0,),
        evaluator_targets=(),
        evaluator_inputs=(),
        evaluator_identity=(),
        target_periods=(0, 1),
        row_topology=topology,
    )
    continuation.validate_query_model_authority(
        spec=replace(spec, row_topology=replace(topology)),
        model_bound=spec.bound,
        state_names=(),
        action_names=(),
        model_spec=spec,
    )
    if field == "owner_capacity":
        changed = replace(topology, owner_capacity=8)
    elif field == "side_ids":
        changed = replace(topology, side_ids=(0, -1, 1))
    elif field == "absent":
        changed = None
    elif field == "kind":
        with pytest.raises(BeartypeCallHintParamViolation, match="kind"):
            replace(topology, kind="other")
        return
    else:
        with pytest.raises(TypeError, match="topology"):
            replace(topology, schema_version=2)
        return
    with pytest.raises(TypeError, match="topology"):
        continuation.validate_query_model_authority(
            spec=replace(spec, row_topology=changed),
            model_bound=spec.bound,
            state_names=(),
            action_names=(),
            model_spec=spec,
        )


def test_installed_query_status_slot_mapping_is_snapshotted_and_sealed() -> None:
    """Late reader declaration cannot replace actual installed status ownership."""
    bound = _bound()
    slots = {0: (("saving", 0, bound), ("done", 1, bound))}
    kernels = SolutionKernels(period_kernels={}, query_input_authorities=slots)
    slots.clear()
    assert kernels.query_input_authorities is not None
    assert tuple(kernels.query_input_authorities) == (0,)
    assert kernels.query_input_authorities[0][0][2] is bound
    declared = replace(kernels, query_input_authorities={0: (("other", 0, bound),)})
    with pytest.raises(RegimeInitializationError, match="query_input_authorities"):
        processing._fail_if_a_declaration_moved_more_than_the_kernels(
            regime_name="saving",
            solver=DCEGM(savings_grid=IrregSpacedGrid(points=np.array([0.0, 1.0]))),
            handed_over=kernels,
            declared=declared,
        )
