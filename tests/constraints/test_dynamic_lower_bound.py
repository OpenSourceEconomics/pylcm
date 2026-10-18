"""Callable borrowing floors are named nodes in the ordinary constraint DAG."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from dags import concatenate_functions, with_signature

from _lcm.constraints.bounds import proves_the_savings_grids_lower_bound
from _lcm.constraints.dispositions import ConstraintContext
from _lcm.constraints.dynamic_bound import (
    PostDecisionBoundSpec,
    ScalarFloor,
    validate_declared_bounds,
)
from _lcm.constraints.processed import ConstraintLike, normalize_constraints
from _lcm.constraints.routes import BoundConstraint, ConstraintSite
from _lcm.continuation import EGMContinuationSpec
from _lcm.regime_law import RegimeLaw
from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    LiquidMargin,
    Model,
    NormalIIDProcess,
    Periods,
    PeriodSpecializedFunction,
    Phased,
    Regime,
    RouwenhorstAR1Process,
    categorical,
    fixed_transition,
    post_decision_lower_bound,
    ref,
)
from lcm.collective import CollectiveUtility
from lcm.consumption_savings_regime import NestedConsumptionSavingsRegime
from lcm.exceptions import ModelInitializationError
from lcm.solvers import EGM, AdaptiveOuterMesh, GridSearch
from lcm.transition import AgeSpecializedFunction
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousState,
    FloatND,
    FunctionName,
    ScalarInt,
    UserFunction,
)
from tests.solution.test_egm_model_validation import _model as _egm_model
from tests.test_models import n_nbegm_toy

LIQUID = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)


def savings(*, resources: FloatND, consumption: FloatND) -> FloatND:
    return resources - consumption


def resources(*, wealth: FloatND) -> FloatND:
    return wealth


def floor_helper(*, income: FloatND, floor_scale: FloatND) -> FloatND:
    return income * floor_scale


def lower(*, floor_helper: FloatND) -> FloatND:
    return floor_helper


def test_callable_floor_resolves_helper_and_parameter_leaves() -> None:
    """The registered floor composes helpers before comparing actual savings."""
    regime = Regime(
        functions={
            "savings": savings,
            "resources": resources,
            "floor_helper": floor_helper,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=LIQUID, lower=lower)
        },
        solver=GridSearch(),
    )
    constraints = normalize_constraints(
        constraints=cast("Mapping[str, ConstraintLike]", regime.decomposed_constraints)
    )
    constraint = constraints["borrowing"]
    pool = dict(cast("Mapping[str, UserFunction]", regime.decomposed_functions))
    assert "borrowing_lower" in pool
    pool["borrowing"] = constraint.as_function(pool=pool)
    evaluate = concatenate_functions(functions=pool, targets="borrowing")
    assert evaluate(
        wealth=jnp.array(2.0),
        consumption=jnp.array(1.0),
        income=jnp.array(2.0),
        floor_scale=jnp.array(0.5),
    )
    assert not evaluate(
        wealth=jnp.array(2.0),
        consumption=jnp.array(1.25),
        income=jnp.array(2.0),
        floor_scale=jnp.array(0.5),
    )


@categorical(ordered=False)
class RegimeId:
    done: ScalarInt


def utility(*, consumption: FloatND, income: FloatND) -> FloatND:
    return consumption + 0.0 * income


def _model(
    *,
    declaration: UserFunction,
    helper: UserFunction | Phased = floor_helper,
    functions: Mapping[FunctionName, UserFunction | Phased] | None = None,
    margin: LiquidMargin = LIQUID,
    extra_actions: Mapping[ActionName, LinSpacedGrid] | None = None,
) -> Model:
    pool = {
        "utility": utility,
        "savings": savings,
        "resources": resources,
        "floor_helper": helper,
        **(functions or {}),
    }
    regime = ConsumptionSavingsRegime(
        liquid=margin,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
            "income": LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
        },
        actions={
            "consumption": LinSpacedGrid(start=0.25, stop=2.0, n_points=8),
            **(extra_actions or {}),
        },
        functions=pool,
        constraints={"borrowing": declaration},
        solver=GridSearch(),
    )
    return Model(
        regimes={"done": regime},
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        regime_id_class=RegimeId,
        initial_nodes={0: "done"},
    )


def _helper_reading(name: str) -> UserFunction:
    @with_signature(args={name: "FloatND"}, return_annotation="FloatND")
    def helper(**kwargs: FloatND) -> FloatND:
        return kwargs[name]

    return helper


@pytest.mark.parametrize("handwritten", [False, True])
@pytest.mark.parametrize("leaf", ["wealth", "consumption", "resources", "savings"])
def test_liquid_or_action_leaf_through_helper_is_refused(
    *, leaf: str, handwritten: bool
) -> None:
    """Both spellings report the forbidden liquid/action path through a helper."""
    declaration = (
        ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=lower)
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"borrowing.*(wealth|consumption).*(floor_helper|resources|savings)",
    ):
        _model(
            declaration=declaration,
            helper=_helper_reading(leaf),
            functions={"floor": lower} if handwritten else None,
        )


def test_declaration_retains_all_margin_roles_under_grid_search() -> None:
    """A declaration using different liquid roles cannot borrow enclosing roles."""
    wrong = replace(LIQUID, state="other_wealth")
    with pytest.raises(ModelInitializationError, match=r"borrowing.*margin"):
        _model(declaration=post_decision_lower_bound(margin=wrong, lower=lower))


def test_generated_floor_name_cannot_overwrite_a_function() -> None:
    """The floor's deterministic name is reserved against declared producers."""
    with pytest.raises(ModelInitializationError, match=r"borrowing.*borrowing_lower"):
        _model(
            declaration=post_decision_lower_bound(margin=LIQUID, lower=lower),
            functions={"borrowing_lower": resources},
        )


@pytest.mark.parametrize("handwritten", [False, True])
def test_phase_varying_floor_helper_is_refused(*, handwritten: bool) -> None:
    """The floor has one feasible set in solve and simulation phases."""
    declaration = (
        ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=lower)
    )
    with pytest.raises(
        ModelInitializationError, match=r"borrowing.*phase-varying.*floor_helper"
    ):
        _model(
            declaration=declaration,
            helper=Phased(solve=floor_helper, simulate=_helper_reading("income")),
            functions={"floor": lower} if handwritten else None,
        )


@pytest.mark.parametrize("handwritten", [False, True])
def test_grid_search_solves_named_floor_with_transitive_parameter(
    *, handwritten: bool
) -> None:
    """Income-specific floors limit consumption at the represented state nodes."""
    declaration = (
        ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=lower)
    )
    model = _model(
        declaration=declaration, functions={"floor": lower} if handwritten else None
    )
    values = model.solve(params={"floor_scale": 0.5}, log_level="off").values[0]["done"]

    np.testing.assert_allclose(values, [[1.0, 0.5], [2.0, 1.5]], atol=1e-6)


@pytest.mark.parametrize("bad", [jnp.array([0.0]), jnp.array(1.0 + 0.0j)])
def test_floor_is_a_real_scalar_when_the_dag_is_evaluated(bad: FloatND) -> None:
    """A floor does not broadcast a vector or impose a complex ordering."""

    def invalid() -> FloatND:
        return bad

    regime = Regime(
        functions={},
        constraints={
            "borrowing": post_decision_lower_bound(margin=LIQUID, lower=invalid)
        },
    )
    node = cast("UserFunction", regime.decomposed_functions["borrowing_lower"])
    with pytest.raises(
        ModelInitializationError, match=r"borrowing.*real floating scalar"
    ):
        node()


def test_plain_egm_literal_floor_must_agree_with_grid_origin() -> None:
    """A literal claim cannot override the savings grid's actual origin."""
    margin = replace(LIQUID, resources="wealth")
    with pytest.raises(ModelInitializationError, match=r"lower bound.*0.25"):
        _egm_model(constraint=post_decision_lower_bound(margin=margin, lower=0.25))


def test_callable_plain_egm_installs_its_model_bound_query_authority() -> None:
    """A scalar callable floor has the canonical target's query declaration."""
    margin = replace(LIQUID, resources="wealth")

    def zero() -> FloatND:
        return jnp.asarray(0.0)

    model = _egm_model(constraint=post_decision_lower_bound(margin=margin, lower=zero))
    continuation = model._regimes["saving"].solution.continuation_spec
    assert isinstance(continuation, EGMContinuationSpec)
    assert continuation.layout.query_spec is not None
    assert continuation.layout.query_spec.bound.lower_target == "cap_lower"


@categorical(ordered=False)
class TemporalRegimeId:
    saving: ScalarInt
    done: ScalarInt


_TEMPORAL_EXECUTION = ExecutionConfig()


def _temporal_model(
    *,
    lower_function: UserFunction,
    clock: str = "age",
    handwritten: bool = False,
    initial_entries: tuple[int, ...] = (0,),
    execution_config: ExecutionConfig = _TEMPORAL_EXECUTION,
) -> Model:
    grid = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
    income = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)

    def next_wealth(*, savings: FloatND) -> FloatND:
        return savings

    def terminal(*, wealth: FloatND, income: FloatND) -> FloatND:
        return 0.0 * (wealth + income)

    saving = ConsumptionSavingsRegime(
        liquid=LIQUID,
        states={"wealth": grid, "income": income},
        actions={"consumption": LinSpacedGrid(start=0.25, stop=2.0, n_points=8)},
        state_transitions={"wealth": next_wealth, "income": fixed_transition("income")},
        functions={
            "utility": utility,
            "resources": resources,
            "savings": savings,
            **({"floor": lower_function} if handwritten else {}),
        },
        constraints={
            "borrowing": (
                ref("savings") >= ref("floor")
                if handwritten
                else post_decision_lower_bound(margin=LIQUID, lower=lower_function)
            )
        },
    )
    done = Regime(
        states={"wealth": grid, "income": income}, functions={"utility": terminal}
    )
    return Model(
        regimes={"saving": saving, "done": done},
        edges={
            "saving": {
                "saving": Periods(values=(0,)) if clock == "period" else 0,
                "done": Periods(values=(1,)) if clock == "period" else 1,
            }
        },
        n_periods=3 if clock == "period" else None,
        ages=None if clock == "period" else AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=TemporalRegimeId,
        initial_nodes=InitialNodes(by_period=dict.fromkeys(initial_entries, "saving"))
        if clock == "period"
        else dict.fromkeys(initial_entries, "saving"),
        execution_config=execution_config,
    )


@pytest.mark.parametrize("handwritten", [False, True])
def test_generated_floor_preserves_age_specialization(*, handwritten: bool) -> None:
    """Named age-resolved floors keep their period-specific values in GridSearch."""

    def build(age: float) -> UserFunction:
        def floor(*, income: FloatND) -> FloatND:
            return 0.0 * income + 0.25 * age

        return floor

    model = _temporal_model(
        lower_function=AgeSpecializedFunction(build=build, signature=lambda age: age),
        handwritten=handwritten,
    )
    values = model.solve(params={"discount_factor": 1.0}, log_level="off").values[1][
        "saving"
    ]

    np.testing.assert_allclose(values, [[0.75, 0.75], [1.75, 1.75]], atol=1e-6)


@pytest.mark.parametrize("handwritten", [False, True])
def test_every_age_specific_floor_dependency_is_validated(*, handwritten: bool) -> None:
    """A liquid dependency in a later concrete producer is still forbidden."""

    def build(age: float) -> UserFunction:
        return _helper_reading("income" if age == 0 else "wealth")

    with pytest.raises(ModelInitializationError, match=r"borrowing.*wealth"):
        _temporal_model(
            lower_function=AgeSpecializedFunction(
                build=build, signature=lambda age: age
            ),
            handwritten=handwritten,
        )


@pytest.mark.parametrize("handwritten", [False, True])
def test_direct_floor_parameter_has_one_declared_namespace(
    *, handwritten: bool
) -> None:
    """A direct floor parameter belongs only to its named producer."""

    def direct(*, theta: FloatND) -> FloatND:
        return theta

    declaration = (
        ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=direct)
    )
    model = _model(
        declaration=declaration, functions={"floor": direct} if handwritten else None
    )
    locations = [
        name
        for name, params in model.get_params_template()["done"].items()
        if "theta" in params
    ]
    assert locations == ["floor" if handwritten else "borrowing_lower"]


@pytest.mark.parametrize(
    ("clock", "marker", "accepted"),
    [
        ("period", PeriodSpecializedFunction, True),
        ("age", PeriodSpecializedFunction, False),
        ("period", AgeSpecializedFunction, False),
    ],
)
@pytest.mark.parametrize("handwritten", [False, True])
def test_floor_marker_keeps_its_clock_kind(
    *,
    clock: str,
    marker: type[PeriodSpecializedFunction | AgeSpecializedFunction],
    accepted: bool,
    handwritten: bool,
) -> None:
    """A named floor obeys the declared age or period clock."""

    def build(coordinate: float) -> UserFunction:
        def floor(*, income: FloatND) -> FloatND:
            return 0.0 * income + 0.25 * coordinate

        return floor

    declaration = marker(build=build, signature=lambda coordinate: coordinate)
    if accepted:
        model = _temporal_model(
            lower_function=declaration, clock=clock, handwritten=handwritten
        )
        values = model.solve(params={"discount_factor": 1.0}, log_level="off").values[
            1
        ]["saving"]

        np.testing.assert_allclose(values, [[0.75, 0.75], [1.75, 1.75]], atol=1e-6)
    else:
        with pytest.raises(ModelInitializationError, match=r"wrong coordinate kind"):
            _temporal_model(
                lower_function=declaration, clock=clock, handwritten=handwritten
            )


@pytest.mark.parametrize("handwritten", [False, True])
def test_grid_search_simulation_checks_actual_subject_floor(
    *, handwritten: bool
) -> None:
    """Simulation chooses the largest grid action satisfying the actual floor."""
    declaration = (
        ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=lower)
    )
    model = _model(
        declaration=declaration, functions={"floor": lower} if handwritten else None
    )
    frame = model.simulate(
        params={"floor_scale": 0.5},
        initial_conditions={
            "wealth": jnp.array([2.0]),
            "income": jnp.array([0.5]),
            "age": jnp.array([0]),
            "regime_id": jnp.array([RegimeId.done], dtype=jnp.int32),
        },
        log_level="off",
    ).to_dataframe()
    assert frame["consumption"].iloc[0] == pytest.approx(1.75)


def test_literal_grid_proof_requires_exact_installed_spec() -> None:
    """Only the installed literal value is discharged by grid construction."""

    spec = PostDecisionBoundSpec(
        post_decision="savings",
        liquid_state="wealth",
        action="consumption",
        resources_target="resources",
        kind="literal",
        literal_value=0.0,
        lower_target=None,
        lower_inputs=(),
        dependency_identity=(((), 0.0),),
    )
    proof = proves_the_savings_grids_lower_bound(
        post_decision="savings", bound_spec=spec
    )
    site = ConstraintSite(
        stage="savings_stage",
        function_pool=MappingProxyType({}),
        available_names=frozenset(),
    )
    context = ConstraintContext(
        regime_name="saving",
        phase="solve",
        grids=MappingProxyType({}),
        function_names=frozenset(),
        param_names=frozenset(),
    )
    normalized = normalize_constraints(
        constraints={
            "same": ref("savings") >= 0.0,
            "different": ref("savings") >= 0.25,
            "dag": ref("savings") >= ref("floor"),
        }
    )

    def bound(name: str) -> BoundConstraint:
        return BoundConstraint(
            constraint=normalized[name], site=site, transitive_inputs=frozenset()
        )

    assert proof(bound=bound("same"), context=context) is not None
    assert proof(bound=bound("different"), context=context) is None
    assert proof(bound=bound("dag"), context=context) is None


@pytest.mark.parametrize("leaf", ["work", "chosen_outer"])
@pytest.mark.parametrize("handwritten", [False, True])
def test_other_chosen_action_cannot_enter_floor(
    *, leaf: str, handwritten: bool
) -> None:
    """A floor is fixed before every liquid or other action is chosen."""
    declaration = (
        ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=lower)
    )
    with pytest.raises(
        ModelInitializationError, match=rf"borrowing.*{leaf}.*floor_helper"
    ):
        _model(
            declaration=declaration,
            helper=_helper_reading(leaf),
            functions={"floor": lower} if handwritten else None,
            extra_actions={leaf: LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        )


def test_direct_post_decision_dependency_reports_its_leaf_path() -> None:
    """A direct savings read is refused through its liquid/action ancestors."""
    declaration = post_decision_lower_bound(
        margin=LIQUID, lower=_helper_reading("savings")
    )
    with pytest.raises(
        ModelInitializationError, match=r"borrowing.*(wealth|consumption).*savings"
    ):
        _model(declaration=declaration)


@pytest.mark.parametrize("slot", ["state", "action", "constraint"])
def test_generated_floor_cannot_overwrite_other_namespaces(slot: str) -> None:
    """Generated floor targets are reserved against variable and provider nodes."""
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    name = "borrowing"
    target = f"{name}_lower"
    with pytest.raises(ModelInitializationError, match=rf"{name}.*{target}"):
        Regime(
            states={target: grid} if slot == "state" else {},
            actions={target: grid} if slot == "action" else {},
            constraints={
                name: post_decision_lower_bound(margin=LIQUID, lower=lower),
                **({target: ref("savings") >= 0.0} if slot == "constraint" else {}),
            },
        )


def _egm_with_constraints(constraints: Mapping[FunctionName, ConstraintLike]) -> Model:
    baseline = _egm_model()
    saving = baseline.user_regimes["saving"].replace(constraints=constraints)
    return Model(
        regimes={"saving": saving, "done": baseline.user_regimes["done"]},
        edges={"saving": {"done": 0}},
        ages=AgeGrid(exact_values=(0, 1)),
        regime_id_class=TemporalRegimeId,
        initial_nodes={0: "saving"},
    )


def test_equivalent_literal_bounds_share_the_installed_floor() -> None:
    """Repeated claims of the same floor remain compatible."""
    model = _egm_with_constraints(
        {"first": ref("savings") >= 0.0, "second": ref("savings") >= 0.0}
    )
    assert "saving" in model.user_regimes


def test_distinct_structural_floors_are_refused_together() -> None:
    """A solver cannot choose one declaration and silently discard another."""
    with pytest.raises(ModelInitializationError, match=r"first.*second"):
        _egm_with_constraints(
            {"first": ref("savings") >= 0.0, "second": ref("savings") >= 0.25}
        )


def test_runtime_savings_grid_is_named_in_callable_route_refusal() -> None:
    """A callable borrowing declaration requires supplied savings-grid points."""

    def zero() -> FloatND:
        return jnp.asarray(0.0)

    baseline = _egm_model()
    saving = baseline.user_regimes["saving"].replace(
        solver=EGM(savings_grid=IrregSpacedGrid(n_points=3)),
        constraints={
            "borrowing": post_decision_lower_bound(
                margin=replace(LIQUID, resources="wealth"), lower=zero
            )
        },
    )
    with pytest.raises(ModelInitializationError, match=r"savings grid.*runtime"):
        Model(
            regimes={"saving": saving, "done": baseline.user_regimes["done"]},
            edges={"saving": {"done": 0}},
            ages=AgeGrid(exact_values=(0, 1)),
            regime_id_class=TemporalRegimeId,
            initial_nodes={0: "saving"},
        )


@pytest.mark.parametrize(
    ("kind", "equivalent"),
    [
        ("direct_param", False),
        ("helper_param", True),
        ("state", True),
        ("constant", True),
    ],
)
def test_bound_identity_retains_parameter_ownership(
    *, kind: str, equivalent: bool
) -> None:
    """Separate function-level parameter slots cannot become one bound identity."""

    def direct(*, theta: FloatND) -> FloatND:
        return theta

    def zero() -> FloatND:
        return jnp.asarray(0.0)

    producer = {
        "direct_param": direct,
        "helper_param": lower,
        "state": _helper_reading("income"),
        "constant": zero,
    }[kind]
    baseline = _model(
        declaration=post_decision_lower_bound(margin=LIQUID, lower=producer)
    )
    regime = baseline.user_regimes["done"].replace(
        constraints={
            "first": post_decision_lower_bound(margin=LIQUID, lower=producer),
            "second": post_decision_lower_bound(margin=LIQUID, lower=producer),
        }
    )
    specs = validate_declared_bounds(
        user_regime=regime,
        regime_name="done",
        law=RegimeLaw(transition=None),
        ages=AgeGrid(exact_values=(0,)),
        periods=(0,),
    )
    assert (specs[0].dependency_identity == specs[1].dependency_identity) is equivalent


def test_engine_overlay_cannot_rewrite_declared_floor() -> None:
    """A generated floor producer retains its declaration's executable authority."""
    regime = Regime(
        functions={"savings": savings},
        constraints={
            "borrowing": post_decision_lower_bound(margin=LIQUID, lower=lower)
        },
    )
    with pytest.raises(ModelInitializationError, match=r"borrowing.*borrowing_lower"):
        regime.with_engine_functions(
            engine_functions={
                **regime.decomposed_functions,
                "borrowing_lower": resources,
            }
        )


@pytest.mark.parametrize("change", ["roundtrip", "missing", "foreign"])
def test_collective_engine_overlay_preserves_complete_floor_provenance(
    *, change: str
) -> None:
    """Collective overlays reproduce the complete declared floor graph or refuse it."""
    declaration = CollectiveUtility(utilities={"alice": utility, "bob": utility})
    regime = Regime(
        functions={
            "utility": declaration,
            "resources": resources,
            "savings": savings,
            "floor_helper": floor_helper,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=LIQUID, lower=lower)
        },
    )
    supplied = dict(regime.decomposed_functions)
    if change == "missing":
        supplied.pop("borrowing_lower")
    elif change == "foreign":
        supplied["borrowing_lower"] = resources
    if change == "roundtrip":
        written = regime.with_engine_functions(engine_functions=supplied)
        assert written.functions["utility"] is declaration
        assert set(written.decomposed_functions) == set(regime.decomposed_functions)
        restored = written.decomposed_functions["borrowing_lower"]
        assert type(restored) is ScalarFloor
        assert restored.function is lower
        assert restored.constraint_name == "borrowing"
    else:
        with pytest.raises(ModelInitializationError, match="borrowing_lower"):
            regime.with_engine_functions(engine_functions=supplied)


@pytest.mark.parametrize("prefix", ["weight_next", "support_next", "key_next"])
def test_generated_floor_cannot_overwrite_actual_process_provider(prefix: str) -> None:
    """Process-provider names retain ownership when a floor is decomposed."""

    with pytest.raises(ModelInitializationError, match=rf"{prefix}.*{prefix}_lower"):
        Regime(
            states={
                "lower": NormalIIDProcess(
                    n_points=3, gauss_hermite=False, mu=0.0, sigma=1.0
                )
            },
            constraints={prefix: post_decision_lower_bound(margin=LIQUID, lower=lower)},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_literal_floor_is_finite_at_model_construction(value: float) -> None:
    """A known nonfinite declaration is invalid model input."""
    with pytest.raises(ModelInitializationError, match=r"borrowing.*finite"):
        _model(declaration=post_decision_lower_bound(margin=LIQUID, lower=value))


def test_helper_named_resources_is_legal_when_its_leaves_are_fixed() -> None:
    """A helper's input domain determines legality, rather than its spelling."""

    def resource(*, income: FloatND) -> FloatND:
        return income + 1.0

    def uses_wealth(
        *, wealth: FloatND, income: FloatND, consumption: FloatND
    ) -> FloatND:
        return consumption + 0.0 * (wealth + income)

    model = _model(
        declaration=post_decision_lower_bound(margin=LIQUID, lower=lower),
        helper=_helper_reading("resources"),
        functions={"resources": resource, "utility": uses_wealth},
    )
    assert "done" in model.user_regimes


def test_period_leaf_is_an_integer_in_abstract_floor_tracing() -> None:
    """Computational period supports integer operations in a floor DAG."""

    def floor(*, period: int, income: FloatND) -> FloatND:
        return 0.25 * (period & 1) + 0.0 * income

    model = _temporal_model(lower_function=floor, clock="period")
    values = model.solve(params={"discount_factor": 1.0}, log_level="off").values[1][
        "saving"
    ]

    np.testing.assert_allclose(values, [[0.75, 0.75], [1.75, 1.75]], atol=1e-6)


@pytest.mark.parametrize("kind", ["vector_parameter", "boolean_parameter"])
def test_unknown_parameter_shape_is_checked_with_actual_inputs(kind: str) -> None:
    """Scalar floors may read vector or Boolean parameters."""

    def vector(*, theta: FloatND) -> FloatND:
        return theta[0]

    def boolean(*, enabled: BoolND) -> FloatND:
        return jnp.where(~enabled, 0.5, 0.0)

    producer = vector if kind == "vector_parameter" else boolean
    params = (
        {"theta": jnp.array([0.5, 0.75])}
        if kind == "vector_parameter"
        else {"enabled": False}
    )
    model = _model(declaration=post_decision_lower_bound(margin=LIQUID, lower=producer))
    values = model.solve(params={**params, "floor_scale": 0.5}, log_level="off").values[
        0
    ]["done"]

    np.testing.assert_allclose(values, [[0.5, 0.5], [1.5, 1.5]], atol=1e-6)


@pytest.mark.parametrize("kind", ["integer", "boolean", "vector"])
@pytest.mark.parametrize("handwritten", [False, True])
def test_actual_floor_output_is_real_floating_scalar(
    *, kind: str, handwritten: bool
) -> None:
    """Both declaration spellings validate actual parameter-dependent floor outputs."""

    def floor(*, theta: FloatND) -> FloatND:
        if kind == "integer":
            return jnp.asarray(theta, dtype=jnp.int32)
        if kind == "boolean":
            return theta > 0.0
        return jnp.stack([theta])

    declaration = (
        ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=floor)
    )
    model = _model(
        declaration=declaration, functions={"floor": floor} if handwritten else None
    )
    with pytest.raises(
        ModelInitializationError, match=r"borrowing.*real floating scalar"
    ):
        model.solve(params={"theta": 0.5, "floor_scale": 0.5}, log_level="off")


def _adaptive_constant_floor() -> FloatND:
    return jnp.asarray(0.5)


def _adaptive_zero_floor() -> FloatND:
    return jnp.asarray(0.0)


def _adaptive_illiquid_floor(illiquid: ContinuousState) -> FloatND:
    return 0.1 * illiquid


@pytest.mark.parametrize(
    "floor",
    [_adaptive_constant_floor, _adaptive_zero_floor, _adaptive_illiquid_floor],
    ids=["constant", "zero", "illiquid"],
)
def test_adaptive_outer_mesh_refuses_a_callable_floor(floor: UserFunction) -> None:
    """Adaptive outer search accepts only literal floors, refused at model build."""
    seed = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=AdaptiveOuterMesh(initial_grid=n_nbegm_toy.OUTER_GRID),
    )
    alive = cast("NestedConsumptionSavingsRegime", seed.user_regimes["alive"])
    feature = alive.replace(
        constraints={
            "borrowing": post_decision_lower_bound(margin=alive.liquid, lower=floor)
        }
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"adaptive outer search supports only literal floors.*slow",
    ):
        Model(
            regimes={**seed.user_regimes, "alive": feature},
            ages=seed.ages,
            edges=seed.edges,
            fixed_params=seed.fixed_params,
            initial_nodes=seed.initial_nodes,
            regime_id_class=n_nbegm_toy.RegimeId,
        )


@beartype
def _shock_income(*, eps: ContinuousState) -> FloatND:
    return 1.25 + 0.5 * eps


def _shock_floor(*, income: FloatND) -> FloatND:
    return -0.5 * income


@pytest.mark.parametrize("process", ["iid", "ar1"])
def test_floor_reads_a_stochastic_process_at_its_node_values(process: str) -> None:
    """A floor helper reading a process state receives the float node values."""
    # Both declarations have the nodes (-2, 0, 2): the IID grid spans two
    # standard deviations, and Rouwenhorst with rho zero spans sqrt(2) sigma.
    shock = (
        NormalIIDProcess(n_points=3, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2.0)
        if process == "iid"
        else RouwenhorstAR1Process(n_points=3, rho=0.0, sigma=2.0**0.5, mu=0.0)
    )
    regime = ConsumptionSavingsRegime(
        liquid=LIQUID,
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2), "eps": shock},
        actions={"consumption": LinSpacedGrid(start=0.25, stop=3.0, n_points=12)},
        functions={
            "utility": utility,
            "savings": savings,
            "resources": resources,
            "income": _shock_income,
        },
        constraints={
            "borrowing": post_decision_lower_bound(margin=LIQUID, lower=_shock_floor)
        },
        solver=GridSearch(),
    )
    model = Model(
        regimes={"done": regime},
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        regime_id_class=RegimeId,
        initial_nodes={0: "done"},
    )
    values = model.solve(params={}, log_level="off").values[0]["done"]

    # Terminal value is the largest grid consumption with savings at or above
    # the floor: c <= wealth + 0.5 * (1.25 + 0.5 * eps), on the 0.25 grid.
    consumption = [0.25 * (k + 1) for k in range(12)]
    expected = [
        [
            max(c for c in consumption if c <= wealth + 0.5 * (1.25 + 0.5 * eps))
            for wealth in (1.0, 2.0)
        ]
        for eps in (-2.0, 0.0, 2.0)
    ]
    np.testing.assert_allclose(values, expected, atol=1e-6)
