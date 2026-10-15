"""Query rows are interpreted by the model's bound, evaluator and axis authority."""

import inspect
from dataclasses import replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import continuation
from _lcm.egm.carry import build_template_egm_carry, build_template_egm_query_data
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.regime_building import Q_and_F as q_and_f_module
from _lcm.regime_building.processing import _missing_capabilities
from _lcm.solution import feasibility
from _lcm.typing import (
    ConstraintFunction,
    EconFunction,
    EconFunctionArg,
    QAndFArg,
    QAndFKwargs,
)
from lcm import post_decision_lower_bound, ref
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import EGM_ENDOGENOUS_COORDINATE, ContinuationCapabilities
from lcm.typing import BoolND, FloatND, FunctionName, ReferenceName, RegimeName
from tests.constraints.test_dynamic_lower_bound import LIQUID, _model, lower


def _bound() -> PostDecisionBoundSpec:
    """Name the fixed-state floor and its canonical dependency identity."""
    return PostDecisionBoundSpec(
        post_decision="savings",
        liquid_state="liquid",
        action="consumption",
        resources_target="resources",
        kind="dag",
        literal_value=None,
        lower_target="floor",
        lower_inputs=("income", "floor_scale"),
        dependency_identity=(
            (("income",), "income"),
            (("floor_scale",), "floor_scale"),
        ),
        declaration_names=("borrowing",),
    )


@pytest.mark.parametrize("corner", ["compact_interval", "liquid_row"])
def test_model_owned_query_spec_seals_corner_coordinate_and_axes(
    *, corner: str
) -> None:
    """A query template carries its exact model-owned coordinate and axis binding."""
    spec_type = getattr(continuation, "EGMQuerySpec", None)
    assert spec_type is not None, "Query rows need a model-owned static interpretation."
    query_spec = spec_type(
        bound=_bound(),
        state_names=("income",),
        action_names=("work",),
        coordinate_kind="resources",
        marginal_convention="ordinary_linear",
        corner_coordinate_kind=corner,
        row_capacity=5,
        interval_ids=(0,),
        interval_side_ids=(0,),
        evaluator_targets=(("resources", "resources"), ("utility", "utility")),
        evaluator_inputs=(
            ("resources", ("liquid", "income")),
            ("utility", ("consumption",)),
        ),
        evaluator_identity=(
            ("resources", "resource-identity"),
            ("utility", "utility-identity"),
        ),
        target_periods=(0, 1),
    )
    template = build_template_egm_carry(
        n_rows=5,
        leading_shape=(2, 3),
        query_data=build_template_egm_query_data(
            state_shape=(2,),
            action_shape=(3,),
            row_capacity=5,
            interval_capacity=1,
            liquid_corner=corner == "liquid_row",
        ),
    )
    layout = continuation.EGMContinuationLayout(query_spec=query_spec)
    result = continuation.EGMContinuationSpec(template=template, layout=layout)
    assert (
        cast("continuation.EGMQuerySpec", result.layout.query_spec).bound
        is query_spec.bound
    )


def test_feature_template_cannot_use_scalar_only_layout_authority() -> None:
    """Numeric floor provenance cannot supply missing evaluator authority."""
    template = build_template_egm_carry(
        n_rows=5,
        query_data=build_template_egm_query_data(
            state_shape=(),
            action_shape=(),
            row_capacity=5,
            interval_capacity=1,
        ),
    )
    with pytest.raises(TypeError, match=r"query.*authority"):
        continuation.EGMContinuationSpec(template=template)


@pytest.mark.parametrize("change", ["liquid", "lower", "period"])
def test_query_authority_rejects_substituted_bound_identity(*, change: str) -> None:
    """A copied query descriptor cannot authenticate another model's bound."""
    validator = getattr(continuation, "validate_query_bound_authority", None)
    assert validator is not None, "Query bound identity needs independent validation."
    original = _bound()
    replacement = (
        replace(original, liquid_state="wealth")
        if change == "liquid"
        else replace(original, lower_target="other_floor")
        if change == "lower"
        else replace(original, dependency_identity=((("1",), "other-time-binding"),))
    )
    with pytest.raises(TypeError, match=r"bound.*authority"):
        validator(declared=replacement, model_bound=original)


def test_query_authority_rejects_swapped_equal_shape_state_roles() -> None:
    """Matching dimensions cannot authenticate swapped labeled state rows."""
    validator = getattr(continuation, "validate_query_model_authority", None)
    assert validator is not None, "Query roles need independent model validation."
    spec = continuation.EGMQuerySpec(
        bound=_bound(),
        state_names=("income", "health"),
        action_names=(),
        coordinate_kind="resources",
        marginal_convention="ordinary_linear",
        corner_coordinate_kind="compact_interval",
        row_capacity=5,
        interval_ids=(0,),
        interval_side_ids=(0,),
        evaluator_targets=(),
        evaluator_inputs=(),
        evaluator_identity=(),
        target_periods=(0,),
    )
    with pytest.raises(TypeError, match="roles"):
        validator(
            spec=spec,
            model_bound=_bound(),
            state_names=("health", "income"),
            action_names=(),
        )


def _compact_spec() -> continuation.EGMQuerySpec:
    """Describe a fixed-state, conditional discrete-action compact bank."""
    return continuation.EGMQuerySpec(
        bound=_bound(),
        state_names=("income",),
        action_names=("work",),
        coordinate_kind="resources",
        marginal_convention="ordinary_linear",
        corner_coordinate_kind="compact_interval",
        row_capacity=5,
        interval_ids=(0,),
        interval_side_ids=(0,),
        evaluator_targets=(),
        evaluator_inputs=(),
        evaluator_identity=(),
        target_periods=(0,),
    )


@pytest.mark.parametrize(
    "change", ["retained_actions", "outer_count", "value", "marginal", "policy"]
)
def test_feature_layout_and_all_row_banks_agree(*, change: str) -> None:
    """Conditional layout flags and every associated row bank share one meaning."""
    spec = _compact_spec()
    template = build_template_egm_carry(
        n_rows=5,
        leading_shape=(2, 3),
        query_data=build_template_egm_query_data(
            state_shape=(2,),
            action_shape=(3,),
            row_capacity=5,
            interval_capacity=1,
        ),
    )
    layout = continuation.EGMContinuationLayout(query_spec=spec)
    if change == "retained_actions":
        layout = replace(layout, retains_discrete_action_rows=False)
    elif change == "outer_count":
        layout = replace(layout, n_stacked_candidates=3)
    else:
        field = "marginal_utility" if change == "marginal" else change
        template = replace(template, **{field: jnp.zeros((2, 2, 5))})
    with pytest.raises(TypeError, match=r"query|Query"):
        continuation.EGMContinuationSpec(template=template, layout=layout)


@pytest.mark.parametrize("change", ["partial", "both", "neither"])
def test_query_corner_group_requires_exactly_one_representation(*, change: str) -> None:
    """Compact interval and liquid-row corner groups are complete and exclusive."""
    query = build_template_egm_query_data(
        state_shape=(),
        action_shape=(),
        row_capacity=5,
        interval_capacity=1,
    )
    if change == "partial":
        changes = {"corner_kind": None}
    elif change == "both":
        liquid = build_template_egm_query_data(
            state_shape=(),
            action_shape=(),
            row_capacity=5,
            interval_capacity=1,
            liquid_corner=True,
        )
        changes = {"liquid_corner_data": liquid.liquid_corner_data}
    else:
        changes = dict.fromkeys(
            (
                "corner_continuation",
                "corner_kind",
                "corner_interval_id",
                "corner_side_id",
            )
        )
    with pytest.raises(TypeError, match="exactly one complete corner"):
        replace(query, **changes)


@pytest.mark.parametrize("active", [(True, True, False), (True, False, False)])
def test_runtime_bound_summary_ignores_inactive_lanes(
    *, active: tuple[bool, ...]
) -> None:
    """Mandatory invalid-input flags OR actual lanes without economic masking."""
    helper = getattr(feasibility, "summarize_bound_inputs", None)
    assert helper is not None, "Runtime bound validity needs a bounded typed summary."
    result = jax.jit(helper)(
        floor_values=jnp.asarray([0.0, jnp.nan, jnp.inf]),
        active=jnp.asarray(active),
    )
    np.testing.assert_array_equal(
        jnp.stack((result.nonfinite_floor, result.invalid_shifted_grid)),
        [active[1], False],
    )


@pytest.mark.parametrize("collapse", [False, True])
def test_shifted_savings_nodes_keep_floor_first_and_report_order(
    *, collapse: bool
) -> None:
    """Shifted nodes use floor-plus-offset arithmetic and flag collapsed increments."""
    helper = getattr(feasibility, "shift_savings_grid", None)
    assert helper is not None, (
        "Shifted grid arithmetic needs a mandatory integrity result."
    )
    floor = jnp.asarray(2.0**70 if collapse else 1.0)
    nodes, result = jax.jit(helper)(floor=floor, offsets=jnp.asarray([0.0, 4.0, 8.0]))
    np.testing.assert_array_equal(
        jnp.stack(
            (nodes[0] == floor, result.nonfinite_floor, result.invalid_shifted_grid)
        ),
        [True, False, collapse],
    )


def test_runtime_bound_host_validation_uses_named_model_authority() -> None:
    """A bounded invalid floor record raises with its canonical declaration context."""
    helper = getattr(feasibility, "summarize_bound_inputs", None)
    validator = getattr(feasibility, "validate_runtime_bound_status", None)
    assert helper is not None, "Runtime bound validity needs host acceptance."
    assert validator is not None, "Runtime bound validity needs host acceptance."
    record = helper(floor_values=jnp.asarray(jnp.nan), active=jnp.asarray(a=True))
    with pytest.raises(
        InvalidValueFunctionError, match=r"borrowing.*floor.*work.*period 1"
    ):
        validator(
            statuses=(record,), bound_specs=(_bound(),), regime_name="work", period=1
        )


@pytest.mark.parametrize("handwritten", [False, True])
def test_canonical_phases_retain_independent_bound_identity(
    *, handwritten: bool
) -> None:
    """Both declaration spellings bind their named floor in solve and simulation."""
    declaration = (
        ref("savings") >= ref("floor")
        if handwritten
        else post_decision_lower_bound(margin=LIQUID, lower=lower)
    )
    model = _model(
        declaration=declaration,
        functions={"floor": lower} if handwritten else None,
    )
    regime = model._regimes["done"]
    solve_specs = getattr(regime.solution, "post_decision_bounds", None)
    simulate_specs = getattr(regime.simulation, "post_decision_bounds", None)
    assert solve_specs is not None, (
        "Canonical phases must retain model-owned bound identities."
    )
    assert simulate_specs is not None, (
        "Canonical phases must retain model-owned bound identities."
    )
    assert tuple(
        (
            bound.post_decision,
            bound.liquid_state,
            bound.action,
            bound.resources_target,
            bound.kind,
            bound.lower_target,
            bound.lower_inputs,
            bound.declaration_names,
        )
        for bound in solve_specs
    ) == (
        (
            "savings",
            "wealth",
            "consumption",
            "resources",
            "dag",
            "floor" if handwritten else "borrowing_lower",
            ("floor_scale", "income"),
            ("borrowing",),
        ),
    )
    assert simulate_specs is solve_specs


@pytest.mark.parametrize("floor_value", [1.0, float("nan"), float("inf")])
def test_actual_utility_feasibility_and_validity_share_one_floor_evaluation(
    *, floor_value: float
) -> None:
    """Canonical floor validity survives infeasibility without a second DAG pass."""
    builder = q_and_f_module._get_U_and_F
    assert "bound_specs" in inspect.signature(builder).parameters, (
        "Actual U/F evaluation must transport mandatory bound validity."
    )
    calls = []

    def floor(*, income: FloatND, floor_scale: FloatND) -> FloatND:
        calls.append("floor")
        return income * floor_scale

    def utility(*, floor: FloatND) -> FloatND:
        return 2.0 * floor

    def borrowing(*, savings: FloatND, floor: FloatND) -> BoolND:
        return savings >= floor

    evaluate = cast(
        "q_and_f_module._BoundUtilitiesAndF",
        builder(
            functions=MappingProxyType(
                {
                    "floor": cast("EconFunction", floor),
                    "utility": cast("EconFunction", utility),
                }
            ),
            constraints=MappingProxyType(
                {"borrowing": cast("ConstraintFunction", borrowing)}
            ),
            bound_specs=(_bound(),),
        ),
    )
    actual = evaluate.evaluated(
        income=jnp.asarray(floor_value),
        floor_scale=jnp.asarray(1.0),
        savings=jnp.asarray(0.0),
    )
    np.testing.assert_array_equal(
        jnp.stack(actual.utilities_and_F),
        [2.0 * floor_value, False],
    )
    np.testing.assert_array_equal(
        jnp.stack(
            (
                actual.bound_status[0].nonfinite_floor,
                actual.bound_status[0].invalid_shifted_grid,
            )
        ),
        [not np.isfinite(floor_value), False],
    )
    assert calls == ["floor"]


@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_terminal_action_evaluation_keeps_bound_status_separate_from_F(
    *, collective: bool, invalid: bool
) -> None:
    """Singleton and household terminal evaluators carry actual RHS validity."""
    builder = (
        q_and_f_module.get_Q_and_F_terminal_collective
        if collective
        else q_and_f_module.get_Q_and_F_terminal
    )
    builder_code = inspect.unwrap(builder).__code__
    assert (
        "bound_specs"
        in builder_code.co_varnames[
            : builder_code.co_argcount + builder_code.co_kwonlyargcount
        ]
    ), "Terminal action evaluation must expose mandatory bound validity."

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    functions: dict[FunctionName, EconFunction] = {
        "floor": cast("EconFunction", floor),
        "utility_a" if collective else "utility": cast("EconFunction", utility),
    }
    kernel = cast(
        "q_and_f_module._TerminalQAndF | q_and_f_module._TerminalCollectiveQAndF",
        builder(
            flat_param_names=frozenset(),
            functions=MappingProxyType(functions),
            constraints=MappingProxyType(
                {"borrowing": cast("ConstraintFunction", borrowing)}
            ),
            bound_specs=(_bound(),),
            **({"stakeholders": ("a",)} if collective else {}),
        ),
    )
    evaluated = jax.jit(kernel.evaluated)(
        next_regime_to_V_arr=MappingProxyType({}),
        income=jnp.asarray(jnp.nan if invalid else 0.0),
        consumption=jnp.asarray(1.0),
    )
    np.testing.assert_array_equal(evaluated.parts.Q, [1.0] if collective else 1.0)
    np.testing.assert_array_equal(evaluated.parts.F, desired=False)
    np.testing.assert_array_equal(evaluated.parts.CE, [0.0] if collective else 0.0)
    np.testing.assert_array_equal(
        jnp.stack(
            (
                evaluated.bound_status[0].nonfinite_floor,
                evaluated.bound_status[0].invalid_shifted_grid,
            )
        ),
        [invalid, False],
    )


@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_nonterminal_action_evaluation_shares_flow_before_continuation(
    *, collective: bool, invalid: bool
) -> None:
    """Continuation aggregation keeps actual bound integrity beside economic F."""

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, consumption: FloatND, floor: FloatND) -> BoolND:
        return -consumption >= floor

    def continuation(
        **_kwargs: QAndFArg,
    ) -> tuple[FloatND, MappingProxyType[RegimeName, FloatND]]:
        return jnp.asarray(1.0), MappingProxyType({})

    def W(*, utility: FloatND, CE: FloatND) -> FloatND:
        return utility + CE

    def build_W_kwargs(_kwargs: QAndFKwargs) -> dict[ReferenceName, EconFunctionArg]:
        return {}

    functions = MappingProxyType(
        {
            "floor": cast("EconFunction", floor),
            "utility_a" if collective else "utility": cast("EconFunction", utility),
        }
    )
    flow = q_and_f_module._get_U_and_F(
        functions=functions,
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        utility_names=("utility_a",) if collective else ("utility",),
        bound_specs=(_bound(),),
    )
    if collective:
        empty = MappingProxyType({})
        machinery = q_and_f_module._ValueConstraintMachinery(
            reference_readers=empty,
            reference_reader_args=empty,
            evaluators=empty,
            evaluator_args=empty,
            q_value_index=empty,
            engine_supplied_names=frozenset(),
        )
        kernel = q_and_f_module._CollectiveQAndF(
            utilities_and_F=flow,
            value_constraint_machinery=machinery,
            compute_CE=continuation,
            koopmans_aggregator=cast("EconFunction", W),
            build_W_kwargs=build_W_kwargs,
            arg_names=("income", "consumption"),
        )
    else:
        kernel = q_and_f_module._QAndF(
            U_and_F=flow,
            continuation_reads=frozenset(),
            selection_aggregator=None,
            build_selection_kwargs=None,
            compute_CE=continuation,
            koopmans_aggregator=cast("EconFunction", W),
            build_W_kwargs=build_W_kwargs,
            arg_names=("income", "consumption"),
        )
    assert hasattr(kernel, "evaluated"), (
        "Actual action evaluation needs bound transport."
    )
    evaluated = jax.jit(kernel.evaluated)(
        next_regime_to_V_arr=MappingProxyType({}),
        income=jnp.asarray(jnp.nan if invalid else 0.0),
        consumption=jnp.asarray(1.0),
    )
    np.testing.assert_array_equal(evaluated.parts.Q, [2.0] if collective else 2.0)
    np.testing.assert_array_equal(evaluated.parts.F, desired=False)
    np.testing.assert_array_equal(evaluated.parts.CE, 1.0)
    np.testing.assert_array_equal(
        jnp.stack(
            (
                evaluated.bound_status[0].nonfinite_floor,
                evaluated.bound_status[0].invalid_shifted_grid,
            )
        ),
        [invalid, False],
    )


@pytest.mark.parametrize("evaluated", [False, True])
def test_private_flow_transport_does_not_shadow_user_inputs(*, evaluated: bool) -> None:
    """A user input keeps its declared name through the actual action evaluation."""

    def floor(*, income: FloatND) -> FloatND:
        return income

    def utility(*, utilities_and_F: FloatND) -> FloatND:
        return utilities_and_F

    def borrowing(*, floor: FloatND) -> BoolND:
        return floor <= 0.0

    def continuation(
        **_kwargs: QAndFArg,
    ) -> tuple[FloatND, MappingProxyType[RegimeName, FloatND]]:
        return jnp.asarray(1.0), MappingProxyType({})

    def W(*, utility: FloatND, CE: FloatND) -> FloatND:
        return utility + CE

    def build_W_kwargs(_kwargs: QAndFKwargs) -> dict[ReferenceName, EconFunctionArg]:
        return {}

    flow = q_and_f_module._get_U_and_F(
        functions=MappingProxyType(
            {
                "floor": cast("EconFunction", floor),
                "utility": cast("EconFunction", utility),
            }
        ),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(_bound(),),
    )
    kernel = q_and_f_module._QAndF(
        U_and_F=flow,
        selection_aggregator=None,
        build_selection_kwargs=None,
        compute_CE=continuation,
        koopmans_aggregator=cast("EconFunction", W),
        build_W_kwargs=build_W_kwargs,
        arg_names=("income", "utilities_and_F"),
        continuation_reads=frozenset(),
    )
    result = jax.jit(kernel.evaluated if evaluated else kernel)(
        next_regime_to_V_arr=MappingProxyType({}),
        income=jnp.asarray(0.0),
        utilities_and_F=jnp.asarray(3.0),
    )
    value, feasible = (result.parts.Q, result.parts.F) if evaluated else result
    np.testing.assert_array_equal(value, 4.0)
    np.testing.assert_array_equal(feasible, desired=True)


@pytest.mark.parametrize("field", ["endog_grid", "taste_shock_scale", "breakpoints"])
def test_feature_templates_require_canonical_coordinate_and_auxiliary_dtypes(
    *, field: str
) -> None:
    """
    Query metadata cannot weaken the numeric contract of coordinate or auxiliary
    rows.
    """
    spec = _compact_spec()
    template = build_template_egm_carry(
        n_rows=5,
        leading_shape=(2, 3),
        query_data=build_template_egm_query_data(
            state_shape=(2,),
            action_shape=(3,),
            row_capacity=5,
            interval_capacity=1,
        ),
    )
    if field == "breakpoints":
        template = replace(template, breakpoints=jnp.zeros((2, 3, 2)))
    tampered = jax.tree.map_with_path(
        lambda path, leaf: (
            leaf.astype(jnp.int32)
            if tuple(key.name for key in path) == (field,)
            else leaf
        ),
        template,
    )
    with pytest.raises(TypeError, match=r"Query|query"):
        continuation.EGMContinuationSpec(
            template=tampered,
            layout=continuation.EGMContinuationLayout(query_spec=spec),
        )


def test_unbound_feature_reader_cannot_satisfy_continuation_capability_demand() -> None:
    """The continuation admission seam requires an actual model-bound query reader."""
    template = build_template_egm_carry(
        n_rows=5,
        query_data=build_template_egm_query_data(
            state_shape=(),
            action_shape=(),
            row_capacity=5,
            interval_capacity=1,
        ),
    )
    missing = _missing_capabilities(
        demanded=ContinuationCapabilities(
            value=True,
            marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE}),
        ),
        offered=template.capabilities,
    )
    assert missing == ("value", f"marginal in '{EGM_ENDOGENOUS_COORDINATE}'")


def test_literal_floors_keep_the_feature_free_utility_feasibility_path() -> None:
    """Finite static bounds retain ordinary outputs without runtime summary records."""
    bound = replace(
        _bound(),
        kind="literal",
        literal_value=0.0,
        lower_target=None,
        lower_inputs=(),
        dependency_identity=(((), 0.0),),
    )

    def utility(*, consumption: FloatND) -> FloatND:
        return consumption

    def borrowing(*, savings: FloatND) -> BoolND:
        return savings >= 0.0

    evaluate = q_and_f_module._get_U_and_F(
        functions=MappingProxyType({"utility": cast("EconFunction", utility)}),
        constraints=MappingProxyType(
            {"borrowing": cast("ConstraintFunction", borrowing)}
        ),
        bound_specs=(bound,),
    )
    assert not hasattr(evaluate, "evaluated"), (
        "Literal models need no runtime summary work."
    )
    values = jax.jit(evaluate)(consumption=jnp.asarray(2.0), savings=jnp.asarray(0.0))
    np.testing.assert_array_equal(jnp.stack(values), [2.0, True])
