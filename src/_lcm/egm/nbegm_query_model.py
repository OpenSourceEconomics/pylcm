"""Canonical liquid-coordinate authority for conditional NB continuation reads."""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field

from _lcm.continuation import EGMQuerySpec
from _lcm.typing import (
    EconFunction,
    EconFunctionKwargs,
    EconFunctionsMapping,
    QualifiedName,
)
from lcm.typing import (
    BoolND,
    Float1D,
    FunctionName,
    ReferenceName,
    StateName,
    UserFunction,
)


@dataclass(frozen=True, kw_only=True)
class NBEGMIntervalRecipe:
    """Reconstruct declaration-ordered economic sides from canonical preimage DAGs."""

    source_identity: tuple[Hashable, ...]
    """Canonical declarations, resolved DAGs, equality owners and source order."""
    input_names: frozenset[ReferenceName]
    """External fixed-state/time/qualified-parameter leaves of the source DAGs."""
    equal_above: tuple[bool, ...]
    """Whether the exact source threshold belongs to its above side."""
    evaluate_preimages: Callable[[EconFunctionKwargs], tuple[Float1D, BoolND]]
    """Return canonical selection preimages and increasing-variable flags."""
    evaluate_sides: Callable[[EconFunctionKwargs], BoolND]
    """Evaluate declaration-ordered source comparisons at the actual liquid state."""


@dataclass(frozen=True, kw_only=True)
class NBEGMQueryModel:
    """Independent NB evaluators and interval identity prepared before kernel build."""

    authority: EGMQuerySpec
    """Canonical liquid rows, budget-run topology and conditional action axes."""
    functions: EconFunctionsMapping
    """Canonical processed functions, before target-period resolution."""
    koopmans_aggregator: EconFunction
    """The target's processed period aggregator."""
    offsets: Float1D
    """One canonical source saving-offset vector."""
    intervals: NBEGMIntervalRecipe
    """Independently reconstructed economic interval identity."""
    intervals_by_period: Mapping[int, NBEGMIntervalRecipe]
    """Canonical recipes resolved at each actual target period."""
    passive_state_names: tuple[StateName, ...]
    """Declared passive axes remaining after discrete-state selection."""
    literal_nodes: Float1D | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Original physical savings nodes on a literal predicate feature route."""
    evaluated_predicate_names: tuple[FunctionName, ...] = field(
        default=(), metadata={"fingerprint_omit_if_default": True}
    )
    """Canonical qualified state-node declarations deferred by the compiler."""
    inverse_eis_name: QualifiedName | None = None
    """Existing EZ elasticity leaf; absent for additive expected utility."""
    finite_inner_models: Mapping[str, NBEGMQueryModel] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Finite keeper/adjuster authorities with their actual substituted DAGs."""
    retain_nested_feasibility: bool = field(
        default=False, metadata={"fingerprint_omit_if_default": True}
    )
    """Retain each actual D feasibility row for the enclosing outer fold."""
    functions_by_period: Mapping[int, EconFunctionsMapping] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Actual period-resolved derived role pools, absent on ordinary NB models."""


def derive_finite_inner_models(
    *,
    model: NBEGMQueryModel,
    functions_by_period: Mapping[int, EconFunctionsMapping],
    declarations_by_period: Mapping[int, Mapping[FunctionName, UserFunction]],
) -> Mapping[str, NBEGMQueryModel]:
    """Build finite roles from the actual period-specific substituted DAGs."""
    if model.authority.finite_outer is None:
        raise TypeError("Derived finite roles require their canonical descriptor.")
    return _derive_inner_models(
        model=model,
        functions_by_period=functions_by_period,
        declarations_by_period=declarations_by_period,
    )


def _derive_inner_models(
    *,
    model: NBEGMQueryModel,
    functions_by_period: Mapping[int, EconFunctionsMapping],
    declarations_by_period: Mapping[int, Mapping[FunctionName, UserFunction]],
) -> Mapping[str, NBEGMQueryModel]:
    """Preserve each nested role's real function pool and interval ownership."""
    from dataclasses import replace  # noqa: PLC0415
    from types import MappingProxyType  # noqa: PLC0415

    from _lcm.continuation import build_query_model_authority  # noqa: PLC0415
    from _lcm.solution.nbegm import build_nbegm_interval_recipe  # noqa: PLC0415
    from _lcm.solution.negm import (  # noqa: PLC0415
        _with_no_adjustment_outer_function,
        _without_outer_post_decision,
    )

    outer = model.authority.finite_outer
    if outer is None:
        raise TypeError("Derived nested roles require their canonical descriptor.")
    result = {}
    for role in ("keeper", "adjuster"):
        pools = {}
        for period, functions in functions_by_period.items():
            if role == "adjuster":
                pools[period] = _without_outer_post_decision(
                    functions=functions,
                    outer_post_decision=outer.outer_post_decision,
                )
            else:
                pools[period] = _with_no_adjustment_outer_function(
                    functions=functions,
                    durable_state=outer.outer_state,
                    outer_post_decision=outer.outer_post_decision,
                    no_adjustment_func=None
                    if outer.keeper_target is None
                    else functions[outer.keeper_target],
                )
        periods = model.authority.target_periods
        first = pools[periods[0]]
        spec = model.authority
        authority = build_query_model_authority(
            bound=spec.bound,
            state_names=spec.state_names,
            action_names=spec.action_names,
            coordinate_kind=spec.coordinate_kind,
            marginal_convention=spec.marginal_convention,
            corner_coordinate_kind=spec.corner_coordinate_kind,
            row_capacity=spec.row_capacity,
            interval_ids=spec.interval_ids,
            interval_side_ids=spec.interval_side_ids,
            evaluator_targets=spec.evaluator_targets,
            functions=first,
            functions_by_period=pools,
            target_periods=periods,
            row_topology=spec.row_topology,
            interval_source_identity=spec.interval_source_identity,
        )
        recipes = MappingProxyType(
            {
                period: build_nbegm_interval_recipe(
                    declarations=declarations_by_period[period],
                    functions=pools[period],
                    liquid_name=spec.bound.liquid_state,
                    state_names=(*spec.state_names, spec.bound.liquid_state),
                    action_names=spec.action_names,
                )
                for period in periods
            }
        )
        result[role] = replace(
            model,
            authority=authority,
            functions=first,
            koopmans_aggregator=first["koopmans_aggregator"],
            intervals=recipes[periods[0]],
            intervals_by_period=recipes,
            finite_inner_models=None,
            functions_by_period=MappingProxyType(pools),
            retain_nested_feasibility=True,
        )
    return MappingProxyType(result)
