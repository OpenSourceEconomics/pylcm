"""Finalize user regimes at model build.

`finalize_regimes` turns each user `Regime` into the complete form the model
runs: model-level `derived_categoricals` are merged in, the model-level
Koopmans aggregator and certainty equivalent are injected into non-terminal
regimes that declare none, and completeness is validated (a
`utility` entry — a per-stakeholder `utility_<s>` and at least one discrete
action for a collective regime — state-transition coverage, no state/action
overlap, distributed-grid rules). The result is a plain
`lcm.regime.Regime`, still in user vocabulary — coarse laws, `Phased`
containers, and per-target dicts survive untouched, so the params template
reads the user's coarseness off it.
"""

import functools
import inspect
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
from dags import get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.constraints.ir import Condition
from _lcm.grids import DiscreteGrid
from _lcm.typing import FunctionName, RegimeName
from _lcm.user_regime_validation import (
    _fail_if_collective_regime_folds,
    _validate_completeness,
)
from _lcm.utils.error_messages import format_messages
from lcm.case_piece import CaseBoundary
from lcm.consumption_savings_regime import (
    NetOfAdjustmentCost,
    _composition_rule_message,
    _EGMFamilyRegime,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import _AgeSpecialized
from lcm.typing import FloatND, UserFunction

# A user `Regime` after model-build finalization. Runtime-equivalent to
# `lcm.regime.Regime`; internal signatures use this alias to mark values
# produced by `finalize_regimes` (model-level slots merged, completeness
# validated).
type FinalizedUserRegime = UserRegime


def finalize_regimes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
    koopmans_aggregator: UserFunction,
    certainty_equivalent: CertaintyEquivalent,
) -> MappingProxyType[RegimeName, FinalizedUserRegime]:
    """Finalize every user regime for the model build.

    Merges model-level `derived_categoricals` into each regime (a regime
    entry with identical categories is tolerated; conflicting categories
    raise), injects the model-level Koopmans aggregator and certainty
    equivalent into non-terminal regimes that declare none, and validates
    completeness.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.
        derived_categoricals: Model-level categorical grids to broadcast.
        koopmans_aggregator: Model-level Bellman aggregator, given to every
            non-terminal regime that declares none of its own.
        certainty_equivalent: Model-level continuation aggregation, given to
            every non-terminal regime that declares none of its own.

    Returns:
        Immutable mapping of regime names to finalized regimes.

    Raises:
        ModelInitializationError: If a regime has a `derived_categoricals`
            entry conflicting with a model-level one.
        RegimeInitializationError: If a regime is incomplete (e.g. missing
            `utility` or state-transition coverage), with the regime name
            prefixed.

    """
    _fail_if_collective_regime_folds(user_regimes=user_regimes)
    _fail_if_continuation_slot_is_mixed(user_regimes, "koopmans_aggregator")
    _fail_if_continuation_slot_is_mixed(user_regimes, "certainty_equivalent")
    # The published frame is one table over every regime, so the names its
    # collective regimes claim are reserved for all of them.
    reserved_value_columns = frozenset(
        f"value_{stakeholder}"
        for regime in user_regimes.values()
        if regime.stakeholders is not None
        for stakeholder in regime.stakeholders
    )
    result: dict[RegimeName, FinalizedUserRegime] = {}
    for regime_name, user_regime in user_regimes.items():
        merged = _merge_derived_categoricals(
            regime_name=regime_name,
            user_regime=user_regime,
            derived_categoricals=derived_categoricals,
        )
        functions = dict(user_regime.decomposed_functions)
        _compose_case_piece_outputs(functions=functions)
        _compose_margin_resources(
            regime_name=regime_name,
            user_regime=user_regime,
            functions=functions,
        )
        # Terminal regimes have no continuation, so they need neither an
        # aggregator (Q = U directly) nor a certainty equivalent. Their slots
        # are carried through untouched so that a declared one still reaches
        # the completeness check below rather than being silently discarded.
        if user_regime.terminal:
            regime_koopmans_aggregator = user_regime.koopmans_aggregator
            regime_certainty_equivalent = user_regime.certainty_equivalent
        else:
            regime_koopmans_aggregator = (
                user_regime.koopmans_aggregator
                if user_regime.koopmans_aggregator is not None
                else koopmans_aggregator
            )
            regime_certainty_equivalent = (
                user_regime.certainty_equivalent
                if user_regime.certainty_equivalent is not None
                else certainty_equivalent
            )
        finalized = user_regime.replace(
            derived_categoricals=merged,
            functions=MappingProxyType(functions),
            koopmans_aggregator=regime_koopmans_aggregator,
            certainty_equivalent=regime_certainty_equivalent,
        )
        error_messages = _validate_completeness(
            regime=finalized, reserved_value_columns=reserved_value_columns
        )
        if error_messages:
            raise RegimeInitializationError(
                f"In regime '{regime_name}': {format_messages(error_messages)}"
            )
        finalized._validate_finalized_structure(regime_name=regime_name)  # noqa: SLF001
        result[regime_name] = finalized
    return MappingProxyType(result)


def _compose_case_piece_outputs(
    *, functions: dict[FunctionName, UserFunction | Phased | None]
) -> None:
    """Build each split output declared by a complete pair of case pieces."""
    from _lcm.egm.nbegm import PieceSet, collect_nbegm_metadata  # noqa: PLC0415

    def build(piece_set: PieceSet) -> UserFunction:
        producer_names = (
            piece_set.predicate_name,
            piece_set.when_func,
            piece_set.otherwise_func,
        )

        @with_signature(
            args={name: _return_annotation(functions[name]) for name in producer_names},
            return_annotation=_case_output_annotation(
                functions=functions,
                output=piece_set.output,
                branch_names=(piece_set.when_func, piece_set.otherwise_func),
            ),
        )
        def composed_case_output(**kwargs: FloatND) -> FloatND:
            return jnp.where(
                kwargs[piece_set.predicate_name],
                kwargs[piece_set.when_func],
                kwargs[piece_set.otherwise_func],
            )

        composed_case_output.__name__ = piece_set.output
        return cast("UserFunction", composed_case_output)

    registry = collect_nbegm_metadata(functions=functions)
    producer_names = dict.fromkeys(
        name
        for piece_set in registry.piece_sets
        for name in (
            piece_set.predicate_name,
            piece_set.when_func,
            piece_set.otherwise_func,
        )
    )
    for name in producer_names:
        function = functions[name]
        if callable(function) and not isinstance(function, _AgeSpecialized):
            annotated = _with_inferred_case_annotations(
                function=cast("UserFunction", function), functions=functions
            )
            if isinstance(function, CaseBoundary):
                annotated.__lcm_case_boundary__ = function  # ty: ignore[unresolved-attribute]
            functions[name] = annotated
    for piece_set in registry.piece_sets:
        functions[piece_set.output] = build(piece_set)


def _with_inferred_case_annotations(
    *,
    function: UserFunction,
    functions: Mapping[FunctionName, UserFunction | Phased | None],
) -> UserFunction:
    """Fill only missing DAG annotations on an internal case-function proxy."""
    annotations = ensure_annotations_are_strings(get_annotations(function))
    argument_annotations = {}
    for name in inspect.signature(function).parameters:
        inferred = _annotation_for_argument(functions=functions, name=name)
        declared = annotations.get(name)
        argument_annotations[name] = (
            inferred
            if isinstance(function, Condition) and inferred != "no_annotation_found"
            else declared
            if declared not in {None, "no_annotation_found"}
            else inferred
        )

    @with_signature(
        args=argument_annotations,
        return_annotation=annotations.get("return", "no_annotation_found"),
    )
    @functools.wraps(function)
    def annotated_case_function(**kwargs: object) -> object:
        return function(**kwargs)

    return cast("UserFunction", annotated_case_function)


def _annotation_for_argument(
    *,
    functions: Mapping[FunctionName, UserFunction | Phased | None],
    name: str,
) -> str:
    """Return a concrete annotation used for a generated function argument."""
    for func in functions.values():
        if not callable(func) or isinstance(func, (Condition, _AgeSpecialized)):
            continue
        annotation = ensure_annotations_are_strings(get_annotations(func)).get(name)
        if annotation not in {None, "no_annotation_found"}:
            return annotation
    return "no_annotation_found"


def _case_output_annotation(
    *,
    functions: Mapping[FunctionName, UserFunction | Phased | None],
    output: str,
    branch_names: tuple[str, str],
) -> str:
    """Return the declared output annotation from a consumer or piece branch."""
    consumer_annotation = _annotation_for_argument(functions=functions, name=output)
    if consumer_annotation != "no_annotation_found":
        return consumer_annotation
    for name in branch_names:
        annotation = _return_annotation(functions[name])
        if annotation != "no_annotation_found":
            return annotation
    return "no_annotation_found"


def _compose_margin_resources(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction | Phased | None],
) -> None:
    """Compose a ``NetOfAdjustmentCost`` resources declaration.

    The regime owns the three names explicitly.  pylcm performs the subtraction
    at the single model-finalization seam, before validation and DAG processing,
    so the coefficient on the cost is exactly ``-1`` by construction rather than
    inferred from probes.  Bare resources declarations are untouched.

    Raises:
        ModelInitializationError: If the regime defines the resources function
            itself, or the cost-free base or the declared cost is missing.

    """
    if not isinstance(user_regime, _EGMFamilyRegime):
        return
    resources = user_regime.liquid.resources
    if not isinstance(resources, NetOfAdjustmentCost):
        return

    resources_name = resources.output
    base_name = resources.before_cost
    cost_name = resources.cost

    if resources_name in functions:
        raise ModelInitializationError(
            _composition_rule_message(
                resources=resources,
                prefix=(
                    f"Regime {regime_name!r} defines the composed resources "
                    f"function {resources_name!r}. "
                ),
            )
        )
    if base_name not in functions:
        raise ModelInitializationError(
            _composition_rule_message(
                resources=resources,
                prefix=(
                    f"Regime {regime_name!r} is missing the cost-free resources "
                    f"function {base_name!r}. "
                ),
            )
        )
    if cost_name not in functions:
        raise ModelInitializationError(
            _composition_rule_message(
                resources=resources,
                prefix=(
                    f"Regime {regime_name!r} is missing the adjustment-cost "
                    f"function {cost_name!r}. "
                ),
            )
        )

    base_annotation = _return_annotation(functions[base_name])
    cost_annotation = _return_annotation(functions[cost_name])

    @with_signature(
        args={base_name: base_annotation, cost_name: cost_annotation},
        return_annotation=base_annotation,
    )
    def composed_resources(**kwargs: FloatND) -> FloatND:
        return kwargs[base_name] - kwargs[cost_name]

    composed_resources.__name__ = resources_name
    functions[resources_name] = cast("UserFunction", composed_resources)


def _return_annotation(func: UserFunction | Phased | None) -> str:
    """Return a function's stringified return annotation, defaulting to `FloatND`.

    The composed resources function copies its producers' annotations so the
    DAG's annotation-consistency check stays satisfied.

    An `AgeSpecializedFunction` carries no annotation of its own — the annotation
    belongs to the concrete functions `build(age)` returns, which this stage has no
    age to ask for. The default stands in, and the DAG's own annotation check still
    compares it against the resolved function once the age is known, so a genuine
    mismatch is caught there rather than passed through.
    """
    if not callable(func) or isinstance(func, _AgeSpecialized):
        return "FloatND"
    annotations = ensure_annotations_are_strings(get_annotations(func))
    return annotations.get("return", "FloatND")


def _merge_derived_categoricals(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> dict[FunctionName, DiscreteGrid]:
    """Merge model-level derived categoricals into one regime's mapping.

    Follows the exactly-one-level rule of the other model-level regime
    slots: a name is defined at model level or regime level, never both.
    """
    merged = dict(user_regime.derived_categoricals)
    for var, grid in derived_categoricals.items():
        if var in merged:
            msg = (
                f"Ambiguous specification for derived_categoricals['{var}'] "
                f"in regime '{regime_name}': defined at model level and "
                f"regime level. Remove one."
            )
            raise ModelInitializationError(msg)
        merged[var] = grid
    return merged


def _fail_if_continuation_slot_is_mixed(
    user_regimes: Mapping[RegimeName, UserRegime],
    slot: str,
) -> None:
    """Reject a model that declares a continuation slot at both levels.

    `koopmans_aggregator` and `certainty_equivalent` are declared once for
    the model, or once in every regime that has a continuation — never some
    of each. A partial declaration reads as if the silent regimes had opted
    out of something they never saw, when in fact they are still taking the
    model-level value.
    """
    with_continuation = {
        name for name, regime in user_regimes.items() if not regime.terminal
    }
    declaring = {
        name
        for name in with_continuation
        if getattr(user_regimes[name], slot) is not None
    }
    silent = with_continuation - declaring
    if declaring and silent:
        msg = (
            f"Ambiguous specification for `{slot}`: declared in regime(s) "
            f"{sorted(declaring)} but not in {sorted(silent)}, which take the "
            f"model-level value. Declare it once on the `Model`, or in every "
            f"regime that has a continuation."
        )
        raise ModelInitializationError(msg)
