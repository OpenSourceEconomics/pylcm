"""Finalize user regimes at model build.

`finalize_regimes` turns each user `Regime` into the complete form the model
runs: model-level `derived_categoricals` are merged in, the model-level
Koopmans aggregator and certainty equivalent are injected into non-terminal
regimes that declare none, and completeness is validated (a
`utility` entry, state-transition coverage, no state/action overlap,
distributed-grid rules). The result is a plain
`lcm.regime.Regime`, still in user vocabulary — coarse laws, `Phased`
containers, and per-target dicts survive untouched, so the params template
reads the user's coarseness off it.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from dags import get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.grids import DiscreteGrid
from _lcm.typing import FunctionName, RegimeName
from _lcm.user_regime_validation import _validate_completeness
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.solvers import NEGM
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
    _fail_if_continuation_slot_is_mixed(user_regimes, "koopmans_aggregator")
    _fail_if_continuation_slot_is_mixed(user_regimes, "certainty_equivalent")
    result: dict[RegimeName, FinalizedUserRegime] = {}
    for regime_name, user_regime in user_regimes.items():
        merged = _merge_derived_categoricals(
            regime_name=regime_name,
            user_regime=user_regime,
            derived_categoricals=derived_categoricals,
        )
        functions = dict(user_regime.functions)
        _compose_negm_resources(
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
        error_messages = _validate_completeness(finalized)
        if error_messages:
            raise RegimeInitializationError(
                f"In regime '{regime_name}': {format_messages(error_messages)}"
            )
        result[regime_name] = finalized
    return MappingProxyType(result)


def _compose_negm_resources(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction | Phased | None],
) -> None:
    """Inject `resources = base - outer_cost` for an NEGM regime with a cost.

    The NEGM stacked-carry lift is exact only when the inner resources are
    affine in the declared outer cost with coefficient exactly `-1`. That is
    not a property finitely many probes can certify on an arbitrary user
    function, so pylcm performs the subtraction itself: the user supplies the
    cost-free base `<resources>_before_outer_cost`, and the resources function
    every consumer sees — the inner EGM solve, the budget constraint, the
    parent's child-resources read — is composed here, at the single point
    before validation and processing. Mutates `functions` in place; regimes
    whose solver is not `NEGM`, or whose `outer_cost` is `None`, are untouched
    (they define the resources function directly).

    Raises:
        ModelInitializationError: If the regime defines the resources function
            itself, or the cost-free base or the declared cost is missing.

    """
    solver = user_regime.solver
    if not isinstance(solver, NEGM) or solver.outer_cost is None:
        return
    resources_name = solver.inner.resources
    base_name = f"{resources_name}_before_outer_cost"
    cost_name = solver.outer_cost

    if resources_name in functions:
        msg = (
            f"Regime '{regime_name}' defines the resources function "
            f"'{resources_name}' alongside a declared `NEGM.outer_cost` "
            f"('{cost_name}'). With a declared outer cost the resources "
            f"function is composed by pylcm as `{base_name} - {cost_name}`, "
            "so its affine use of the cost holds by construction — define "
            f"the cost-free base '{base_name}' instead of '{resources_name}'."
        )
        raise ModelInitializationError(msg)
    if base_name not in functions:
        msg = (
            f"Regime '{regime_name}' declares `NEGM.outer_cost` "
            f"('{cost_name}') but no cost-free resources base "
            f"'{base_name}'. pylcm composes the resources function as "
            f"`{base_name} - {cost_name}` — declare the base function."
        )
        raise ModelInitializationError(msg)
    if cost_name not in functions:
        msg = (
            f"NEGM.outer_cost '{cost_name}' is not a declared function of "
            f"regime '{regime_name}'. The credited outer cost must be a "
            "regime function reading only the durable state, the outer "
            "post-decision, and params."
        )
        raise ModelInitializationError(msg)

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
