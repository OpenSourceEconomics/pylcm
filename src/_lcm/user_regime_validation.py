"""Validation helpers for the user-facing `Regime`.

These functions back `Regime.__post_init__`. They raise nothing themselves —
instead they collect error messages, which `__post_init__` aggregates into a
single `RegimeInitializationError`. Splitting the validators out of the
public module keeps `lcm.regime` to class definitions.

"""

import ast
import inspect
import math
import textwrap
from collections import Counter
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from dags.tree import QNAME_DELIMITER

from _lcm.grids import DiscreteGrid, Grid
from _lcm.identity_transition import _IdentityTransition
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.processes.iid import _IIDProcess
from _lcm.typing import ActiveFunction, ProcessName, RegimeName, StateName
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.solvers import DCEGM, GridSearch
from lcm.transition import MarkovTransition

if TYPE_CHECKING:
    import lcm.regime


def _grid_mapping_errors(
    *, attr_name: str, mapping: Mapping[str, object], allow_phase_variants: bool
) -> list[str]:
    """Collect key/value type errors for a grid-valued mapping (states/actions)."""
    allowed = Grid | Phased if allow_phase_variants else Grid
    suffix = " or Phased" if allow_phase_variants else ""
    error_messages: list[str] = []
    for k, v in mapping.items():
        if not isinstance(k, str):
            error_messages.append(f"{attr_name} key {k!r} must be a string.")
        if v is None:
            # A mask of a model-level entry; bound at model build.
            continue
        if not allow_phase_variants and isinstance(v, Phased):
            error_messages.append(
                f"{attr_name}['{k}'] cannot be phase-variant: the simulated "
                f"argmax must range over the same menu the value function was "
                f"computed for."
            )
        elif not isinstance(v, allowed):
            error_messages.append(
                f"{attr_name} value {v!r} must be an LCM grid{suffix}."
            )
    return error_messages


def _callable_mapping_errors(
    *, attr_name: str, mapping: Mapping[str, object], allow_phase_variants: bool
) -> list[str]:
    """Collect key/value type errors for a callable-valued mapping."""
    error_messages: list[str] = []
    for k, v in mapping.items():
        if not isinstance(k, str):
            error_messages.append(f"{attr_name} key {k!r} must be a string.")
        if v is None:
            # A mask of a model-level entry; bound at model build.
            continue
        if isinstance(v, Phased):
            if not allow_phase_variants:
                error_messages.append(
                    f"{attr_name}['{k}'] cannot be phase-variant: a "
                    f"phase-specific feasible set would let the simulated "
                    f"argmax range over actions the value function was never "
                    f"computed for."
                )
        elif not callable(v):
            error_messages.append(f"{attr_name} value {v!r} must be a callable.")
    return error_messages


def _validate_collective_regime(regime: lcm.regime.Regime) -> None:
    """Validate a collective (stakeholder-valued) regime — E1.

    Both the terminal (slice 1b) and non-terminal continuation (slice 2) cases
    are implemented; a regime is checked for the invariants the collective
    kernels rely on: a non-empty, duplicate-free `stakeholders` tuple, a
    per-stakeholder `utility_<s>` function for every stakeholder, at least one
    discrete action (the household argmax runs over the discrete-action
    product), and — if `weights` is given — weight keys matching the
    stakeholder names, with every weight finite, non-negative, and a positive
    total (the zero-safe scalarization in `collective._weighted_sum` treats an
    exact-zero weight as a deliberate exclusion, so a non-finite or negative
    weight, or an all-zero `weights` mapping, would silently produce a
    meaningless or undefined household objective rather than erroring).

    Features outside the E1 scope raise `NotImplementedError`: EV1 taste shocks
    (the collective argmax is the hard household maximum), a nonlinear
    certainty equivalent (the per-stakeholder continuation is the linear
    expectation), and any solver other than `GridSearch`.

    Called from `Regime.__post_init__` only when `stakeholders is not None`, so
    the default singleton path never reaches it.
    """
    stakeholders = cast("tuple[str, ...]", regime.stakeholders)

    if regime.taste_shocks is not None:
        raise NotImplementedError(
            "EV1 taste shocks on a collective (stakeholder-valued) regime are "
            "not yet implemented: the collective solve takes the hard household "
            "argmax of the Pareto-weighted objective, not a smoothed maximum. "
            "See the design doc `pylcm-extension-collective-regimes.md` (v2.1)."
        )
    if regime.certainty_equivalent is not None:
        raise NotImplementedError(
            "A nonlinear certainty equivalent on a collective "
            "(stakeholder-valued) regime is not yet implemented: the "
            "per-stakeholder continuation is the linear expectation "
            "E[V'^s]. See the design doc "
            "`pylcm-extension-collective-regimes.md` (v2.1)."
        )
    if not isinstance(regime.solver, GridSearch):
        raise NotImplementedError(
            "Collective (stakeholder-valued) regimes are only implemented for "
            "the GridSearch solver: the household argmax and per-stakeholder "
            "value readout run over the full action product. Use "
            "`solver=GridSearch()` for this regime."
        )
    if regime.terminal and (regime.value_constraints or regime.same_period_refs):
        raise NotImplementedError(
            "`value_constraints` / `same_period_refs` on a TERMINAL collective "
            "regime are not implemented: value-aware feasibility (E2) masks a "
            "within-period household decision; the terminal kernel carries no "
            "such mask. Declare them on the non-terminal regime whose decision "
            "they constrain. See the design doc "
            "`pylcm-extension-collective-regimes.md` (v2.1), §2 E2."
        )

    error_messages: list[str] = []
    error_messages.extend(_collective_value_constraint_errors(regime))
    error_messages.extend(_stakeholders_tuple_errors(stakeholders))

    missing = [s for s in stakeholders if f"utility_{s}" not in regime.functions]
    if missing:
        needed = ", ".join(f"'utility_{s}'" for s in stakeholders)
        raise RegimeInitializationError(
            f"A collective regime with stakeholders {stakeholders} must supply "
            f"a per-stakeholder utility for each stakeholder ({needed}) in its "
            f"functions. Missing: {[f'utility_{s}' for s in missing]}."
        )

    if not any(isinstance(grid, DiscreteGrid) for grid in regime.actions.values()):
        error_messages.append(
            "A collective regime must have at least one discrete action: the "
            "household argmax of the scalarization runs over the discrete-action "
            "product."
        )

    error_messages.extend(_collective_weights_errors(regime, stakeholders))

    if error_messages:
        raise RegimeInitializationError(format_messages(error_messages))


def _stakeholders_tuple_errors(stakeholders: tuple[str, ...]) -> list[str]:
    """Collect errors for an empty or duplicate-containing `stakeholders` tuple."""
    if not stakeholders:
        return ["`stakeholders` must be a non-empty tuple for a collective regime."]

    duplicate_counts = Counter(stakeholders)
    duplicates = sorted(name for name, count in duplicate_counts.items() if count > 1)
    if duplicates:
        return [
            f"`stakeholders` must not contain duplicate names; got "
            f"{stakeholders} (duplicated: {duplicates})."
        ]
    return []


def _collective_weights_errors(
    regime: lcm.regime.Regime, stakeholders: tuple[str, ...]
) -> list[str]:
    """Collect errors for a collective regime's `weights` mapping.

    Weight keys must match `stakeholders`, and every weight must be finite,
    non-negative, with a positive total — the zero-safe scalarization in
    `collective._weighted_sum` treats an exact-zero weight as a deliberate
    exclusion, so a non-finite or negative weight, or an all-zero `weights`
    mapping, would silently produce a meaningless or undefined household
    objective rather than erroring.
    """
    if regime.weights is None:
        return []

    if set(regime.weights) != set(stakeholders):
        return [
            f"`weights` keys {sorted(regime.weights)} must match the "
            f"stakeholders {sorted(stakeholders)}."
        ]

    non_finite_or_negative = {
        name: w for name, w in regime.weights.items() if not math.isfinite(w) or w < 0
    }
    if non_finite_or_negative:
        return [
            "`weights` must be finite and non-negative for every "
            f"stakeholder; got {non_finite_or_negative}."
        ]
    if sum(regime.weights.values()) <= 0:
        return [
            "`weights` must sum to a positive total; an all-zero "
            "Pareto-weight mapping leaves the household scalarization "
            "identically zero for every action, so the argmax is undefined."
        ]
    return []


def _collective_value_constraint_errors(regime: lcm.regime.Regime) -> list[str]:
    """Collect the regime-local errors of `value_constraints` / `same_period_refs`.

    COLLECTIVE-REGIMES (E2). Cross-regime properties — the reference regime's
    existence, its stakeholder layout, projection coverage, co-activity, and
    reference cycles — are validated at model processing, where the other
    regimes are known.
    """
    stakeholders = cast("tuple[str, ...]", regime.stakeholders)
    error_messages: list[str] = []

    if regime.same_period_refs and not regime.value_constraints:
        error_messages.append(
            "`same_period_refs` without `value_constraints` has no consumer: "
            "the reference values are only readable by value-constraint "
            "predicates. Declare the predicates, or drop the references."
        )

    invalid_names = [
        name
        for name in [*regime.value_constraints, *regime.same_period_refs]
        if QNAME_DELIMITER in name
    ]
    if invalid_names:
        error_messages.append(
            f"Value-constraint and reference-value names cannot contain the "
            f"reserved separator '{QNAME_DELIMITER}': {invalid_names}."
        )

    taken = (
        set(regime.functions)
        | set(regime.constraints)
        | set(regime.states)
        | set(regime.actions)
    )
    colliding = sorted(
        (set(regime.value_constraints) | set(regime.same_period_refs)) & taken
    )
    if colliding:
        error_messages.append(
            f"Value-constraint / reference-value name(s) {colliding} collide "
            "with a function, constraint, state, or action of the regime. "
            "Reference values enter the predicates as named arguments, so the "
            "names must be unambiguous."
        )

    if regime.value_constraints:
        shadowed_q = sorted({f"Q_{s}" for s in stakeholders} & taken)
        if shadowed_q:
            error_messages.append(
                f"Name(s) {shadowed_q} are reserved for the per-stakeholder "
                "action values that `value_constraints` predicates read; the "
                "regime declares a function, constraint, state, or action of "
                "the same name. Rename it."
            )

    return error_messages


def _validate_gated_edges(regime: lcm.regime.Regime) -> None:
    """Validate a regime's `gated_edges` declarations — E3' (regime-local part).

    COLLECTIVE-REGIMES (E3'). Checks the properties knowable without the other
    regimes: the gate is a plain boolean callable (a stochastic `kappa` gate —
    a `MarkovTransition` — is out of scope for this slice); every declared edge
    targets one of the regime's reachable transition targets; the legs cover the
    SOURCE's stakeholder structure (exactly one leg for a singleton source, one
    per stakeholder for a collective source). Cross-regime properties — the
    target and fallback regimes exist, the stakeholder names resolve, the
    projections cover the reference states — are validated at model processing.

    A gated-edge SOURCE is restricted to the `GridSearch` solver with no taste
    shocks or certainty equivalent: the source reads the folded ``Wbar`` through
    the grid-search continuation machinery, and edges touching DC-EGM /
    taste-shock / certainty-equivalent regimes are out of scope for this slice.

    Called from `Regime.__post_init__` only when `gated_edges` is non-empty, so
    the default path never reaches it.
    """
    _fail_if_gated_edge_source_out_of_scope(regime)

    error_messages: list[str] = []
    transition_targets = _regime_transition_target_names(regime.transition)
    source_stakeholders = regime.stakeholders

    for target_name, edge in regime.gated_edges.items():
        prefix = f"gated_edges['{target_name}']: "
        if isinstance(edge.gate, MarkovTransition):
            error_messages.append(
                f"{prefix}the gate must be a plain boolean function. A "
                "`MarkovTransition` (stochastic / probabilistic gate) is out of "
                "scope for this slice — E3' gates are boolean."
            )
        elif not callable(edge.gate):
            error_messages.append(f"{prefix}the gate must be a callable.")
        if transition_targets is not None and target_name not in transition_targets:
            error_messages.append(
                f"{prefix}a gated edge must target one of the regime's reachable "
                f"transition targets {sorted(transition_targets)}; declare the "
                f"transition into '{target_name}' as well."
            )
        if not edge.legs:
            error_messages.append(f"{prefix}must declare at least one leg.")
        elif source_stakeholders is None:
            if len(edge.legs) != 1:
                error_messages.append(
                    f"{prefix}a singleton source regime must declare exactly one "
                    f"leg (got {sorted(edge.legs)})."
                )
        elif set(edge.legs) != set(source_stakeholders):
            error_messages.append(
                f"{prefix}the legs must be keyed by exactly the source "
                f"stakeholders {sorted(source_stakeholders)}; got "
                f"{sorted(edge.legs)}."
            )

    if error_messages:
        raise RegimeInitializationError(format_messages(error_messages))


def _fail_if_gated_edge_source_out_of_scope(regime: lcm.regime.Regime) -> None:
    """Reject a gated-edge source outside the GridSearch / no-shock scope (E3')."""
    if not isinstance(regime.solver, GridSearch):
        raise NotImplementedError(
            "Gated edges (E3') are only implemented for GridSearch source "
            "regimes: the source reads the folded continuation through the "
            "grid-search machinery. Edges touching DC-EGM regimes are out of "
            "scope for this slice. Use `solver=GridSearch()`."
        )
    if regime.taste_shocks is not None:
        raise NotImplementedError(
            "Gated edges (E3') on a taste-shock source regime are out of scope "
            "for this slice."
        )
    if regime.certainty_equivalent is not None:
        raise NotImplementedError(
            "Gated edges (E3') on a certainty-equivalent source regime are out "
            "of scope for this slice."
        )


def _regime_transition_target_names(transition: object) -> set[str] | None:
    """Return the reachable target regime names of a regime transition, if known.

    A per-target dict names them directly; a bare callable or `MarkovTransition`
    resolves its target only at runtime, so returns `None` (skip the membership
    check). `Phased` uses its solve variant.
    """
    from lcm.phased import Phased  # noqa: PLC0415

    if isinstance(transition, Phased):
        transition = transition.solve
    if isinstance(transition, Mapping):
        return {str(key) for key in cast("Mapping[str, object]", transition)}
    return None


def _validate_mapping_contents(regime: lcm.regime.Regime) -> None:
    """Exhaustively check key/value types of `regime`'s mapping fields.

    Beartype on `Regime` catches top-level type mismatches and a sampled
    Mapping entry, but does not deep-check every key/value of a Mapping
    parameter — especially when the value type is a `Callable`/`Protocol`,
    which beartype skips entirely. This function fills that gap by
    iterating each entry and reporting all type violations together via
    the standard `error_messages` aggregator.

    """
    error_messages = [
        *_grid_mapping_errors(
            attr_name="states", mapping=regime.states, allow_phase_variants=True
        ),
        *_grid_mapping_errors(
            attr_name="actions", mapping=regime.actions, allow_phase_variants=False
        ),
        *_callable_mapping_errors(
            attr_name="functions", mapping=regime.functions, allow_phase_variants=True
        ),
        *_callable_mapping_errors(
            attr_name="constraints",
            mapping=regime.constraints,
            allow_phase_variants=False,
        ),
    ]

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_logical_consistency(regime: lcm.regime.Regime) -> None:
    """Validate the local, value-shape consistency of a regime.

    Completeness properties (a `utility` entry, state-transition coverage,
    state/action overlap, distributed-grid rules) may be satisfied only after
    model-level slots are merged in; `_validate_completeness` checks them when
    the regimes are finalized at model build.
    """
    error_messages: list[str] = []

    all_function_names = [*regime.constraints.keys(), *regime.functions.keys()]
    invalid_function_names = [
        name for name in all_function_names if QNAME_DELIMITER in name
    ]
    if invalid_function_names:
        error_messages.append(
            f"Function names cannot contain the reserved separator "
            f"'{QNAME_DELIMITER}'. The following names are invalid: "
            f"{invalid_function_names}.",
        )

    next_prefixed = [name for name in all_function_names if name.startswith("next_")]
    if next_prefixed:
        error_messages.append(
            f"Function names must not start with 'next_' — this prefix is "
            f"reserved for auto-generated state transition functions. "
            f"Invalid names: {next_prefixed}.",
        )

    all_variable_names = [*regime.states.keys(), *regime.actions.keys()]
    invalid_variable_names = [
        name for name in all_variable_names if QNAME_DELIMITER in name
    ]
    if invalid_variable_names:
        error_messages.append(
            f"State and action names cannot contain the reserved separator "
            f"'{QNAME_DELIMITER}'. The following names are invalid: "
            f"{invalid_variable_names}.",
        )

    error_messages.extend(_validate_active(regime.active))
    error_messages.extend(_state_transition_grammar_errors(regime))
    error_messages.extend(_regime_transition_grammar_errors(regime.transition))

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _regime_transition_grammar_errors(transition: object) -> list[str]:
    """Validate the regime `transition` value vocabulary.

    A `Phased` container's sides are each held to the bare vocabulary
    (callable, `MarkovTransition`, or a per-target dict); per-target cells
    must be `MarkovTransition`-wrapped probability functions.
    """
    error_messages: list[str] = []
    sides = (
        (
            (transition.solve, " solve variant"),
            (transition.simulate, " simulate variant"),
        )
        if isinstance(transition, Phased)
        else ((transition, ""),)
    )
    for side, label in sides:
        if not isinstance(side, Mapping):
            continue
        if not side:
            error_messages.append(
                f"transition{label}: an empty per-target dict declares no "
                f"reachable targets — use `transition=None` for a terminal "
                f"regime.",
            )
        for target_regime_name, cell in side.items():
            if not isinstance(target_regime_name, str):
                error_messages.append(
                    f"transition{label} per-target dict key {target_regime_name!r} "
                    f"must be a string.",
                )
            if isinstance(cell, Phased):
                error_messages.append(
                    f"transition{label}['{target_regime_name}'] cannot be `Phased` — "
                    f"`Phased` is outermost-only: wrap the whole entry, e.g. "
                    f"`Phased(solve={{...}}, simulate={{...}})`.",
                )
            elif not isinstance(cell, MarkovTransition):
                error_messages.append(
                    f"transition{label}['{target_regime_name}'] must be a "
                    f"`MarkovTransition`-wrapped probability function — "
                    f"deterministic per-target regime transitions are not yet "
                    f"supported (use the coarse form, or `MarkovTransition` "
                    f"with indicator probabilities).",
                )
    return error_messages


def _validate_completeness(regime: lcm.regime.Regime) -> list[str]:
    """Collect completeness errors for a finalized (post-merge) regime.

    These properties hold only for the regime the model actually runs —
    a bare user `Regime` may legitimately lack them until model-level slots
    are merged in:

    - a `utility` entry in `functions`
    - state-transition coverage (every non-process state has a law; terminal
      regimes have none; identity laws refer to existing states)
    - no state/action name overlap
    - distributed-grid rules
    - no function-output / discrete-grid-indexed-input name clash
    """
    error_messages: list[str] = []

    if regime.stakeholders is not None:
        # A collective regime supplies a per-stakeholder `utility_<s>` instead
        # of a single `utility` (validated in full by `_validate_collective_regime`
        # at construction); require each here so a finalized regime stays complete.
        missing = [
            s for s in regime.stakeholders if f"utility_{s}" not in regime.functions
        ]
        if missing:
            error_messages.append(
                "A collective regime must provide a per-stakeholder utility "
                f"function for each stakeholder. Missing: "
                f"{[f'utility_{s}' for s in missing]}.",
            )
    elif "utility" not in regime.functions:
        error_messages.append(
            "A 'utility' function must be provided in the functions dictionary.",
        )

    error_messages.extend(_state_transition_coverage_errors(regime))
    error_messages.extend(_validate_function_output_grid_indexing(regime))
    error_messages.extend(_validate_distributed_grids(regime))
    error_messages.extend(_certainty_equivalent_errors(regime))

    states_and_actions_overlap = set(regime.states) & set(regime.actions)
    if states_and_actions_overlap:
        error_messages.append(
            "States and actions cannot have overlapping names. The following names "
            f"are used in both states and actions: {states_and_actions_overlap}.",
        )

    return error_messages


def _validate_distributed_grids(regime: lcm.regime.Regime) -> list[str]:
    """Reject `distributed=True` on action grids.

    Distribution shards the V-array along state axes; an action grid has no
    corresponding V-array axis, so marking one as distributed has no
    consistent meaning. To shard an axis a user cares about, set
    `distributed=True` on the matching state.
    """
    offending_actions = [
        name
        for name, grid in regime.actions.items()
        if grid is not None and grid.distributed
    ]
    if not offending_actions:
        return []
    return [
        "Action grids cannot be marked `distributed=True` — distribution "
        "shards V-array axes, which come from states. Move `distributed=True` "
        "to the corresponding state grid. Offending actions: "
        f"{offending_actions}.",
    ]


def _certainty_equivalent_errors(regime: lcm.regime.Regime) -> list[str]:
    """Collect errors for a regime's `certainty_equivalent` declaration.

    - terminal regimes have no continuation value to aggregate
    - only `GridSearch` supports a nonlinear certainty equivalent (the
      Euler inversion in DC-EGM assumes expected utility)
    """
    if regime.certainty_equivalent is None:
        return []
    error_messages: list[str] = []
    if regime.terminal:
        error_messages.append(
            "A terminal regime cannot declare `certainty_equivalent`: there "
            "is no continuation value to aggregate."
        )
    if isinstance(regime.solver, DCEGM):
        error_messages.append(
            "The DCEGM solver does not support a nonlinear "
            "`certainty_equivalent`: the Euler inversion assumes expected "
            "utility. Use GridSearch() for this regime."
        )
    return error_messages


def _validate_function_output_grid_indexing(
    regime: lcm.regime.Regime,
) -> list[str]:
    """Detect the regime-function-output / discrete-grid-indexed-input name clash.

    The unsafe pattern is: a regime function `f` takes a discrete grid `g`
    (state, action, or derived categorical) as an input — so `f`'s output
    is a per-cell scalar — and a consumer then indexes `f[g]`. The
    consumer is indexing a 0-d array by a scalar integer, which raises
    `IndexError` at trace time. The fix is to drop the redundant `[g]`
    in the consumer (or refactor `f` not to take `g`).

    This check is deliberately best-effort: it catches the common
    `func_output[discrete_grid]` subscript form and nothing else. It is not
    meant to grow into a general correctness checker for user functions —
    if it ever produces false positives, prefer deleting it over hardening
    it to chase every way the pattern can hide.
    """
    function_output_names = set(regime.functions)
    discrete_grid_names = (
        {name for name, grid in regime.states.items() if isinstance(grid, DiscreteGrid)}
        | {
            name
            for name, grid in regime.actions.items()
            if isinstance(grid, DiscreteGrid)
        }
        | set(regime.derived_categoricals)
    )
    if not function_output_names or not discrete_grid_names:
        return []

    # Only treat `func_output[grid]` as unsafe when the producing function
    # *also* takes `grid` as an input — that is the case where the output
    # is per-cell scalar and the consumer's indexing is wrong. If the
    # producing function does not take `grid`, its output shape is
    # whatever it computed (typically an array indexable by `grid`) and
    # the consumer pattern is correct.
    function_inputs = _function_input_names(
        {name: func for name, func in regime.functions.items() if func is not None}
    )
    consumers = _collect_indexing_consumers(regime)

    errors: list[str] = []
    for consumer_name, func in consumers:
        clashes = _find_function_output_grid_indexing(
            func=func,
            function_output_names=function_output_names,
            discrete_grid_names=discrete_grid_names,
        )
        for func_output_name, grid_name in clashes:
            if grid_name not in function_inputs.get(func_output_name, set()):
                continue
            errors.append(
                f"Consumer '{consumer_name}' indexes regime function output "
                f"'{func_output_name}' by discrete grid '{grid_name}' "
                f"(`{func_output_name}[{grid_name}]`), but '{func_output_name}' "
                f"already takes '{grid_name}' as input — its output is a "
                f"per-cell scalar, so the indexing raises IndexError at trace "
                f"time. Drop the redundant `[{grid_name}]` in '{consumer_name}', "
                f"or refactor '{func_output_name}' not to take '{grid_name}' "
                f"as input."
            )
    return errors


def _function_input_names(
    functions: Mapping[str, Callable | Phased],
) -> dict[str, set[str]]:
    """Return each regime function's input-parameter names.

    A `Phased` contributes the union of both variants' parameters;
    unintrospectable callables contribute the empty set.
    """
    result: dict[str, set[str]] = {}
    for name, func in functions.items():
        params: set[str] = set()
        for variant in _function_variants(func):
            try:
                params |= set(inspect.signature(variant).parameters)
            except ValueError, TypeError:
                continue
        result[name] = params
    return result


def _collect_indexing_consumers(
    regime: lcm.regime.Regime,
) -> list[tuple[str, Callable]]:
    """Return `(name, callable)` pairs whose bodies are scanned for the clash.

    Functions and constraints contribute every variant of a
    `Phased`; the regime transition contributes itself.
    """
    consumers: list[tuple[str, Callable]] = []
    for name, func in regime.functions.items():
        if func is None:
            continue
        consumers.extend((name, variant) for variant in _function_variants(func))
    for name, constraint in regime.constraints.items():
        if constraint is None:
            continue
        consumers.extend((name, variant) for variant in _function_variants(constraint))
    if callable(regime.transition):
        consumers.append(("regime_transition", regime.transition))
    return consumers


def _function_variants(
    func: Callable | Phased,
) -> tuple[Callable, ...]:
    """Return the callable variants of a regime-function entry.

    A plain function is itself; a `Phased` yields its `solve` and `simulate`
    callables so both phases are scanned.
    """
    if isinstance(func, Phased):
        return (cast("Callable", func.solve), cast("Callable", func.simulate))
    return (cast("Callable", func),)


def _find_function_output_grid_indexing(
    *,
    func: Callable,
    function_output_names: set[str],
    discrete_grid_names: set[str],
) -> list[tuple[str, str]]:
    """Return `(function_output_name, grid_name)` clashes inside `func`'s body."""
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except OSError, TypeError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    clashes: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in function_output_names:
            continue
        if not isinstance(node.slice, ast.Name):
            continue
        if node.slice.id not in discrete_grid_names:
            continue
        clashes.append((node.value.id, node.slice.id))
    return clashes


def _validate_active(active: ActiveFunction) -> list[str]:
    """Validate the active attribute is a callable."""
    if not callable(active):
        return ["active must be a callable that takes age (float) and returns bool."]
    return []


def _state_transition_grammar_errors(regime: lcm.regime.Regime) -> list[str]:
    """Validate each `state_transitions` entry against the value vocabulary."""
    error_messages: list[str] = []
    for name, value in regime.state_transitions.items():
        error_messages.extend(_state_transition_value_errors(name=name, value=value))
    return error_messages


def _state_transition_coverage_errors(regime: lcm.regime.Regime) -> list[str]:
    """Validate that `state_transitions` covers exactly the regime's states."""
    error_messages: list[str] = []

    process_names: set[ProcessName] = {
        name
        for name, grid in regime.states.items()
        if isinstance(grid, _ContinuousStochasticProcess)
    }
    non_process_names: set[StateName] = set(regime.states) - process_names

    # Keys not in states are allowed only with actual transitions. A
    # `fixed_transition` entry is an identity law, which requires the state
    # to exist in this regime.
    extra_keys = set(regime.state_transitions) - set(regime.states)
    for key in extra_keys:
        value = regime.state_transitions[key]
        if isinstance(value, _IdentityTransition):
            error_messages.append(
                f"state_transitions['{key}'] is `fixed_transition` but '{key}' "
                f"is not in states. Identity laws require the state to exist "
                f"in this regime.",
            )

    process_in_transitions = process_names & set(regime.state_transitions)
    if process_in_transitions:
        error_messages.append(
            f"Stochastic process states have intrinsic transitions and must not "
            f"appear in state_transitions: {process_in_transitions}.",
        )

    if regime.terminal:
        if regime.state_transitions:
            error_messages.append(
                "Terminal regimes must have empty state_transitions.",
            )
        return error_messages

    missing = non_process_names - set(regime.state_transitions)
    if missing:
        error_messages.append(
            f"Every non-process state must have an entry in state_transitions. "
            f"Missing: {missing}. Use `fixed_transition(state_name)` for "
            f"fixed states.",
        )

    return error_messages


def _state_transition_value_errors(*, name: StateName, value: object) -> list[str]:
    """Validate one `state_transitions` entry against the value vocabulary.

    Each variant of a `Phased` entry is held to the vocabulary of a bare
    value — callable or a per-target Mapping — except that a stochastic
    (`MarkovTransition`) variant is rejected: per-phase stochasticity of a
    law of motion is not yet supported. `None` is not a law of motion; the
    error points to `fixed_transition`.
    """
    error_messages: list[str] = []
    phase_variant = isinstance(value, Phased)
    for variant, label in _state_transition_variants(value):
        if phase_variant and isinstance(variant, MarkovTransition):
            error_messages.append(
                f"state_transitions['{name}']{label}: a stochastic "
                f"(`MarkovTransition`) variant inside `Phased` is not yet "
                f"supported.",
            )
            continue
        if variant is None:
            if phase_variant:
                error_messages.append(
                    f"state_transitions['{name}']{label}: a mask is "
                    f"whole-entry only — `None` cannot appear inside "
                    f"`Phased`.",
                )
            # Bare `None` masks a model-level law; bound at model build.
            continue
        error_messages.extend(
            _fixed_transition_name_mismatch(state_name=name, value=variant, label=label)
        )
        if callable(variant):
            continue
        if isinstance(variant, Mapping):
            error_messages.extend(
                _validate_per_target_dict(
                    state_name=name,
                    targets=cast("Mapping[RegimeName, object]", variant),
                )
            )
        else:
            error_messages.append(
                f"state_transitions['{name}']{label} must be callable, "
                f"MarkovTransition, `fixed_transition(...)`, or a per-target "
                f"Mapping, got {type(variant).__name__}.",
            )
    return error_messages


def _fixed_transition_name_mismatch(
    *, state_name: StateName, value: object, label: str
) -> list[str]:
    """Reject a `fixed_transition` whose argument differs from its dict key."""
    if (
        isinstance(value, _IdentityTransition) and value._state_name != state_name  # noqa: SLF001
    ):
        return [
            f"state_transitions['{state_name}']{label}: "
            f"`fixed_transition('{value._state_name}')` is assigned to state "  # noqa: SLF001
            f"'{state_name}' — the names must match.",
        ]
    return []


def _state_transition_variants(value: object) -> tuple[tuple[object, str], ...]:
    """Return a state-transition entry's per-phase variants with display labels.

    A bare value is its own single variant; a `Phased` yields its solve and
    simulate variants, each validated against the same vocabulary.
    """
    if isinstance(value, Phased):
        return ((value.solve, " solve variant"), (value.simulate, " simulate variant"))
    return ((value, ""),)


def _fold_state_names(regime: lcm.regime.Regime) -> tuple[StateName, ...]:
    """Return the regime's IID-process states declared `fold=True`."""
    return tuple(name for name, grid in regime.states.items() if _is_folded(grid))


def _is_folded(grid: object) -> bool:
    """Whether a state's grid is an IID process declaring `fold=True`."""
    return isinstance(grid, _IIDProcess) and grid.fold


def _flatten_transition_callables(value: object) -> list[Callable]:
    """Return every callable reachable from a `state_transitions` / `transition` entry.

    Unwraps `Phased` (both variants) and per-target `Mapping`s; `MarkovTransition`
    and `_IdentityTransition` are themselves callables with an introspectable
    signature (`MarkovTransition` sets `__wrapped__`; `_IdentityTransition` sets
    `__signature__`), so `inspect.signature` resolves them correctly downstream.
    """
    if value is None:
        return []
    if isinstance(value, Phased):
        return _flatten_transition_callables(
            value.solve
        ) + _flatten_transition_callables(value.simulate)
    if isinstance(value, Mapping):
        out: list[Callable] = []
        for entry in value.values():
            out.extend(_flatten_transition_callables(entry))
        return out
    if callable(value):
        return [cast("Callable", value)]
    return []


def _fold_dependency_closure(
    *, roots: tuple[Callable, ...], resolution_table: Mapping[str, Callable | Phased]
) -> set[str]:
    """Return every argument name in the transitive DAG ancestry of `roots`.

    Walks argument names of each root; whenever a name matches an entry of
    `resolution_table` (an ordinary regime function or constraint), its
    variants' argument names are pulled in too, recursively. Leaf names
    (states, actions, params, `period`/`age`) terminate the walk. This is the
    same "does X depend on Y" question `_find_function_output_grid_indexing`
    answers by AST-walking; here a signature walk suffices because the
    question is about *argument names*, not source text.
    """
    seen_names: set[str] = set()
    stack: list[Callable] = list(roots)
    visited_ids: set[int] = set()
    while stack:
        func = stack.pop()
        if id(func) in visited_ids:
            continue
        visited_ids.add(id(func))
        try:
            params = inspect.signature(func).parameters
        except ValueError, TypeError:
            continue
        for name in params:
            if name in seen_names:
                continue
            seen_names.add(name)
            entry = resolution_table.get(name)
            if entry is not None:
                stack.extend(_function_variants(entry))
    return seen_names


def _validate_fold_declarations(regime: lcm.regime.Regime) -> None:
    """Reject `fold=True` IID-process declarations the fold machinery can't support.

    A fold integrates a shock's node axis into the stored value by quadrature
    immediately after the period's max-over-actions / collective readout, so
    nothing may condition on which node was realized beyond that point:

    - a same-period gate / value-constraint predicate (E2/E3') that reads the
      shock's realized value — these read live per-node values to decide a
      within-period household choice or a dissolution/consent gate, which the
      fold has already averaged away by the time any *other* regime's
      same-period logic runs;
    - a next-period transition (state or regime) that reads the shock's
      realized value — a folded shock is integrated out, so no downstream
      state may depend on which node was realized;
    - EV1 `taste_shocks` on the same regime — the taste-shock reduction is a
      separate closed-form max over discrete actions; the two reductions are
      not composed;
    - a nonlinear `certainty_equivalent` on the same regime — the fold
      reduction is always the ARITHMETIC `zero_safe_average`, exact only for
      the linear expectation `E[V']`; a nonlinear certainty equivalent needs
      the shock's node axis intact to apply its own aggregator;
    - any solver other than `GridSearch` — the fold reduction is implemented
      only in the grid-search max-Q-over-a kernel;
    - a process with runtime-supplied distribution params — the fold weights
      are computed once at kernel-build time from the process's own
      (fully-specified) `compute_transition_probs`.

    A persistent (non-IID) process has no `fold` field at all, so `fold=True`
    on one is rejected by the type system before this validator ever runs.

    Whether the folded state structurally PERSISTS (is redeclared, with an
    intrinsic `next_<name>` continuation, in any regime reachable from this
    one — including itself) is a cross-regime property, checked once every
    regime's transitions are built (`_fail_if_folded_state_persists` in
    `regime_building/processing.py`), not here.
    """
    fold_names = _fold_state_names(regime)
    if not fold_names:
        return

    error_messages = [
        *_fold_scope_errors(regime, fold_names),
        *_fold_same_period_read_errors(regime, fold_names),
        *_fold_transition_read_errors(regime, fold_names),
    ]
    if error_messages:
        raise RegimeInitializationError(format_messages(error_messages))


def _fold_scope_errors(
    regime: lcm.regime.Regime, fold_names: tuple[StateName, ...]
) -> list[str]:
    """Collect the regime-wide (not DAG-dependency) fold restrictions."""
    error_messages: list[str] = []
    runtime_params = [
        name
        for name in fold_names
        if cast("_IIDProcess", regime.states[name]).params_to_pass_at_runtime
    ]
    if runtime_params:
        error_messages.append(
            f"fold=True on state(s) {sorted(runtime_params)} is not yet "
            "supported together with runtime-supplied distribution params: "
            "the fold weights are computed once, at kernel-build time, from "
            "the process's own (fully-specified) `compute_transition_probs`. "
            "Supply the distribution params directly on the process, or drop "
            "`fold=True`."
        )
    if regime.taste_shocks is not None:
        error_messages.append(
            f"fold=True on state(s) {sorted(fold_names)} is not supported "
            "together with EV1 `taste_shocks` on the same regime: the "
            "taste-shock reduction is a closed-form max over discrete "
            "actions, and folding an IID state is a separate quadrature "
            "reduction over a state axis — the two reductions are not "
            "composed."
        )
    if regime.certainty_equivalent is not None:
        error_messages.append(
            f"fold=True on state(s) {sorted(fold_names)} is not supported "
            "together with a nonlinear `certainty_equivalent`: the fold "
            "reduction (`_wrap_with_fold_reduction`) always averages the "
            "shock's node axis ARITHMETICALLY (`zero_safe_average`), exact "
            "only for the LINEAR expectation E[V']. A nonlinear certainty "
            "equivalent needs the shock's node axis intact to apply its own "
            "(possibly nonlinear) aggregator instead. Drop `fold=True`, or "
            "drop `certainty_equivalent`."
        )
    if not isinstance(regime.solver, GridSearch):
        error_messages.append(
            f"fold=True on state(s) {sorted(fold_names)} is only implemented "
            "for the `GridSearch` solver: the fold reduction runs inside the "
            "grid-search max-Q-over-a kernel. Use `solver=GridSearch()`, or "
            "drop `fold=True`."
        )
    return error_messages


def _fold_resolution_table(regime: lcm.regime.Regime) -> dict[str, Callable | Phased]:
    """Regime functions/constraints usable as DAG-ancestor resolution targets."""
    return {
        **{k: v for k, v in regime.constraints.items() if v is not None},
        **{k: v for k, v in regime.functions.items() if v is not None},
    }


def _fold_same_period_roots(regime: lcm.regime.Regime) -> list[tuple[str, Callable]]:
    """Named same-period gate / value-constraint / reference-projection roots.

    Deliberately EXCLUDES this regime's own OUTBOUND `gated_edges[...].gate`
    and `gated_edges[...].gate_refs[...]` projections (F4 audit finding): a
    `GatedEdge`'s `gate` and its `gate_refs` projections are compiled and
    evaluated on the TARGET regime's grid/DAG
    (`_attach_gated_edge_folds`/`_resolve_gated_edge`), never on this
    (source) regime's own — so an argument name they declare has no
    relationship to a state THIS regime folds, even when the names happen to
    collide (e.g. both regimes declare a `wage_shock`). Walking them here as
    if they were source-local reads produced false positives: a source
    folding a state whose name a target-grid gate merely reuses was
    incorrectly rejected. The genuine cross-regime hazard — this regime
    itself being read nodewise as a gated-edge target, leg fallback, or
    same-period reference — is the now-COMPLETE
    `_fail_if_folded_regime_is_same_period_endpoint` guard in
    `regime_building/processing.py`, which correctly checks the TARGET side
    of the very same declarations. `value_constraints` and
    `same_period_refs` stay as roots here: both are evaluated on THIS
    (source) regime's own grid/DAG, so a source-local name collision there
    is a real same-period read of this regime's own fold name.
    """
    roots: list[tuple[str, Callable]] = []
    for name, predicate in regime.value_constraints.items():
        roots.extend(
            (f"value_constraints['{name}']", variant)
            for variant in _function_variants(predicate)
        )
    for ref_name, ref in regime.same_period_refs.items():
        roots.extend(
            (f"same_period_refs['{ref_name}']", func)
            for func in ref.projection.values()
        )
    return roots


def _fold_same_period_read_errors(
    regime: lcm.regime.Regime, fold_names: tuple[StateName, ...]
) -> list[str]:
    """Reject a fold name read by a same-period gate / value-constraint (E2/E3')."""
    resolution_table = _fold_resolution_table(regime)
    error_messages: list[str] = []
    for label, func in _fold_same_period_roots(regime):
        hit = _fold_names_ordered_intersection(
            fold_names,
            _fold_dependency_closure(roots=(func,), resolution_table=resolution_table),
        )
        if hit:
            error_messages.append(
                f"fold=True on state(s) {hit} conflicts with {label}, which "
                "reads the shock's realized value: a same-period gate / "
                "value-constraint / reference projection needs the unfolded "
                "per-node value (E2/E3'), but a folded state's node axis is "
                "averaged away before the period's value is published. Drop "
                "`fold=True` on the shock, or stop reading it there."
            )
    return error_messages


def _fold_transition_read_errors(
    regime: lcm.regime.Regime, fold_names: tuple[StateName, ...]
) -> list[str]:
    """Reject a fold name read by a next-period state / regime transition."""
    transition_roots: list[Callable] = []
    for value in regime.state_transitions.values():
        transition_roots.extend(_flatten_transition_callables(value))
    transition_roots.extend(_flatten_transition_callables(regime.transition))

    resolution_table = _fold_resolution_table(regime)
    transition_hit: set[str] = set()
    for func in transition_roots:
        transition_hit |= _fold_dependency_closure(
            roots=(func,), resolution_table=resolution_table
        )
    transition_hit &= set(fold_names)
    if not transition_hit:
        return []
    return [
        f"fold=True on state(s) {sorted(transition_hit)} conflicts with a "
        "next-period transition that reads the shock's realized value: a "
        "folded shock is integrated out at solve time, so no downstream "
        "state or regime transition may condition on which node was "
        "realized. Drop `fold=True`, or stop conditioning the transition "
        "on it."
    ]


def _fold_names_ordered_intersection(
    fold_names: tuple[StateName, ...], other: set[str]
) -> list[StateName]:
    """Return `fold_names` filtered to `other`, preserving `fold_names`' order."""
    return [name for name in fold_names if name in other]


def _validate_per_target_dict(
    *, state_name: StateName, targets: Mapping[RegimeName, object]
) -> list[str]:
    """Validate a per-target transition dict for stochastic consistency and types."""
    error_messages: list[str] = []
    markov_count = 0
    for target_regime_name, law in targets.items():
        if not isinstance(target_regime_name, str):
            error_messages.append(
                f"state_transitions['{state_name}'] per-target dict key "
                f"{target_regime_name!r} must be a string.",
            )
        if isinstance(law, Phased):
            error_messages.append(
                f"state_transitions['{state_name}']['{target_regime_name}'] cannot "
                f"be `Phased` — `Phased` is outermost-only: wrap the whole "
                f"entry, e.g. `Phased(solve={{...}}, simulate={{...}})`.",
            )
        elif isinstance(law, MarkovTransition):
            markov_count += 1
        elif isinstance(law, _IdentityTransition):
            error_messages.extend(
                _fixed_transition_name_mismatch(
                    state_name=state_name,
                    value=law,
                    label=f"['{target_regime_name}']",
                )
            )
        elif not callable(law):
            error_messages.append(
                f"state_transitions['{state_name}']['{target_regime_name}'] must be "
                f"callable or MarkovTransition, got "
                f"{type(law).__name__}.",
            )
    if 0 < markov_count < len(targets):
        error_messages.append(
            f"state_transitions['{state_name}'] per-target dict must be "
            f"consistently stochastic: either all values are "
            f"MarkovTransition or none are.",
        )
    return error_messages
