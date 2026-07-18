"""Validation helpers for the user-facing `Regime`.

These functions back `Regime.__post_init__`. They raise nothing themselves —
instead they collect error messages, which `__post_init__` aggregates into a
single `RegimeInitializationError`. Splitting the validators out of the
public module keeps `lcm.regime` to class definitions.

"""

import ast
import inspect
import textwrap
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from dags.tree import QNAME_DELIMITER

from _lcm.certainty_equivalent import PowerMean
from _lcm.grids import DiscreteGrid, Grid
from _lcm.identity_transition import _IdentityTransition
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import ActiveFunction, ProcessName, RegimeName, StateName
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.solvers import NBEGM, NNBEGM, GridSearch
from lcm.temporal_aggregation import H_epstein_zin
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

    if "utility" not in regime.functions:
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
    - `GridSearch` and `NBEGM` support a nonlinear certainty equivalent (the
      Epstein-Zin recursion); the other endogenous-grid solvers' Euler inversion
      assumes expected utility, so a declared certainty equivalent must be
      rejected rather than silently ignored
    - Epstein-Zin and extreme-value taste shocks do not compose: the taste-shock
      logsum is not invariant under the certainty-equivalent transform, so the
      combination is rejected
    """
    if regime.certainty_equivalent is None:
        return []
    error_messages: list[str] = []
    if regime.terminal:
        error_messages.append(
            "A terminal regime cannot declare `certainty_equivalent`: there "
            "is no continuation value to aggregate."
        )
    if not isinstance(regime.solver, (GridSearch, NBEGM, NNBEGM)):
        error_messages.append(
            f"The {type(regime.solver).__name__} solver does not support a "
            "nonlinear `certainty_equivalent`: its Euler inversion assumes "
            "expected utility. Use GridSearch(), NBEGM(), or NNBEGM() for "
            "this regime."
        )
    if isinstance(regime.solver, (NBEGM, NNBEGM)):
        # The endogenous-grid kernels implement the Epstein-Zin recursion for
        # exactly one pairing: they read the power mean's `risk_aversion`
        # parameter for the transform partials and the aggregator's
        # intertemporal elasticity for the Euler inversion and period value.
        # NNBEGM's inner solve runs the same NBEGM kernels, so the contract
        # binds it identically. GridSearch aggregates any certainty
        # equivalent in concrete values, so only the endogenous-grid routes
        # are narrowed.
        solver_name = type(regime.solver).__name__
        if not isinstance(regime.certainty_equivalent, PowerMean):
            error_messages.append(
                f"{solver_name} implements the recursive certainty "
                f"equivalent for `PowerMean` only, got "
                f"{type(regime.certainty_equivalent).__name__}. Use "
                f"`certainty_equivalent=PowerMean()` or solve the regime with "
                f"GridSearch()."
            )
        if regime.functions.get("H") is not H_epstein_zin:
            error_messages.append(
                f"{solver_name} with a `certainty_equivalent` requires the "
                "regime's aggregator to be `H_epstein_zin` "
                '(`functions={"H": lcm.H_epstein_zin, ...}`): the Euler '
                "inversion and period value read its intertemporal "
                "elasticity. With a different `H` the kernels would solve a "
                "recursion the regime does not declare."
            )
    if regime.taste_shocks is not None:
        error_messages.append(
            "A regime cannot combine `certainty_equivalent` with "
            "`taste_shocks`: the extreme-value logsum is not invariant under "
            "the certainty-equivalent transform, so the Epstein-Zin recursion "
            "and taste shocks do not compose."
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


def _phased_per_target_shape_mismatch(*, name: StateName, value: Phased) -> list[str]:
    """Inside `Phased`, two per-target dicts must declare the same target set.

    `Phased(solve={...}, simulate={...})` is normalized at collection into one entry per
    target, each holding a `Phased` of that target's two laws — the form the engine
    actually consumes (`transitions._add_raw_transition`).

    Two shapes normalize cleanly and are both allowed:

    - **both per-target** over the SAME targets — paired cell by cell;
    - **one per-target, the other a bare law** — the bare law BROADCASTS over the
      per-target side's targets, exactly as a bare state law broadcasts outside
      `Phased`. The per-target side's key set defines the targets, and per-target
      coverage validation independently requires that set to be the full reachable
      set — so the broadcast target list is authoritative, not arbitrary.

    Only the remaining shape is rejected: two per-target dicts over DIFFERENT
    targets, which would leave a target with a law in one phase and none in the
    other. (`Phased` is outermost-only — `_validate_per_target_dict` rejects a
    `Phased` cell — so the outer form is the only spelling for a per-target law
    that varies by phase.)
    """
    solve_per_target = isinstance(value.solve, Mapping)
    simulate_per_target = isinstance(value.simulate, Mapping)
    if not (solve_per_target and simulate_per_target):
        # neither per-target (a plain phased law), or map-vs-bare (broadcast):
        # both normalize cleanly, so nothing to reject here.
        return []
    solve_targets = set(cast("Mapping[RegimeName, object]", value.solve))
    simulate_targets = set(cast("Mapping[RegimeName, object]", value.simulate))
    if solve_targets != simulate_targets:
        return [
            f"state_transitions['{name}']: the per-target dicts inside `Phased` "
            f"declare different targets — solve has {sorted(solve_targets)}, simulate "
            f"has {sorted(simulate_targets)}. Both phases must cover the same targets, "
            f"or give one phase a single law to broadcast over the other's targets.",
        ]
    return []


def _state_transition_value_errors(*, name: StateName, value: object) -> list[str]:
    """Validate one `state_transitions` entry against the value vocabulary.

    Each variant of a `Phased` entry is held to the vocabulary of a bare value —
    callable, `MarkovTransition`, or a per-target Mapping. A stochastic variant inside
    `Phased` is supported: the solve variant is the perceived law that prices the
    continuation in Q, the simulate variant is the true law the next state is drawn
    from.

    The two variants need NOT agree on whether the law is stochastic. A deterministic
    law is a degenerate kernel, so the state has the same domain either way, and the two
    phase cores classify their stochastic names independently — a perceived kernel with
    a point-valued truth, and the reverse, both build and carry the intended meaning
    (`tests/regime_building/test_mixed_stochasticity_phases.py`).

    The two variants may differ in shape: per-target on one side and a bare law on
    the other broadcasts the bare law over the per-target side's targets (map-vs-bare).
    What is rejected is two per-target dicts over DIFFERENT target sets — an ambiguous
    normalization with no single authoritative key set — see
    `_phased_per_target_shape_mismatch`.

    `None` is not a law of motion; the error points to `fixed_transition`.
    """
    error_messages: list[str] = []
    phase_variant = isinstance(value, Phased)
    if phase_variant:
        error_messages.extend(_phased_per_target_shape_mismatch(name=name, value=value))
    for variant, label in _state_transition_variants(value):
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
