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

from _lcm.grids import DiscreteGrid, Grid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import ActiveFunction, ProcessName, RegimeName, StateName
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.transition import (
    MarkovTransition,
    SolveSimulateFunctionPair,
    SolveSimulateStatePair,
)

if TYPE_CHECKING:
    import lcm.regime


def _grid_mapping_errors(
    attr_name: str, mapping: Mapping[str, object], *, allow_phase_variants: bool
) -> list[str]:
    """Collect key/value type errors for a grid-valued mapping (states/actions)."""
    allowed = Grid | SolveSimulateStatePair | Phased if allow_phase_variants else Grid
    suffix = ", SolveSimulateStatePair, or Phased" if allow_phase_variants else ""
    error_messages: list[str] = []
    for k, v in mapping.items():
        if not isinstance(k, str):
            error_messages.append(f"{attr_name} key {k!r} must be a string.")
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
    attr_name: str, mapping: Mapping[str, object], *, allow_phase_variants: bool
) -> list[str]:
    """Collect key/value type errors for a callable-valued mapping."""
    error_messages: list[str] = []
    for k, v in mapping.items():
        if not isinstance(k, str):
            error_messages.append(f"{attr_name} key {k!r} must be a string.")
        if isinstance(v, Phased | SolveSimulateFunctionPair):
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
        *_grid_mapping_errors("states", regime.states, allow_phase_variants=True),
        *_grid_mapping_errors("actions", regime.actions, allow_phase_variants=False),
        *_callable_mapping_errors(
            "functions", regime.functions, allow_phase_variants=True
        ),
        *_callable_mapping_errors(
            "constraints", regime.constraints, allow_phase_variants=False
        ),
    ]

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_logical_consistency(regime: lcm.regime.Regime) -> None:
    """Validate the logical consistency of the regime."""
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

    if "utility" not in regime.functions:
        error_messages.append(
            "A 'utility' function must be provided in the functions dictionary.",
        )

    error_messages.extend(_validate_active(regime.active))
    error_messages.extend(_state_pair_field_errors(regime))
    error_messages.extend(_validate_state_transitions(regime))
    error_messages.extend(_validate_function_output_grid_indexing(regime))
    error_messages.extend(_validate_distributed_grids(regime))

    states_and_actions_overlap = set(regime.states) & set(regime.actions)
    if states_and_actions_overlap:
        error_messages.append(
            "States and actions cannot have overlapping names. The following names "
            f"are used in both states and actions: {states_and_actions_overlap}.",
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_distributed_grids(regime: lcm.regime.Regime) -> list[str]:
    """Reject `distributed=True` on action grids.

    Distribution shards the V-array along state axes; an action grid has no
    corresponding V-array axis, so marking one as distributed has no
    consistent meaning. To shard an axis a user cares about, set
    `distributed=True` on the matching state.
    """
    offending_actions = [
        name for name, grid in regime.actions.items() if grid.distributed
    ]
    if not offending_actions:
        return []
    return [
        "Action grids cannot be marked `distributed=True` — distribution "
        "shards V-array axes, which come from states. Move `distributed=True` "
        "to the corresponding state grid. Offending actions: "
        f"{offending_actions}.",
    ]


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
    function_inputs = _function_input_names(regime.functions)
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
    functions: Mapping[str, Callable | SolveSimulateFunctionPair | Phased],
) -> dict[str, set[str]]:
    """Return each regime function's input-parameter names.

    A `Phased` or `SolveSimulateFunctionPair` contributes the union of both
    variants' parameters; unintrospectable callables contribute the empty set.
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
    `SolveSimulateFunctionPair`; the regime transition contributes itself.
    """
    consumers: list[tuple[str, Callable]] = []
    for name, func in regime.functions.items():
        consumers.extend((name, variant) for variant in _function_variants(func))
    for name, constraint in regime.constraints.items():
        consumers.extend((name, variant) for variant in _function_variants(constraint))
    if callable(regime.transition):
        consumers.append(("regime_transition", regime.transition))
    return consumers


def _function_variants(
    func: Callable | SolveSimulateFunctionPair | Phased,
) -> tuple[Callable, ...]:
    """Return the callable variants of a regime-function entry.

    A plain function is itself; a `Phased` or `SolveSimulateFunctionPair`
    yields its `solve` and `simulate` callables so both phases are scanned.
    """
    if isinstance(func, SolveSimulateFunctionPair | Phased):
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


def _state_pair_names(regime: lcm.regime.Regime) -> set[StateName]:
    """Return the names of states declared as `SolveSimulateStatePair`."""
    return {
        name
        for name, grid in regime.states.items()
        if isinstance(grid, SolveSimulateStatePair)
    }


def _state_pair_field_errors(regime: lcm.regime.Regime) -> list[str]:
    """Validate every state pair's `solve`, `grid`, and `transition` fields.

    The pair contract:
    - `solve` is the derived function imputing the value during backward
      induction ⇒ must be callable.
    - `grid` is the simulate-phase domain of a carried per-subject value, not a
      solve dimension ⇒ must be an LCM grid, without `batch_size`/`distributed`
      (those knobs only apply to solve grid axes).
    - `transition` evolves the carried value each period ⇒ must be a plain
      callable; `MarkovTransition` is not supported for state pairs.
    - A terminal regime has no transitions, so it cannot carry a pair.
    """
    error_messages: list[str] = []
    pair_names: list[StateName] = []
    for name, spec in regime.states.items():
        if not isinstance(spec, SolveSimulateStatePair):
            continue
        pair_names.append(name)
        if not callable(spec.solve):
            error_messages.append(
                f"State pair '{name}': `solve` must be a callable, got {spec.solve!r}."
            )
        if isinstance(spec.transition, MarkovTransition):
            error_messages.append(
                f"State pair '{name}': `transition` must be a deterministic "
                f"callable — `MarkovTransition` is not supported for state "
                f"pairs."
            )
        elif not callable(spec.transition):
            error_messages.append(
                f"State pair '{name}': `transition` must be a callable, "
                f"got {spec.transition!r}."
            )
        if not isinstance(spec.grid, Grid):
            error_messages.append(
                f"State pair '{name}': `grid` must be an LCM grid, got {spec.grid!r}."
            )
        elif spec.grid.batch_size > 0 or spec.grid.distributed:
            error_messages.append(
                f"State pair '{name}': `grid` is the simulate-phase domain of "
                f"a carried per-subject value — `batch_size` and `distributed` "
                f"apply only to solve grid axes and must not be set on a "
                f"pair's grid."
            )
    if pair_names and regime.terminal:
        error_messages.append(
            f"Terminal regimes cannot carry SolveSimulateStatePair states "
            f"(no next period to carry {pair_names} into)."
        )
    return error_messages


def _state_pair_transition_errors(
    regime: lcm.regime.Regime, state_pair_names: set[StateName]
) -> list[str]:
    """Error if a state pair is also listed in `state_transitions`."""
    pairs_in_transitions = state_pair_names & set(regime.state_transitions)
    if pairs_in_transitions:
        return [
            "SolveSimulateStatePair states carry their own transition and must "
            f"not appear in state_transitions: {pairs_in_transitions}."
        ]
    return []


def _validate_state_transitions(regime: lcm.regime.Regime) -> list[str]:
    """Validate state_transitions against states."""
    error_messages: list[str] = []

    process_names: set[ProcessName] = {
        name
        for name, grid in regime.states.items()
        if isinstance(grid, _ContinuousStochasticProcess)
    }
    # Phase-variant state pairs carry their own transition (`pair.transition`),
    # so they neither belong in `state_transitions` nor count as missing one.
    state_pair_names = _state_pair_names(regime)
    error_messages.extend(_state_pair_transition_errors(regime, state_pair_names))
    non_process_names: set[StateName] = (
        set(regime.states) - process_names - state_pair_names
    )

    # Keys not in states are allowed only with actual transitions (not None).
    # None means identity, which requires the state to exist in this regime.
    extra_keys = set(regime.state_transitions) - set(regime.states)
    for key in extra_keys:
        value = regime.state_transitions[key]
        if value is None:
            error_messages.append(
                f"state_transitions['{key}'] is None but '{key}' is not in states. "
                "Identity transitions require the state to exist in this regime.",
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
            f"Missing: {missing}. Use None for fixed states.",
        )

    for name, value in regime.state_transitions.items():
        error_messages.extend(_state_transition_value_errors(name=name, value=value))

    return error_messages


def _state_transition_value_errors(*, name: StateName, value: object) -> list[str]:
    """Validate one `state_transitions` entry against the value vocabulary.

    Each variant of a `Phased` entry is held to the vocabulary of a bare
    value — callable, `None`, or a per-target Mapping — except that a
    stochastic (`MarkovTransition`) variant is rejected: per-phase
    stochasticity of a law of motion is not yet supported.
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
        if variant is None or callable(variant):
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
                f"MarkovTransition, None, or a per-target Mapping, got "
                f"{type(variant).__name__}.",
            )
    return error_messages


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
    for target_name, target_value in targets.items():
        if not isinstance(target_name, str):
            error_messages.append(
                f"state_transitions['{state_name}'] per-target dict key "
                f"{target_name!r} must be a string.",
            )
        if isinstance(target_value, Phased):
            error_messages.append(
                f"state_transitions['{state_name}']['{target_name}'] cannot "
                f"be `Phased` — `Phased` is outermost-only: wrap the whole "
                f"entry, e.g. `Phased(solve={{...}}, simulate={{...}})`.",
            )
        elif isinstance(target_value, MarkovTransition):
            markov_count += 1
        elif not callable(target_value):
            error_messages.append(
                f"state_transitions['{state_name}']['{target_name}'] must be "
                f"callable or MarkovTransition, got "
                f"{type(target_value).__name__}.",
            )
    if 0 < markov_count < len(targets):
        error_messages.append(
            f"state_transitions['{state_name}'] per-target dict must be "
            f"consistently stochastic: either all values are "
            f"MarkovTransition or none are.",
        )
    return error_messages
