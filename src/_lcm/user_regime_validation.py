"""Validation helpers for the user-facing `Regime`.

These functions back `Regime.__post_init__`. They raise nothing themselves —
instead they collect error messages, which `__post_init__` aggregates into a
single `RegimeInitializationError`. Splitting the validators out of the
public module keeps `lcm.regime` to class definitions.

"""

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterator, Mapping
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
from lcm.transition import (
    AgeSpecializedFunction,
    AgeSpecializedGrid,
    MarkovTransition,
)

if TYPE_CHECKING:
    import lcm.regime


def _grid_mapping_errors(
    *,
    attr_name: str,
    mapping: Mapping[str, object],
    allow_phase_variants: bool,
    allow_age_specialized_grid: bool = False,
) -> list[str]:
    """Collect key/value type errors for a grid-valued mapping (states/actions)."""
    allowed: object = Grid | Phased if allow_phase_variants else Grid
    suffix = " or Phased" if allow_phase_variants else ""
    if allow_age_specialized_grid:
        # An age-varying continuous-state grid is a valid state (states only).
        allowed = allowed | AgeSpecializedGrid  # type: ignore[operator]
        suffix += " or AgeSpecializedGrid"
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
            attr_name="states",
            mapping=regime.states,
            allow_phase_variants=True,
            allow_age_specialized_grid=True,
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
    error_messages.extend(
        _age_specialized_scope_errors(
            transition=regime.transition,
            state_transitions=regime.state_transitions,
            functions=regime.functions,
            constraints=regime.constraints,
            terminal=regime.terminal,
        )
    )

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _iter_transition_nodes(value: object) -> Iterator[object]:
    """Yield leaf transition nodes, unwrapping `Phased` sides and per-target dicts."""
    if isinstance(value, Phased):
        yield from _iter_transition_nodes(value.solve)
        yield from _iter_transition_nodes(value.simulate)
    elif isinstance(value, Mapping):
        for cell in value.values():
            yield from _iter_transition_nodes(cell)
    else:
        yield value


def _state_transition_marker_errors(
    state_transitions: Mapping[str, object],
) -> list[str]:
    """Collect errors for `AgeSpecializedFunction` markers inside state transitions."""
    error_messages: list[str] = []
    for name, value in state_transitions.items():
        for node in _iter_transition_nodes(value):
            if isinstance(node, MarkovTransition) and isinstance(
                node.func, AgeSpecializedFunction
            ):
                error_messages.append(
                    f"state_transitions['{name}']: a `MarkovTransition` wrapping an "
                    f"`AgeSpecializedFunction` (a policy-specialized stochastic "
                    f"transition) is not supported.",
                )
            elif isinstance(node, AgeSpecializedFunction):
                error_messages.append(
                    f"state_transitions['{name}']: an `AgeSpecializedFunction` cannot "
                    f"be a state-transition value. Express the policy-dependent law of "
                    f"motion as a plain transition function that reads an "
                    f"`AgeSpecializedFunction` entry of `functions` instead.",
                )
    return error_messages


def _first_age_specialized_ancestor_of_transition(
    *,
    transition: object,
    functions: Mapping[str, object],
) -> str | None:
    """Return the name of an `AgeSpecializedFunction` the transition reads, if any.

    Walks the regime transition's parameter names transitively through plain
    entries of `functions` (unwrapping `Phased` sides, per-target dicts, and
    `MarkovTransition` wrappers along the way). Returns the first specialized
    function name reached, or `None` when the transition's dependency graph is
    policy-free.
    """
    specialized_names = {
        name
        for name, value in functions.items()
        if any(
            isinstance(node, AgeSpecializedFunction)
            for node in _iter_transition_nodes(value)
        )
    }
    if transition is None or not specialized_names:
        return None

    def arg_names_of(value: object) -> list[str]:
        names: list[str] = []
        for node in _iter_transition_nodes(value):
            func = node.func if isinstance(node, MarkovTransition) else node
            if callable(func):
                try:
                    names.extend(inspect.signature(func).parameters)
                except TypeError, ValueError:
                    continue
        return names

    seen: set[str] = set()
    stack = arg_names_of(transition)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in specialized_names:
            return name
        if name in functions:
            stack.extend(arg_names_of(functions[name]))
    return None


def _age_specialized_scope_errors(
    *,
    transition: object,
    state_transitions: Mapping[str, object],
    functions: Mapping[str, object],
    constraints: Mapping[str, object],
    terminal: bool,
) -> list[str]:
    """Reject the `AgeSpecializedFunction` compositions that are out of scope.

    `AgeSpecializedFunction` is supported in `functions` and `constraints` of
    non-terminal regimes only. Rejected — loudly, before any per-period program
    is built:

    - a regime `transition` that is (or contains) an `AgeSpecializedFunction` — a
      policy-specialized *regime* transition;
    - a `MarkovTransition` wrapping an `AgeSpecializedFunction` in a state
      transition — a policy-specialized *stochastic* transition;
    - an `AgeSpecializedFunction` directly as a state-transition value — express the
      policy-dependent law of motion as a plain transition reading an
      `AgeSpecializedFunction` helper function instead;
    - a regime transition whose dependency graph reads an `AgeSpecializedFunction`
      function (directly or through plain helper functions) — regime-transition
      probabilities are built once, not per period, so a policy-specialized
      value flowing into them would reuse one age's policy closure everywhere;
    - any `AgeSpecializedFunction` in a terminal regime — the terminal value program is
      built once and shared across all periods.
    """
    error_messages: list[str] = []

    if any(
        isinstance(node, AgeSpecializedFunction)
        or (
            isinstance(node, MarkovTransition)
            and isinstance(node.func, AgeSpecializedFunction)
        )
        for node in _iter_transition_nodes(transition)
    ):
        error_messages.append(
            "A regime `transition` cannot be `AgeSpecializedFunction` (bare or "
            "wrapped in `MarkovTransition`): policy-specialized regime transitions "
            "are not "
            "supported. Specialize `functions` or `constraints` instead.",
        )

    specialized_ancestor = _first_age_specialized_ancestor_of_transition(
        transition=transition, functions=functions
    )
    if specialized_ancestor is not None:
        error_messages.append(
            f"The regime `transition` depends on the `AgeSpecializedFunction` function "
            f"'{specialized_ancestor}'. Regime-transition probabilities are built "
            f"once, not per period, so a policy-specialized value flowing into "
            f"them would silently reuse one age's policy closure across all "
            f"periods. Route the policy dependency through `functions` consumed "
            f"by utility, constraints, or state transitions instead.",
        )

    error_messages.extend(_state_transition_marker_errors(state_transitions))

    if terminal:
        for slot_name, slot in (("functions", functions), ("constraints", constraints)):
            for name, value in slot.items():
                if any(
                    isinstance(node, AgeSpecializedFunction)
                    for node in _iter_transition_nodes(value)
                ):
                    error_messages.append(
                        f"{slot_name}['{name}']: `AgeSpecializedFunction` is not "
                        f"supported in a terminal regime — the terminal value "
                        f"program is built once and shared across all periods.",
                    )

    return error_messages


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
        error_messages.extend(
            _state_transition_value_errors(name=name, value=value, regime=regime)
        )
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


def _phased_per_target_shape_mismatch(
    *, name: StateName, value: Phased, regime: lcm.regime.Regime
) -> list[str]:
    """Inside `Phased`, constrain per-target/bare combinations of the two variants.

    `Phased(solve={...}, simulate={...})` is normalized at collection into one entry per
    target, each holding a `Phased` of that target's two laws — the form the engine
    actually consumes (`transitions._add_raw_transition`).

    Accepted shapes:

    - **both bare** — one coarse `next_<state>` node (a plain phased law), including a
      PARAMETERIZED coarse law: both phases bind the same single node, so its parameter
      is one shared leaf.
    - **both per-target** over the SAME targets — paired cell by cell.
    - **PARAMETER-FREE map-vs-bare** — one side per-target, the other a bare law with no
      free parameter. The bare law broadcasts over the per-target side's targets (the
      same meaning a bare state law has outside `Phased`); with no parameter it carries
      no template leaf, so its broadcast cells merge by object identity and nothing is
      replicated. The per-phase provenance stamp
      (`processing._phase_coarse_state_law_names` → `_rename_params_to_qnames`) keys a
      within-period read's merge/conflict off each phase's OWN declaration shape.

    Rejected shapes:

    - **two per-target dicts over DIFFERENT targets** — a target would carry a law in
      one phase and none in the other, with no single authoritative key set.
    - **PARAMETERIZED map-vs-bare** — a bare (coarse) side carrying a free parameter
      opposite a per-target dict. The phase-union params template replicates that
      parameter into one leaf per target, but the coarse side binds a single law: the
      build merges its per-target cells and silently DROPS all but the first leaf, so
      the extra leaves are dead and a user setting them differently is ignored. Rather
      than expose that trap, require the parameterized coarse law to be spelled as
      **both-bare** `Phased` (coarse in both phases — one shared leaf) or as an explicit
      **per-target dict** on this side (one honest leaf per target). Only the bare
      side's spelling is constrained; the per-target side is unaffected.

    (`Phased` is outermost-only — `_validate_per_target_dict` rejects a `Phased` cell —
    so the outer form is the only spelling for a per-target law that varies by phase.)
    """
    solve_per_target = isinstance(value.solve, Mapping)
    simulate_per_target = isinstance(value.simulate, Mapping)
    if solve_per_target and simulate_per_target:
        solve_targets = set(cast("Mapping[RegimeName, object]", value.solve))
        simulate_targets = set(cast("Mapping[RegimeName, object]", value.simulate))
        if solve_targets != simulate_targets:
            return [
                f"state_transitions['{name}']: the per-target dicts inside `Phased` "
                f"declare different targets — solve has {sorted(solve_targets)}, "
                f"simulate has {sorted(simulate_targets)}. Both phases must cover the "
                f"same targets.",
            ]
        return []
    if solve_per_target == simulate_per_target:
        # Both bare: one coarse node (parameterized or not) — nothing to reject.
        return []
    # Map-vs-bare: one side per-target, the other bare. The bare side broadcasts; a
    # free parameter on it would be replicated per target with only the first leaf live.
    bare_side, phase_label = (
        (value.simulate, "simulate") if solve_per_target else (value.solve, "solve")
    )
    if _law_has_free_parameter(bare_side, regime):
        return [
            f"state_transitions['{name}']: the {phase_label} variant is a bare "
            f"(coarse) law with a free parameter, opposite a per-target dict "
            f"(map-vs-bare). A parameterized coarse law broadcast over targets would "
            f"have its parameter replicated per target with only one binding live, so "
            f"this shape is not supported. Spell it as a both-bare `Phased` (coarse in "
            f"both phases, one shared parameter) or as an explicit per-target dict on "
            f"the {phase_label} side (one parameter per target).",
        ]
    return []


def _law_has_free_parameter(law: object, regime: lcm.regime.Regime) -> bool:
    """Whether a bare state-transition law reads a free parameter (a template leaf).

    A free parameter is any argument that is not a state, action, `next_<state>` node,
    or reserved name (`period`, `age`, `E_next_V`) — i.e., an argument that would appear
    in the params template. `fixed_transition` identities and any callable whose
    signature cannot be read are treated as parameter-free (nothing to replicate).
    """
    if isinstance(law, _IdentityTransition) or not callable(law):
        return False
    try:
        arg_names = set(inspect.signature(law).parameters)
    except TypeError, ValueError:
        return False
    variables = (
        set(regime.states)
        | set(regime.actions)
        | {f"next_{state_name}" for state_name in regime.states}
        | {f"next_{state_name}" for state_name in regime.state_transitions}
        | {"period", "age", "E_next_V"}
    )
    return bool(arg_names - variables)


def _state_transition_value_errors(
    *, name: StateName, value: object, regime: lcm.regime.Regime
) -> list[str]:
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

    The two variants may be both bare (one coarse node), both per-target dicts over the
    SAME targets, or a map-vs-bare mix (per-target on one side, a bare law that
    broadcasts on the other). Only two per-target dicts over DIFFERENT target sets are
    rejected, as an ambiguous normalization — see `_phased_per_target_shape_mismatch`.

    `None` is not a law of motion; the error points to `fixed_transition`.
    """
    error_messages: list[str] = []
    phase_variant = isinstance(value, Phased)
    if phase_variant:
        error_messages.extend(
            _phased_per_target_shape_mismatch(name=name, value=value, regime=regime)
        )
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
