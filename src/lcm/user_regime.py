import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAliasType, cast, overload

import jax.numpy as jnp
from beartype import beartype
from dags.tree import QNAME_DELIMITER

from lcm._beartype_conf import REGIME_CONF
from lcm.exceptions import RegimeInitializationError, format_messages
from lcm.grids import DiscreteGrid, Grid
from lcm.interfaces import SolveSimulateFunctionPair
from lcm.shocks._base import _ShockGrid
from lcm.typing import (
    ActionName,
    ActiveFunction,
    ContinuousState,
    DiscreteState,
    FloatND,
    FunctionName,
    RegimeName,
    ShockName,
    StateName,
    UserFunction,
)
from lcm.utils.ast_inspection import _get_func_indexing_params
from lcm.utils.containers import (
    ensure_containers_are_immutable,
)

# Genuine circular import: model.py imports from this module at module level.
# The `model` parameter of `validate_transition_probs` is annotated with the
# fully-qualified `lcm.model.Model` so the beartype claw resolves it by
# importing `lcm.model` at first call — long after the import cycle settles —
# rather than at module-init time. Importing `lcm.model` here keeps `lcm` a
# bound name for the type checker.
if TYPE_CHECKING:
    import lcm.model


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True)
class MarkovTransition:
    """Wrapper marking a transition function as stochastic (Markov).

    Wrap a transition function in `MarkovTransition` to indicate that it returns
    a probability distribution over next states (for state transitions) or over
    next regimes (for regime transitions), rather than a deterministic next value.

    Use at both the state and regime level:

        # Stochastic state transition (in Regime.state_transitions)
        state_transitions={"health": MarkovTransition(health_probs)}

        # Stochastic regime transition
        Regime(transition=MarkovTransition(regime_probs), ...)

    A bare callable (without the wrapper) is deterministic at both levels.

    """

    func: Callable[..., FloatND]
    """The transition function returning a probability distribution."""

    def __post_init__(self) -> None:
        # Copy __wrapped__ and __annotations__ from the wrapped function so
        # that inspect.signature and dags see the original signature. We use
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "__wrapped__", self.func)
        object.__setattr__(
            self, "__annotations__", getattr(self.func, "__annotations__", {})
        )

    def __call__(self, *args: Any, **kwargs: Any) -> FloatND:  # noqa: ANN401
        return self.func(*args, **kwargs)


def _default_H(
    utility: FloatND, E_next_V: FloatND, discount_factor: FloatND
) -> FloatND:
    return utility + discount_factor * E_next_V


class _IdentityTransition:
    """Identity transition function for fixed states.

    Used by `get_all_functions()` so the params template includes fixed states.
    The `_is_auto_identity` attribute lets validation distinguish auto-generated
    identities from user-provided transitions.

    """

    _is_auto_identity: bool = True

    def __init__(self, state_name: StateName, *, annotation: TypeAliasType) -> None:
        self._state_name = state_name
        self.__name__ = f"next_{state_name}"
        param = inspect.Parameter(
            state_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
        )
        self.__signature__ = inspect.Signature(
            [param],
            return_annotation=annotation,
        )
        self.__annotations__ = {state_name: annotation, "return": annotation}

    @overload
    def __call__(self, **kwargs: DiscreteState) -> DiscreteState: ...
    @overload
    def __call__(self, **kwargs: ContinuousState) -> ContinuousState: ...
    def __call__(
        self, **kwargs: DiscreteState | ContinuousState
    ) -> DiscreteState | ContinuousState:
        return kwargs[self._state_name]


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Regime:
    """User-facing regime definition.

    `Model` processes instances of this class into the canonical regime form
    (`lcm.interfaces.Regime`) used internally by the solver and simulator.

    State transitions are specified via `state_transitions`, mapping state names to
    transition functions. A bare callable is deterministic; wrap in `MarkovTransition`
    for stochastic transitions. `None` marks a fixed state (identity auto-generated).
    ShockGrids have intrinsic transitions and must not appear in `state_transitions`.

    The `transition` field on the regime itself is the *regime* transition function.
    A regime with `transition=None` is terminal — no separate `terminal` flag is
    needed.

    """

    transition: UserFunction | MarkovTransition | None
    """Regime transition function, or `None` for terminal regimes.

    A bare callable is deterministic. Wrap in `MarkovTransition` for stochastic
    regime transitions that return probability distributions.
    """

    active: ActiveFunction = lambda _age: True
    """Callable that takes age (float) and returns True if regime is active."""

    states: Mapping[StateName, Grid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of state variable names to grid objects."""

    state_transitions: Mapping[
        StateName,
        UserFunction
        | MarkovTransition
        | None
        | Mapping[RegimeName, UserFunction | MarkovTransition],
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of state names to transition functions, `None`, or per-target dicts.

    Every non-shock state must have an entry — omitting a state raises an error.
    `None` marks a fixed state (identity auto-generated internally). Wrap in
    `MarkovTransition` for stochastic transitions. Per-target dicts map target
    regime names to transition functions — every reachable target must be listed.
    """

    actions: Mapping[ActionName, Grid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of action variable names to grid objects."""

    functions: Mapping[FunctionName, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of function names to callables; must include 'utility'."""

    constraints: Mapping[FunctionName, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of constraint names to constraint functions."""

    derived_categoricals: Mapping[FunctionName, DiscreteGrid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Categorical grids for DAG function outputs not in states/actions."""

    description: str = ""
    """Description of the regime."""

    @property
    def terminal(self) -> bool:
        """Whether this is a terminal regime (derived from transition being None)."""
        return self.transition is None

    @property
    def stochastic_regime_transition(self) -> bool:
        """Whether the regime transition is stochastic (MarkovTransition)."""
        return isinstance(self.transition, MarkovTransition)

    def __post_init__(self) -> None:
        _validate_mapping_contents(self)
        _validate_logical_consistency(self)

        def make_immutable(name: str) -> None:
            value = ensure_containers_are_immutable(getattr(self, name))
            object.__setattr__(self, name, value)

        # Inject default aggregation function H if not provided by user.
        # Terminal regimes don't need H since Q = U directly (no E_next_V).
        if not self.terminal and "H" not in self.functions:
            object.__setattr__(self, "functions", {**self.functions, "H": _default_H})
        make_immutable("functions")
        make_immutable("states")
        make_immutable("state_transitions")
        make_immutable("actions")
        make_immutable("constraints")
        make_immutable("derived_categoricals")

    def get_all_functions(
        self,
        phase: Literal["solve", "simulate"] = "solve",
    ) -> MappingProxyType[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Collect functions from four sources:
        - `self.functions` (utility, helpers, H)
        - `self.constraints`
        - State transitions from `self.state_transitions`
        - The regime transition (`self.transition`, keyed as `"next_regime"`)

        For `SolveSimulateFunctionPair` entries, the variant matching `phase` is
        used.

        Args:
            phase: Which variant to use for `SolveSimulateFunctionPair` entries.

        Returns:
            Read-only mapping of all regime functions.

        """
        result: dict[str, UserFunction] = {}
        for name, func in self.functions.items():
            if isinstance(func, SolveSimulateFunctionPair):
                result[name] = cast(
                    "UserFunction",
                    func.solve if phase == "solve" else func.simulate,
                )
            else:
                result[name] = func
        result |= dict(self.constraints)
        if callable(self.transition):
            from lcm.regime_building.transitions import (  # noqa: PLC0415
                collect_state_transitions,
            )

            result |= collect_state_transitions(self.states, self.state_transitions)
            result["next_regime"] = self.transition
        return MappingProxyType(result)

    def replace(self, **kwargs: Any) -> Regime:  # noqa: ANN401
        """Replace the attributes of the regime.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the regime.

        Returns:
            A new regime with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise RegimeInitializationError(
                f"Failed to replace attributes of the regime. The error was: {e}"
            ) from e


def _validate_mapping_contents(regime: Regime) -> None:
    """Exhaustively check key/value types of `regime`'s mapping fields.

    Beartype on `Regime` catches top-level type mismatches and a sampled
    Mapping entry, but does not deep-check every key/value of a Mapping
    parameter — especially when the value type is a `Callable`/`Protocol`,
    which beartype skips entirely. This function fills that gap by
    iterating each entry and reporting all type violations together via
    the standard `error_messages` aggregator.

    """
    error_messages: list[str] = []

    for attr_name in ("states", "actions"):
        for k, v in getattr(regime, attr_name).items():
            if not isinstance(k, str):
                error_messages.append(f"{attr_name} key {k!r} must be a string.")
            if not isinstance(v, Grid):
                error_messages.append(f"{attr_name} value {v!r} must be an LCM grid.")

    for attr_name in ("functions", "constraints"):
        for k, v in getattr(regime, attr_name).items():
            if not isinstance(k, str):
                error_messages.append(f"{attr_name} key {k!r} must be a string.")
            if not callable(v) and not isinstance(v, SolveSimulateFunctionPair):
                error_messages.append(f"{attr_name} value {v!r} must be a callable.")

    if error_messages:
        msg = format_messages(error_messages)
        raise RegimeInitializationError(msg)


def _validate_logical_consistency(regime: Regime) -> None:
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


def _validate_distributed_grids(regime: Regime) -> list[str]:
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


def _validate_function_output_grid_indexing(regime: Regime) -> list[str]:
    """Detect the regime-function-output / discrete-grid-indexed-input name clash.

    The unsafe pattern is: a regime function `f` takes a discrete grid `g`
    (state, action, or derived categorical) as an input — so `f`'s output
    is a per-cell scalar — and a consumer then indexes `f[g]`. The
    consumer is indexing a 0-d array by a scalar integer, which raises
    `IndexError` at trace time. The fix is to drop the redundant `[g]`
    in the consumer (or refactor `f` not to take `g`).
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
    function_inputs: dict[str, set[str]] = {}
    for name, func in regime.functions.items():
        try:
            function_inputs[name] = set(inspect.signature(func).parameters)
        except ValueError, TypeError:
            function_inputs[name] = set()

    consumers: list[tuple[str, Callable]] = []
    consumers.extend(regime.functions.items())
    consumers.extend(regime.constraints.items())
    if callable(regime.transition):
        consumers.append(("regime_transition", regime.transition))

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


def _validate_state_transitions(regime: Regime) -> list[str]:
    """Validate state_transitions against states."""
    error_messages: list[str] = []

    shock_names: set[ShockName] = {
        name for name, grid in regime.states.items() if isinstance(grid, _ShockGrid)
    }
    non_shock_names: set[StateName] = set(regime.states) - shock_names

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

    shock_in_transitions = shock_names & set(regime.state_transitions)
    if shock_in_transitions:
        error_messages.append(
            f"ShockGrid states have intrinsic transitions and must not appear "
            f"in state_transitions: {shock_in_transitions}.",
        )

    if regime.terminal:
        if regime.state_transitions:
            error_messages.append(
                "Terminal regimes must have empty state_transitions.",
            )
        return error_messages

    missing = non_shock_names - set(regime.state_transitions)
    if missing:
        error_messages.append(
            f"Every non-shock state must have an entry in state_transitions. "
            f"Missing: {missing}. Use None for fixed states.",
        )

    for name, value in regime.state_transitions.items():
        if value is None or callable(value):
            continue
        if isinstance(value, Mapping):
            error_messages.extend(
                _validate_per_target_dict(state_name=name, targets=value)
            )
        else:
            error_messages.append(
                f"state_transitions['{name}'] must be callable, MarkovTransition, "
                f"None, or a per-target Mapping, got {type(value).__name__}.",
            )

    return error_messages


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
        if isinstance(target_value, MarkovTransition):
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


def validate_transition_probs(
    *,
    probs: FloatND,
    model: lcm.model.Model,
    regime_name: RegimeName,
    state_name: StateName,
    target_regime_name: RegimeName | None = None,
) -> None:
    """Validate a state transition probability array.

    Check that the array has the shape expected from the function signature,
    that all values are in [0, 1], that rows sum to 1, and that the function's
    `probs_array[…]` subscripts match the signature parameter order.

    For per-target state transitions (where `state_transitions[state_name]` is
    a dict mapping target regime names to `MarkovTransition` instances), pass
    `target_regime_name` to select the specific transition to validate.

    Regime transition probabilities are validated automatically before solve
    via `validate_regime_transitions_all_periods` in
    `regime_building/runtime_checks.py`; this helper covers only state
    transitions.

    Args:
        probs: The transition probability array to validate.
        model: The LCM Model instance.
        regime_name: Name of the regime.
        state_name: Name of the state with a `MarkovTransition`.
        target_regime_name: Target regime name for per-target state
            transitions. Required when the state transition is a per-target
            dict.

    Raises:
        TypeError: If the transition is not a `MarkovTransition`.
        ValueError: If the shape is wrong, values are outside [0, 1], or rows
            don't sum to 1.

    """
    regime = model.user_regimes[regime_name]
    raw_transition = regime.state_transitions[state_name]
    markov = _extract_markov_transition(
        raw_transition=raw_transition,
        state_name=state_name,
        regime_name=regime_name,
        target_regime_name=target_regime_name,
    )
    func = markov.func
    grids = _build_grids(regime)
    n_outcomes = len(grids[state_name].categories)

    indexing_params = _get_func_indexing_params(
        func=func, array_param_name="probs_array"
    )

    sig = inspect.signature(func)
    sig_order = [
        p for p in sig.parameters if p != "probs_array" and p in indexing_params
    ]
    if indexing_params != sig_order:
        func_name = getattr(func, "__name__", "<unknown>")
        msg = (
            f"In function '{func_name}', `probs_array` is indexed as "
            f"`probs_array[{', '.join(indexing_params)}]` but the signature "
            f"order is `probs_array[{', '.join(sig_order)}]`."
        )
        raise ValueError(msg)

    expected_shape = _build_expected_shape(
        indexing_params=indexing_params,
        n_outcomes=n_outcomes,
        grids=grids,
        model=model,
    )

    if probs.shape != expected_shape:
        msg = f"Expected shape {expected_shape} but got {probs.shape}."
        raise ValueError(msg)

    if jnp.any(probs < 0) or jnp.any(probs > 1):
        msg = "All values must be in [0, 1]."
        raise ValueError(msg)

    row_sums = jnp.sum(probs, axis=-1)
    if not jnp.allclose(row_sums, 1.0, atol=1e-6):
        msg = "Rows must sum to 1 along the last axis."
        raise ValueError(msg)


def _extract_markov_transition(
    *,
    raw_transition: object,
    state_name: StateName,
    regime_name: RegimeName,
    target_regime_name: RegimeName | None,
) -> MarkovTransition:
    """Extract a MarkovTransition from a raw transition, handling per-target dicts."""
    if isinstance(raw_transition, MarkovTransition):
        return raw_transition

    if isinstance(raw_transition, Mapping):
        if target_regime_name is None:
            targets = sorted(raw_transition.keys())
            msg = (
                f"State '{state_name}' in regime '{regime_name}' uses per-target "
                f"transitions. Pass target_regime_name to select one of: {targets}."
            )
            raise TypeError(msg)
        if target_regime_name not in raw_transition:
            msg = (
                f"Target regime '{target_regime_name}' not found in per-target "
                f"transitions for state '{state_name}' in regime '{regime_name}'. "
                f"Available targets: {sorted(raw_transition.keys())}."
            )
            raise ValueError(msg)
        entry = raw_transition[target_regime_name]  # ty: ignore[invalid-argument-type]
        if not isinstance(entry, MarkovTransition):
            msg = (
                f"Per-target transition for '{target_regime_name}' in state "
                f"'{state_name}' of regime '{regime_name}' is not a "
                f"MarkovTransition. Got {type(entry).__name__}."
            )
            raise TypeError(msg)
        return entry

    msg = (
        f"State '{state_name}' in regime '{regime_name}' is not a "
        f"MarkovTransition. Got {type(raw_transition).__name__}."
    )
    raise TypeError(msg)


def _build_grids(user_regime: Regime) -> dict[str, DiscreteGrid]:
    """Collect all DiscreteGrid instances from regime states and actions."""
    return {
        name: grid
        for name, grid in (*user_regime.states.items(), *user_regime.actions.items())
        if isinstance(grid, DiscreteGrid)
    }


def _build_expected_shape(
    *,
    indexing_params: list[str],
    n_outcomes: int,
    grids: dict[str, DiscreteGrid],
    model: lcm.model.Model,
) -> tuple[int, ...]:
    """Compute expected shape for a transition probability array."""
    shape: list[int] = []
    for param_name in indexing_params:
        if param_name == "period":
            shape.append(model.n_periods)
        elif param_name in grids:
            shape.append(len(grids[param_name].categories))
        else:
            msg = (
                f"Cannot determine expected size for parameter '{param_name}'. "
                f"It is not 'period' and not a DiscreteGrid state or action."
            )
            raise ValueError(msg)
    shape.append(n_outcomes)
    return tuple(shape)
