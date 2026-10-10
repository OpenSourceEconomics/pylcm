"""User-facing transition vocabulary: `fixed_transition`, `StochasticTransition`,
`DeterministicTransition`, `Transition`, `ByAge`, `AgeRange`, `JointTransition`,
`AgeSpecializedFunction`, and `AgeSpecializedGrid`.

A thin leaf module with no dependency on `Regime`, the validators, or the
regime-building code. Keeping the vocabulary here lets the user-facing
`Regime`, the engine-internal regime-building code, and the regime validators
all import it without an import cycle.

"""

import dataclasses
import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self, cast

import jax
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.grids.continuous import ContinuousGrid
from _lcm.identity_transition import _IdentityTransition
from _lcm.time import ModelTime, TimeAxis, coordinate_kind
from _lcm.typing import StateName
from lcm.collective import Gate
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.typing import FloatND, RegimeName, UserAge, UserFunction


def fixed_transition(state_name: StateName) -> UserFunction:
    """Create the law of motion for a fixed state: next value = current value.

    The returned callable is an ordinary deterministic law, so it is legal
    wherever a law of motion is — as a bare `state_transitions` entry, inside
    a `Phased` side, and inside a per-target dict.

    Args:
        state_name: Name of the fixed state. Must match the
            `state_transitions` key the law is assigned to.

    Returns:
        The identity law of motion for `state_name`.

    """
    return _IdentityTransition(state_name=state_name)


@dataclass(frozen=True, kw_only=True)
class AgeRange:
    """The existing grid ages in `[start, exclusive_stop)`.

    Either bound may be omitted. The bounds need not be grid points; the
    selection is always a subset of the model's ages.
    """

    start: UserAge | float | None = None
    """Inclusive lower bound, or `None` for the first age."""

    exclusive_stop: UserAge | float | None = None
    """Exclusive upper bound, or `None` for beyond the last age."""


@dataclass(frozen=True, kw_only=True)
class PeriodRange:
    """The computational periods in `[start, exclusive_stop)`."""

    start: int | None = None
    """Inclusive first period, or `None` for the horizon's start."""

    exclusive_stop: int | None = None
    """Exclusive last period, or `None` for the horizon's end."""

    def __post_init__(self) -> None:
        for value in (self.start, self.exclusive_stop):
            if value is not None and type(value) is not int:
                raise RegimeInitializationError("PeriodRange bounds must be integers.")
        if (
            self.start is not None
            and self.exclusive_stop is not None
            and self.start >= self.exclusive_stop
        ):
            raise RegimeInitializationError(
                "PeriodRange start must be below exclusive_stop."
            )


@dataclass(frozen=True, kw_only=True)
class Periods:
    """Explicit computational period positions in a graph declaration."""

    values: tuple[int, ...]
    """Zero-based period positions; no age-to-period interpretation is inferred."""

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in self.values):
            raise RegimeInitializationError("Periods values must be integers.")


type PeriodSelector = PeriodRange | Periods
type AgeSelector = UserAge | float | tuple[UserAge | float, ...] | range | AgeRange

# One phase's edges: each source regime maps to its destinations' source-age
# selectors, or to a `Transition` whose law chooses among them.
type PhaseEdges = Mapping[
    RegimeName, Transition | Mapping[RegimeName, AgeSelector | PeriodSelector]
]

# What `Model(edges=...)` takes: one phase's edges for both phases, or a
# `Phased` pair of them.
type ModelEdges = PhaseEdges | Phased[PhaseEdges, PhaseEdges]

# One target's cell of a per-target law: its probability, or a `Phased` pair.
type TargetLawCell = (
    StochasticTransition
    | UserFunction
    | Phased[StochasticTransition | UserFunction, StochasticTransition | UserFunction]
)

# A law one phase evaluates at a source age.
type PhaseTransitionLaw = (
    RegimeName
    | DeterministicTransition
    | StochasticTransition
    | UserFunction
    | Mapping[RegimeName, TargetLawCell]
)

# A law one `ByAge` case selects: one phase's law for both phases, or a
# `Phased` pair of them.
type AgeCaseLaw = PhaseTransitionLaw | Phased[PhaseTransitionLaw, PhaseTransitionLaw]

# What `Transition(law=...)` takes: a case law, or a `ByAge` selecting among
# case laws per source age.
type TransitionLaw = AgeCaseLaw | ByAge

if TYPE_CHECKING:
    type _DeclaredCaseLaw = AgeCaseLaw
    type _DeclaredTransitionLaw = TransitionLaw
else:
    # The runtime checks also admit `None` and a nested `ByAge`, so that
    # `Transition` and `ByAge` refuse them with their own messages rather than
    # with a type violation.
    type _DeclaredCaseLaw = AgeCaseLaw | ByAge | None
    type _DeclaredTransitionLaw = TransitionLaw | None


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Transition:
    """A source's graph edges together with the law that chooses among them.

    `Model(edges=...)` maps a source regime either to a plain
    `{target: age_selector}` mapping or to a `Transition`. The plain mapping is
    enough while the source has exactly one outgoing edge at every age: the
    graph is then the law. Wherever a source age has more than one outgoing
    edge, the source is declared as a `Transition` whose `law` chooses among
    them:

    - a per-target mapping of `StochasticTransition` probabilities, keyed by
      target;
    - a plain function or `DeterministicTransition` returning a global regime
      code, which is how a discrete choice between regimes is written;
    - a `StochasticTransition` returning the full regime-code probability vector;
    - a regime name;
    - `ByAge(...)` selecting one of the above per source age, or `Phased(...)`
      giving each phase its own.

        edges = {
            "working": Transition(
                law=ByAge(
                    cases={
                        (60, 61): {"working": survive, "dead": die},
                        62: {"retired": certain},
                    }
                ),
            ),
            "retired": {"dead": (63, 64)},
        }

    A law is evaluated at every source age with outgoing edges, also where only
    one edge leaves the source, and there it must put unit mass on that edge.
    With declared `targets`, a `ByAge` law need not select ages with a single
    outgoing edge; the edge is the law there. It must select every age with more
    than one. With derived targets, an age no case selects has no edge.

    `gates` makes the transition into a target value-dependent: the law still
    supplies the probability of reaching it, and the target's `Gate` decides
    whether a row stays there or takes its route's fallback.
    """

    targets: Mapping[RegimeName, AgeSelector | PeriodSelector] | None = None
    """Destination regimes and the source ages at which each edge fires.

    Optional when the law names its targets — a per-target mapping, a regime
    name, or a `ByAge` / `Phased` of those. The destinations are then read off
    the law: each key of a case is reached at the non-final ages that case
    covers, and each route fallback of a gate wherever its gated target is.
    Supplied anyway, it must equal what the law names. A law over all targets
    names none, so it requires `targets`.
    """

    law: _DeclaredTransitionLaw
    """The numerical law choosing among the destinations."""

    gates: Mapping[RegimeName, Gate] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """One `Gate` per value-dependent destination, keyed by that destination."""

    def __post_init__(self) -> None:
        if self.law is None:
            raise RegimeInitializationError(
                "`Transition.law` cannot be `None`. A regime with no outgoing "
                "edges is terminal; leave it out of `Model(edges=...)`."
            )
        if self.targets is None:
            if not law_names_its_targets(self.law):
                raise RegimeInitializationError(
                    "`Transition.targets` is required when the law does not name "
                    "its targets: a function, `DeterministicTransition` or "
                    "vector `StochasticTransition` chooses among regime codes, "
                    "so declare the destinations and their source ages; got "
                    f"law={self.law!r}."
                )
        elif not self.targets:
            raise RegimeInitializationError(
                "`Transition.targets` must be a nonempty mapping from destination "
                f"regimes to source-age selectors; got {self.targets!r}."
            )
        else:
            object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        object.__setattr__(self, "law", snapshot_transition_containers(self.law))
        object.__setattr__(self, "gates", MappingProxyType(dict(self.gates)))


def snapshot_transition_containers(value: object) -> object:
    """Copy edge and law mappings, including phase variants, preserving callables.

    A mapping proxy may still view a caller-owned dictionary, so it also needs
    a copy. Transition and ByAge declarations own their containers at construction.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: snapshot_transition_containers(item) for key, item in value.items()}
        )
    if isinstance(value, Phased):
        return Phased(
            solve=snapshot_transition_containers(value.solve),
            simulate=snapshot_transition_containers(value.simulate),
        )
    return value


def law_names_its_targets(law: object) -> bool:
    """Whether every case and phase of `law` is a per-target mapping or a name.

    Args:
        law: A `Transition` law.

    Returns:
        Whether the law's destinations can be read off its declaration.

    """
    if isinstance(law, ByAge):
        return all(law_names_its_targets(case) for case in law.laws)
    if isinstance(law, Phased):
        return law_names_its_targets(law.solve) and law_names_its_targets(law.simulate)
    return isinstance(law, Mapping | str)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class StochasticTransition:
    """Wrapper marking a transition function as stochastic (Markov).

    Wrap a transition function in `StochasticTransition` to indicate that it returns
    a probability distribution over next states (for state transitions) or over
    next regimes (for regime transitions), rather than a deterministic next value.

    Use at both the state and regime level:

        # Stochastic state transition (in Regime.state_transitions)
        state_transitions={"health": StochasticTransition(func=health_probs)}

        # Stochastic regime transition over the full regime-ID vector
        edges = {
            "working": Transition(
                targets={"working": ages, "dead": ages},
                law=StochasticTransition(func=regime_probs),
            ),
        }

    A bare callable (without the wrapper) is a deterministic state law.

    At the regime level, `Model.edges` declares structural support separately.
    Every returned vector entry outside the current graph support must be zero.
    A probability that is zero at runtime does not narrow the graph.

    """

    func: Callable[..., FloatND]
    """The transition function returning a probability distribution."""

    fixed_component: tuple[int, ...] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """For a state transition: the value of a component the law never changes, per code.

    `fixed_component[code]` names the group a state code belongs to; the law must give
    probability zero to every target outside the current code's group. `Model` then
    carries the group as its own identity-law state and sums the continuation over one
    group only. Every group must have the same number of codes.
    """

    def __post_init__(self) -> None:
        # Copy __wrapped__ and __annotations__ from the wrapped function so
        # that inspect.signature and dags see the original signature. We use
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "__wrapped__", self.func)
        object.__setattr__(
            self, "__annotations__", getattr(self.func, "__annotations__", {})
        )

    def __call__(self, *args: Any, **kwargs: Any) -> FloatND:
        return self.func(*args, **kwargs)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class DeterministicTransition:
    """Mark a deterministic state or regime transition.

    A state law returns its next value, just like a plain state function.
    A regime selector returns an existing global regime code. `Model.edges`
    declares every regime it may select. Evaluation adds no random draw.

        edges = {
            "working": Transition(
                targets={"working": ages, "retired": ages},
                law=DeterministicTransition(func=next_regime),
            ),
        }

    Returning a code outside the model graph is an error; an edge remains
    declared even when its target is never selected at runtime.
    """

    func: Callable[..., Any]
    """The selector returning a global regime code."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "__wrapped__", self.func)
        object.__setattr__(
            self, "__annotations__", getattr(self.func, "__annotations__", {})
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


@beartype(conf=REGIME_CONF)
def deterministic_transition() -> Callable[[UserFunction], DeterministicTransition]:
    """Create a deterministic-law decorator preserving the DAG signature."""

    def decorate(func: UserFunction) -> DeterministicTransition:
        return DeterministicTransition(func=func)

    return decorate


@beartype(conf=REGIME_CONF)
def stochastic_transition(
    *,
    fixed_component: tuple[int, ...] | None = None,
) -> Callable[[Callable[..., FloatND]], StochasticTransition]:
    """Create a stochastic-law decorator preserving the DAG signature.

    Args:
        fixed_component: Fixed categorical groups of a stochastic state law.

    Returns:
        A decorator preserving the probability function's DAG signature.
    """

    def decorate(func: Callable[..., FloatND]) -> StochasticTransition:
        return StochasticTransition(func=func, fixed_component=fixed_component)

    return decorate


def _freeze_joint_support(value: Any) -> Any:
    """Freeze the container structure of a literal joint-support pytree."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_joint_support(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_joint_support(item) for item in value)
    return value


def _literal_joint_support_schema(
    support: Any,
) -> tuple[object, tuple[tuple[tuple[int, ...], object], ...]] | None:
    """Return a literal support's pytree and leaf event-shape/dtype schema."""
    if callable(support):
        return None
    leaves, tree = jax.tree_util.tree_flatten(support)
    schema = tuple(
        (tuple(leaf.shape[1:]), leaf.dtype)
        for leaf in leaves
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype")
    )
    return tree, schema


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class JointTransition:
    """One finite-support lottery shared by one or more target-state laws.

    A source regime owns joint transitions on an explicitly named target edge.
    ``support`` is either a literal JAX pytree whose leaves have leading axis
    ``support_size``, or a callable returning such a pytree. ``probabilities``
    returns the matching probability vector. Each entry of ``outputs`` is a
    target-state law; its function can read the sampled support node through the
    joint-transition mapping key declared on :class:`lcm.Regime`.

    Transition-local joint lotteries are currently supported only by
    `GridSearch`. A terminal regime may not declare one. Each output owns one
    target-state producer cell. A bare ordinary state law may coexist and
    broadcasts only to other unclaimed targets; an explicit law or another
    joint kernel on the same target-state cell is rejected.

    The declaration only specifies the joint lottery. The enclosing mapping
    supplies both its target regime and the local node name::

        joint_transitions={
            "couple": {
                "partner_match": JointTransition(
                    support_size=2,
                    support={"wage": wages, "health": health_codes},
                    probabilities=match_probabilities,
                    outputs={"wage": next_wage, "health": next_health},
                )
            }
        }

    `Phased` may wrap the whole `JointTransition`. Its solve and simulation
    variants must keep identical output names and `support_size`; literal
    supports must also keep one pytree structure, leaf event shapes, and
    dtypes. Support values, probability functions, and output-law bodies may
    differ. Callable supports may change values, but their schema is checked
    across every active period and both phases after parameters are bound.
    Probability rows are validated on the applicable phase grids, including
    carried-only simulation states, and must be finite, in `[0, 1]`, and sum to
    one within the transition tolerance.
    """

    support_size: int
    """Number of nodes on the joint finite support."""

    support: Any
    """Literal support pytree, or callable returning one."""

    probabilities: Callable[..., FloatND]
    """Function returning probabilities along the joint-support axis."""

    outputs: Mapping[StateName, UserFunction]
    """Target-state laws sharing the sampled support node."""

    def __post_init__(self) -> None:
        if self.support_size < 1:
            raise RegimeInitializationError(
                "`JointTransition.support_size` must be a positive integer."
            )
        if not self.outputs:
            raise RegimeInitializationError(
                "`JointTransition.outputs` must contain at least one target-state "
                "output law."
            )
        invalid_outputs = [
            name
            for name, output in self.outputs.items()
            if not isinstance(name, str) or not callable(output)
        ]
        if invalid_outputs:
            raise RegimeInitializationError(
                "Every `JointTransition.outputs` key must be a state-name string "
                "and every value must be callable; invalid output(s): "
                f"{invalid_outputs}."
            )

        if not callable(self.support):
            leaves, _ = jax.tree_util.tree_flatten(self.support)
            if not leaves:
                raise RegimeInitializationError(
                    "`JointTransition.support` must contain at least one support leaf."
                )
            invalid_shapes = [
                getattr(leaf, "shape", None)
                for leaf in leaves
                if not hasattr(leaf, "shape")
                or not leaf.shape
                or leaf.shape[0] != self.support_size
            ]
            if invalid_shapes:
                raise RegimeInitializationError(
                    "Every literal `JointTransition.support` leaf must have leading "
                    f"axis `support_size={self.support_size}`; got invalid leaf "
                    f"shape(s) {invalid_shapes}."
                )
            try:
                nonfinite_leaves = [
                    index
                    for index, leaf in enumerate(leaves)
                    if not bool(jax.numpy.all(jax.numpy.isfinite(leaf)))
                ]
            except TypeError:
                nonfinite_leaves = list(range(len(leaves)))
            if nonfinite_leaves:
                raise RegimeInitializationError(
                    "Every literal `JointTransition.support` leaf must contain "
                    "only finite numeric or boolean values; nonfinite or "
                    f"unsupported leaf index(es): {nonfinite_leaves}."
                )

        object.__setattr__(self, "support", _freeze_joint_support(self.support))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))


@dataclass(frozen=True)
class _AgeSpecialized:
    """Base for the age-specialized build-time markers.

    An age-specialized marker binds a per-age object (a function or a grid) at model
    build: pylcm calls `build(age)` for each period's age to obtain that period's
    concrete object, so ages resolving to the same object share a single compiled
    program.

    **`build(age)` must be deterministic and side-effect-free.** The same age is
    resolved more than once (validation, the representative regime, and the per-period
    map), so a stateful factory could be validated as one grid and installed as
    another. Repeated calls for one age must return behaviourally identical objects.

    **`signature(age)` is the dedup key for both markers**, but they differ in how
    an equal signature is trusted:
    - `AgeSpecializedFunction` — a function's closure cannot be inspected, so an
      equal signature must imply an identical resolved closure on the author's word;
      pylcm has no way to check it.
    - `AgeSpecializedGrid` — a grid can be asked what it actually is, so an equal
      signature is checked against the resolved nodes at build time: if two periods
      share a signature but their grids genuinely differ, construction raises
      instead of silently sharing the wrong grid.

    The two concrete markers are `AgeSpecializedFunction` (a function whose closure
    varies with age) and `AgeSpecializedGrid` (a continuous-state grid whose
    bounds/nodes vary with age at a fixed shape). A marker is resolved before
    it is used, so calling it directly is a loud error.
    """

    build: Callable[[float], Any]
    """Factory returning the concrete object (function or grid) for a given age. Must be
    deterministic and side-effect-free; it is called more than once per age."""

    signature: Callable[[float], Hashable]
    """Hashable identity of the age's object. The dedup key for both markers — a
    correctness precondition, trusted on the author's word, for
    `AgeSpecializedFunction`; cross-checked against the resolved nodes at build
    time for `AgeSpecializedGrid`. See the class docstrings."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        msg = (
            f"{type(self).__name__} is a build-time marker and must be resolved to a "
            "concrete object via build(age) before it is used."
        )
        raise TypeError(msg)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True)
class AgeSpecializedFunction(_AgeSpecialized):
    """Wrapper marking a function whose closure is bound per age at build time.

    Wrap a function *factory* to indicate that its closure depends on the agent's
    age — for example a tax-transfer system pinned to a policy date that moves with
    calendar time as the agent ages. At build time pylcm calls `build(age)` for each
    period's age to obtain that period's concrete function, and uses `signature(age)`
    as a dedup key so ages resolving to the same closure share a single compiled
    program.

    Usable in `functions` and `constraints` of non-terminal regimes. A
    policy-dependent law of motion is expressed as a plain state transition that
    reads an `AgeSpecializedFunction` entry of `functions`. A direct
    `AgeSpecializedFunction` state-transition value and a
    `StochasticTransition(func=AgeSpecializedFunction(...))` state transition are
    rejected at `Regime` construction. A specialized regime transition law, a
    regime transition whose dependency graph reads an `AgeSpecializedFunction`,
    and any `AgeSpecializedFunction` in a terminal regime are rejected when the
    model binds each regime's law from `Model(edges=...)`. Every concrete function
    returned by `build` must expose the same
    call signature — only the constants it closes over may differ across ages.

        functions={"tax": AgeSpecializedFunction(build=make_tax, signature=policy_key)}

    `signature` is a **correctness precondition**, not a performance hint: ages
    with equal signatures share one compiled program, so an equal signature must
    imply identical closure behavior (policy date, price level, overrides, and
    every other closed-over constant). An incomplete signature silently shares a
    wrong program across ages.

    A bare callable (without the wrapper) is age-invariant, as before.
    `AgeSpecializedFunction` is a build-time marker: it is resolved to a concrete
    function via `build(age)` before the DAG is traced, so calling it directly is an
    error.
    """

    build: Callable[[float], UserFunction]
    """Factory returning the concrete function for a given age."""

    signature: Callable[[float], Hashable]
    """Returns a hashable identity of the age's closure; used as the dedup key."""


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True)
class AgeSpecializedGrid(_AgeSpecialized):
    """Wrapper marking a continuous-state grid whose bounds vary per age.

    Wrap a grid *factory* to indicate that the grid's bounds/nodes depend on the
    agent's age — the canonical case is an asset state with an age-dependent
    borrowing floor `a̲(age)`. At build time pylcm calls `build(age)` for each of the
    owning regime's active ages to obtain that period's concrete `ContinuousGrid`;
    ages resolving to the same grid share a single compiled program.

        states={"assets": AgeSpecializedGrid(
            build=lambda age: LinSpacedGrid(start=floor(age), stop=A_MAX, n_points=40),
            signature=lambda age: floor(age))}

    **Shape-invariance contract (validated at construction):** across the owning
    regime's active ages, every `build(age)` must return the *same grid class*, with
    the same `batch_size`, the same points mode (concrete vs supplied at runtime), and —
    for concrete grids — the same resolved **node-array shape and dtype**. Only the
    bounds (start/stop) or node *values* may vary with age. This keeps every period's
    value array the same shape *and* keeps one compiled kernel valid for every period:
    pylcm lowers a shared kernel against a representative axis and then feeds it each
    period's axis, so a differing shape or dtype would be rejected by the compiled
    executable. Concrete grids are validated on their resolved `to_jax()` array (the
    same source of truth used for dedup), and any declared `n_points` must agree with
    it. Allowed only for continuous states (not actions, discrete states, or process
    states in this version). A builder may be undefined (raise) outside its regime's
    active ages; it is never called there.

    `signature` is the dedup key, same as `AgeSpecializedFunction`: ages with equal
    signatures share one compiled program. Unlike a function's closure, a grid can
    be asked what it actually is, so an equal signature is not blindly trusted —
    the resolved nodes of every period in a shared group are cross-checked at
    build time, and a genuine mismatch raises `RegimeInitializationError` instead
    of silently sharing the wrong grid.

    **Grid bounds are interpolation *support*, not hard feasibility limits.** The
    continuation value `V_{t+1}` is interpolated on period `t+1`'s grid; pylcm's
    interpolation extrapolates linearly beyond the grid rather than rejecting
    out-of-support points. So a period-`t` action whose next state lands *below* a
    tighter `t+1` floor (or above the ceiling) is evaluated by extrapolation, not
    excluded. **The model must therefore keep every feasible transition within the next
    period's grid** — either the grid bounds coincide with the true feasibility limits,
    or an explicit constraint keeps next states in range. The canonical borrowing-floor
    use satisfies this by construction: the feasibility constraint enforces
    `a_{t+1} ≥ a̲(t)` and period `t+1`'s grid floor is exactly `a̲(t)`, so every feasible
    `a_{t+1}` lies in support. Combining extrapolation with `-inf` edge values can
    otherwise produce `NaN`; if your bounds are *not* the feasibility limits, add a
    constraint (or widen the grid) so no reachable next state falls outside period
    `t+1`'s support.
    """

    build: Callable[[float], ContinuousGrid]
    """Factory returning the concrete continuous grid for a given age."""

    signature: Callable[[float], Hashable]
    """Returns a hashable identity of the age's grid; the dedup key, cross-checked
    against the resolved nodes at build time."""


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True)
class PeriodSpecializedFunction(AgeSpecializedFunction):
    """Bind a function at an integer computational period.

    ``build(period)`` and ``signature(period)`` follow the same determinism,
    signature and deduplication contracts as ``AgeSpecializedFunction``.
    This declaration is valid only in a period model.
    """

    # Coordinate-kind validation prevents using period callbacks in age models.
    build: Callable[[int], UserFunction]
    """Factory returning the concrete function for an integer period."""

    signature: Callable[[int], Hashable]
    """Hashable identity of the period's closure."""


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True)
class PeriodSpecializedGrid(AgeSpecializedGrid):
    """Bind a continuous-state grid at an integer computational period.

    Shape invariance, signature verification and interpolation support follow
    ``AgeSpecializedGrid``. This declaration is valid only in a period model.
    """

    # Coordinate-kind validation prevents using period callbacks in age models.
    build: Callable[[int], ContinuousGrid]
    """Factory returning the concrete continuous grid for an integer period."""

    signature: Callable[[int], Hashable]
    """Hashable identity of the period's grid, checked against its nodes."""


_MISSING = object()


class ByAge:
    """A nonterminal regime transition that depends on the source age.

    Each case maps an age selector to a nonterminal law: a regime name, a
    `DeterministicTransition`, a vector `StochasticTransition`,
    a per-target mapping, or a `Phased` of those. The selected ages are where a law
    is available; a schedule selects behavior and does not itself create a
    solved problem.
    `default` is a fallback law for every otherwise unmatched age, including the
    last.

    Selectors name exact grid coordinates:

    - a scalar or a tuple of scalars selects those ages;
    - a `range` selects its integers;
    - an `AgeRange` selects every grid age in `[start, exclusive_stop)`.

    Cases may not overlap. A law available at the last age is legal while no
    nonterminal problem is required there.

    When a `Transition` derives its targets from the schedule, every source age
    with an edge needs a case. An age with a single certain destination takes
    that regime's bare name, which mixes with per-target cases in one schedule:
    `cases={AgeRange(exclusive_stop=64): {"worker": p, "dead": q}, 64: "retiree"}`.
    `None` — terminality — is never a case: a terminal regime is a source
    without outgoing edges in `Model(edges=...)`.
    """

    def __init__[K: AgeSelector | PeriodSelector](
        self,
        *,
        cases: Mapping[K, _DeclaredCaseLaw],
        default: object = _MISSING,
    ) -> None:
        if not cases and default is _MISSING:
            raise RegimeInitializationError("`ByAge` needs at least one case.")
        for law in (*cases.values(), *(() if default is _MISSING else (default,))):
            _fail_if_not_a_nonterminal_law(law)
        for selector in cases:
            if isinstance(self, ByPeriod):
                _fail_if_invalid_period_selector(selector)
            elif isinstance(selector, PeriodRange | Periods):
                raise RegimeInitializationError(
                    "ByAge requires age selectors, not period selectors."
                )
            _fail_if_invalid_age_selector(selector)
        self._cases: tuple[tuple[object, object], ...] = tuple(
            (selector, snapshot_transition_containers(law))
            for selector, law in cases.items()
        )
        # Stored as `None` rather than the signature sentinel, so the model
        # fingerprint sees plain data.
        self._default = (
            None if default is _MISSING else snapshot_transition_containers(default)
        )
        self._until: _Until | None = None

    @classmethod
    def _from_until(cls, *, until: _Until) -> Self:
        """A schedule that resolves through `until` instead of cases."""
        schedule = cls.__new__(cls)
        vars(schedule).update(
            _cases=(),
            _default=None,
            _until=dataclasses.replace(
                until,
                law=snapshot_transition_containers(until.law),
                then=snapshot_transition_containers(until.then),
            ),
        )
        return schedule

    @classmethod
    def until(
        cls,
        *,
        stop_age_exclusive: UserAge | float,
        law: object,
        then: object,
        start_age_inclusive: UserAge | float | None = None,
    ) -> ByAge:
        """Supply `law` on `[start_age_inclusive, stop_age_exclusive)`, then exit.

        Both bounds are exact source grid ages. `then` is selected at the last
        source grid age below `stop_age_exclusive`, so its destinations are at
        `stop_age_exclusive`; `law` is selected at the earlier source ages. No
        law is supplied at or after the stop, or before the start. On an annual
        grid a stop of 62 exits at 61; on a quarterly grid at 61.75. An omitted
        start selects the first age. Both legs are nonterminal laws — an exit
        into a terminal regime names that regime.
        """
        _fail_if_not_a_nonterminal_law(law)
        _fail_if_not_a_nonterminal_law(then)
        return cls._from_until(
            until=_Until(
                stop_age_exclusive=stop_age_exclusive,
                law=law,
                then=then,
                start_age_inclusive=start_age_inclusive,
            )
        )

    def with_mapped_laws(self, *, func: Callable[[object], object]) -> Self:
        """Return this schedule with every law replaced by `func(law)`.

        The selectors are kept. Returns `self` when `func` leaves every law
        unchanged, so identity comparisons of declarations stay meaningful.
        """
        mapped = tuple(func(law) for law in self.laws)
        if all(new is old for new, old in zip(mapped, self.laws, strict=True)):
            return self
        if self._until is not None:
            law, then = mapped
            return type(self)._from_until(  # noqa: SLF001
                until=dataclasses.replace(self._until, law=law, then=then)
            )
        cases = cast(
            "Mapping[AgeSelector, AgeCaseLaw]",
            {
                selector: law
                for (selector, _), law in zip(self._cases, mapped, strict=False)
            },
        )
        if self._default is None:
            return type(self)(cases=cases)
        return type(self)(cases=cases, default=mapped[-1])

    @property
    def laws(self) -> tuple[object, ...]:
        """Every law the schedule may select, in declaration order."""
        if self._until is not None:
            return (self._until.law, self._until.then)
        return (
            *(law for _, law in self._cases),
            *(() if self._default is None else (self._default,)),
        )

    def resolve(self, ages: TimeAxis) -> ResolvedSchedule:
        """Resolve the selectors against `ages` without evaluating any law."""
        if isinstance(self, ByPeriod) != (coordinate_kind(ages) == "period"):
            raise RegimeInitializationError(
                "ByPeriod requires a period model; ByAge requires an age model."
            )
        period_by_age: dict[object, int] = {
            age: period for period, age in enumerate(ages.exact_values)
        }
        if self._until is not None:
            law_by_period = _resolve_until(
                until=self._until, ages=ages, period_by_age=period_by_age
            )
        else:
            law_by_period = {}
            for selector, law in self._cases:
                periods = _select_periods(
                    selector=selector, ages=ages, period_by_age=period_by_age
                )
                if not periods:
                    raise RegimeInitializationError(
                        f"`ByAge` selector {selector!r} selects no age of the model."
                    )
                overlap = sorted(set(periods) & set(law_by_period))
                if overlap:
                    raise RegimeInitializationError(
                        f"`ByAge` selector {selector!r} overlaps another case at "
                        f"age(s) {[ages.exact_values[p] for p in overlap]}."
                    )
                law_by_period.update(dict.fromkeys(periods, law))
            if self._default is not None:
                for period in range(ages.n_periods):
                    law_by_period.setdefault(period, self._default)
        return ResolvedSchedule(
            ages=ages,
            law_by_period=MappingProxyType(dict(sorted(law_by_period.items()))),
        )


class ByPeriod(ByAge):
    """Select transition laws by zero-based computational period.

    Integer keys, integer tuples, ranges, PeriodRange and Periods declare periods.
    A stage advances one computational slot, without implying elapsed calendar time.
    """

    @classmethod
    # The explicit period vocabulary deliberately differs from ByAge's keywords.
    def until(  # ty: ignore[invalid-method-override]
        cls,
        *,
        stop_period_exclusive: int,
        law: object,
        then: object,
        start_period_inclusive: int | None = None,
    ) -> ByPeriod:
        """Use `law` before the predecessor of the stop and `then` on it."""
        for value in (stop_period_exclusive, start_period_inclusive):
            if value is not None:
                _fail_if_invalid_period_selector(value)
        _fail_if_not_a_nonterminal_law(law)
        _fail_if_not_a_nonterminal_law(then)
        return cls._from_until(
            until=_Until(
                stop_age_exclusive=stop_period_exclusive,
                start_age_inclusive=start_period_inclusive,
                law=law,
                then=then,
            )
        )

    # Period horizons are named; the inherited age-only form retains positional use.
    def resolve(  # ty: ignore[invalid-method-override]
        self, *, ages: TimeAxis | None = None, n_periods: int | None = None
    ) -> ResolvedSchedule:
        """Resolve against a period horizon, with no implied age labels."""
        if ages is not None and n_periods is not None:
            raise RegimeInitializationError("Supply exactly one period horizon.")
        axis = (
            ModelTime.from_inputs(ages=None, n_periods=n_periods)
            if ages is None
            else ages
        )
        return super().resolve(ages=axis)


def _fail_if_invalid_period_selector(selector: object) -> None:
    """Require genuinely integer period coordinates, without numeric coercion."""
    if isinstance(selector, PeriodRange | Periods):
        return
    values = selector if isinstance(selector, tuple | range) else (selector,)
    if any(type(value) is not int for value in values):
        raise RegimeInitializationError(
            "Period selectors must contain integers or PeriodRange/Periods."
        )


@dataclass(frozen=True, kw_only=True)
class ResolvedSchedule:
    """A schedule resolved against one age grid; inspection only."""

    ages: TimeAxis
    """The grid the schedule was resolved against."""

    law_by_period: MappingProxyType[int, object]
    """The selected law at each covered period."""

    @property
    def covered_ages(self) -> tuple[UserAge, ...]:
        """The exact ages the schedule covers, ascending."""
        if coordinate_kind(self.ages) == "period":
            raise AttributeError("A period schedule has no covered_ages; use periods.")
        return tuple(self.ages.exact_values[period] for period in self.law_by_period)

    @property
    def periods(self) -> tuple[int, ...]:
        """The covered period indices, ascending."""
        return tuple(self.law_by_period)

    def at(self, age: UserAge | float) -> object:
        """Return the law selected at `age`; raise `KeyError` if it is uncovered.

        `age` must equal a grid age exactly, as a selector does; a float that
        only approximates a grid age is uncovered.
        """
        if coordinate_kind(self.ages) == "period" and type(age) is not int:
            raise TypeError("Schedule lookup requires an integer period.")
        period = {exact: p for p, exact in enumerate(self.ages.exact_values)}.get(age)
        if period is None or period not in self.law_by_period:
            raise KeyError(age)

        return self.law_by_period[period]


@dataclass(frozen=True, kw_only=True)
class _Until:
    """The declaration of a `ByAge.until` schedule."""

    stop_age_exclusive: UserAge | float
    """The exact source age at and after which no law is supplied."""
    law: object
    """The law at the earlier source ages."""
    then: object
    """The law at the last source age below the stop."""
    start_age_inclusive: UserAge | float | None
    """The first source age with a law, or `None` for the first grid age."""


_NESTED_SCHEDULE = (
    "`ByAge` cannot be nested inside `ByAge` or `Phased`. Put one `Phased` "
    "inside each `ByAge` case instead."
)


def fail_if_phased_wraps_a_schedule(transition: object) -> None:
    """Reject a top-level `Phased` regime transition with a `ByAge` side.

    A schedule varies by age and `Phased` by phase; age is the outer dimension,
    so the `Phased` goes inside each `ByAge` case.
    """
    if not isinstance(transition, Phased):
        return
    sides = tuple(
        f"`{name}`"
        for name, side in (
            ("solve", transition.solve),
            ("simulate", transition.simulate),
        )
        if isinstance(side, ByAge)
    )
    if sides:
        raise RegimeInitializationError(
            f"{_NESTED_SCHEDULE} The top-level `Phased` regime transition has a "
            f"`ByAge` as its {' and '.join(sides)} side."
        )


def _fail_if_not_a_nonterminal_law(law: object) -> None:
    """Reject terminality and nested schedules inside a schedule."""
    sides = (law.solve, law.simulate) if isinstance(law, Phased) else (law,)
    for side in sides:
        if side is None:
            raise RegimeInitializationError(
                "`None` marks a terminal regime only as the top-level "
                "law of a source without outgoing edges; a schedule case must be a "
                "nonterminal law. "
                'Name the terminal regime to exit into it, e.g. `then="dead"`.'
            )
        if isinstance(side, ByAge):
            raise RegimeInitializationError(_NESTED_SCHEDULE)
        if isinstance(side, Mapping) and not side:
            raise RegimeInitializationError(
                "A per-target transition mapping must name at least one target."
            )


def _fail_if_invalid_age_selector(selector: object) -> None:
    """Reject selectors whose values can never be grid ages."""
    values = selector if isinstance(selector, tuple | range) else (selector,)
    if isinstance(selector, AgeRange | PeriodRange):
        values = tuple(
            v for v in (selector.start, selector.exclusive_stop) if v is not None
        )
    if isinstance(selector, Periods):
        values = selector.values
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float | Fraction):
            raise RegimeInitializationError(
                f"Age selector {selector!r} must name numeric ages, not {value!r}."
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise RegimeInitializationError(
                f"Age selector {selector!r} contains a nonfinite age."
            )
    if (
        isinstance(selector, AgeRange)
        and selector.start is not None
        and selector.exclusive_stop is not None
        and selector.start >= selector.exclusive_stop
    ):
        raise RegimeInitializationError(
            f"`AgeRange` start {selector.start} must be below exclusive_stop "
            f"{selector.exclusive_stop}."
        )


def _select_periods(
    *, selector: object, ages: TimeAxis, period_by_age: Mapping[object, int]
) -> tuple[int, ...]:
    """Return the periods an exact selector names; off-grid points raise."""
    if isinstance(selector, AgeRange | PeriodRange):
        return tuple(
            period
            for period, age in enumerate(ages.exact_values)
            if (selector.start is None or age >= selector.start)
            and (selector.exclusive_stop is None or age < selector.exclusive_stop)
        )
    values = selector if isinstance(selector, tuple | range) else (selector,)
    periods = set()
    if isinstance(selector, Periods):
        values = selector.values
    for value in values:
        if value not in period_by_age:
            kind = coordinate_kind(ages)
            article = "an" if kind == "age" else "a"
            raise RegimeInitializationError(
                f"{kind.title()} {value} in selector {selector!r} is not "
                f"{article} {kind} of the model; "
                f"valid {kind}s are {list(ages.exact_values)}."
            )
        periods.add(period_by_age[value])
    return tuple(sorted(periods))


def _resolve_until(
    *,
    until: _Until,
    ages: TimeAxis,
    period_by_age: Mapping[object, int],
) -> dict[int, object]:
    """Resolve `ByAge.until` into per-period laws."""
    boundary, start = until.stop_age_exclusive, until.start_age_inclusive
    for name, value in (
        ("stop_age_exclusive", boundary),
        ("start_age_inclusive", start),
    ):
        if value is not None and value not in period_by_age:
            raise RegimeInitializationError(
                f"`ByAge.until` {name} {value} is not an age of the model; valid "
                f"ages are {list(ages.exact_values)}."
            )
    stop = period_by_age[boundary]
    if stop == 0:
        raise RegimeInitializationError(
            f"`ByAge.until` stop_age_exclusive {boundary} is the first age and has no "
            "predecessor on which to exit."
        )
    first = 0 if start is None else period_by_age[start]
    if first >= stop:
        raise RegimeInitializationError(
            f"`ByAge.until` start_age_inclusive {start} is not before "
            f"stop_age_exclusive {boundary}."
        )
    return dict.fromkeys(range(first, stop - 1), until.law) | {stop - 1: until.then}
