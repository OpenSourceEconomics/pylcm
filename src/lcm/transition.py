"""User-facing transition vocabulary: `fixed_transition`, `MarkovTransition`,
`Choose`, `ByAge`, `AgeRange`, `JointTransition`, `AgeSpecializedFunction`, and
`AgeSpecializedGrid`.

A thin leaf module with no dependency on `Regime`, the validators, or the
regime-building code. Keeping the vocabulary here lets the user-facing
`Regime`, the engine-internal regime-building code, and the regime validators
all import it without an import cycle.

"""

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import jax
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.grids.continuous import ContinuousGrid
from _lcm.identity_transition import _IdentityTransition
from _lcm.typing import StateName
from lcm.ages import AgeGrid
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.typing import FloatND, UserAge, UserFunction


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


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class MarkovTransition:
    """Wrapper marking a transition function as stochastic (Markov).

    Wrap a transition function in `MarkovTransition` to indicate that it returns
    a probability distribution over next states (for state transitions) or over
    next regimes (for regime transitions), rather than a deterministic next value.

    Use at both the state and regime level:

        # Stochastic state transition (in Regime.state_transitions)
        state_transitions={"health": MarkovTransition(func=health_probs)}

        # Stochastic regime transition
        Regime(transition=MarkovTransition(func=regime_probs), ...)

    A bare callable (without the wrapper) is deterministic at both levels.

    At the regime level, a bare callable or a bare `MarkovTransition` (as
    opposed to a per-target dict) declares conservative support over every
    regime active in the next period: every temporally compatible candidate
    must have a valid state handoff, and the check runs regardless of what
    probability the transition function happens to return at runtime. Use a
    per-target dict on `Regime.regime_transitions` to declare narrower, structural
    support — a runtime-zero probability does not narrow it.

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

    targets: Sequence[str] | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """For a regime transition returning the full regime-ID vector: its support.

    Names the regimes the vector may assign positive probability to. Every other
    entry of the returned vector must be exactly zero. State laws declare none.
    """

    def __post_init__(self) -> None:
        if self.targets is not None:
            object.__setattr__(
                self,
                "targets",
                _declared_targets(targets=self.targets, owner="MarkovTransition"),
            )
        # Copy __wrapped__ and __annotations__ from the wrapped function so
        # that inspect.signature and dags see the original signature. We use
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "__wrapped__", self.func)
        object.__setattr__(
            self, "__annotations__", getattr(self.func, "__annotations__", {})
        )

    def __call__(self, *args: Any, **kwargs: Any) -> FloatND:  # noqa: ANN401
        return self.func(*args, **kwargs)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Choose:
    """A deterministic regime transition with declared support.

    `func` returns an existing global regime code — the same integer a bare
    deterministic transition returns — and `targets` names every regime it may
    select. It is evaluated on the deterministic route and adds no random draw.

        Regime(transition=Choose(func=next_regime, targets=("work", "retired")), ...)

    Returning a code outside `targets` is an error; a target that is never
    selected at runtime remains a declared edge.
    """

    func: Callable[..., Any]
    """The selector returning a global regime code."""

    targets: Sequence[str]
    """The regimes `func` may select."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "targets", _declared_targets(targets=self.targets, owner="Choose")
        )
        object.__setattr__(self, "__wrapped__", self.func)
        object.__setattr__(
            self, "__annotations__", getattr(self.func, "__annotations__", {})
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.func(*args, **kwargs)


def _freeze_joint_support(value: Any) -> Any:  # noqa: ANN401
    """Freeze the container structure of a literal joint-support pytree."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_joint_support(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_joint_support(item) for item in value)
    return value


def _literal_joint_support_schema(
    support: Any,  # noqa: ANN401
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
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
    reads an `AgeSpecializedFunction` entry of `functions`; a direct
    `AgeSpecializedFunction` state-transition value, a specialized regime
    `regime_transitions`, a regime transition whose dependency graph reads an
    `AgeSpecializedFunction`, a
    `MarkovTransition(func=AgeSpecializedFunction(...))`, and
    any `AgeSpecializedFunction` in a terminal regime are rejected at `Regime`
    construction. Every concrete function returned by `build` must expose the same
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


@dataclass(frozen=True, kw_only=True)
class AgeRange:
    """The existing grid ages in the half-open interval `[start, stop)`.

    Either bound may be omitted. The bounds need not be grid points; the
    selection is always a subset of the model's ages.
    """

    start: UserAge | float | None = None
    """Inclusive lower bound, or `None` for the first age."""

    stop: UserAge | float | None = None
    """Exclusive upper bound, or `None` for beyond the last age."""


_MISSING = object()

type AgeSelector = UserAge | float | tuple[UserAge | float, ...] | range | AgeRange


class ByAge:
    """A nonterminal regime transition that depends on the source age.

    Each case maps an age selector to a nonterminal law: a regime name, a
    `Choose`, a vector `MarkovTransition` with `targets`, a per-target mapping,
    or a `Phased` of those. The selected ages are where a law is available; a
    schedule selects behavior and does not by itself create a solved problem.
    `default` is a fallback law for every otherwise unmatched age, including the
    last.

    Selectors name exact grid coordinates:

    - a scalar or a tuple of scalars selects those ages;
    - a `range` selects its integers;
    - an `AgeRange` selects every grid age in `[start, stop)`.

    Cases may not overlap. A law available at the last age is legal while no
    nonterminal problem is required there.
    `None` — terminality — is only ever the top-level `Regime.regime_transitions`.
    """

    def __init__(
        self,
        *,
        cases: Mapping[AgeSelector, object],
        default: object = _MISSING,
        _until: tuple[object, object, object, object] | None = None,
    ) -> None:
        if not cases and default is _MISSING and _until is None:
            raise RegimeInitializationError("`ByAge` needs at least one case.")
        for law in (*cases.values(), *(() if default is _MISSING else (default,))):
            _fail_if_not_a_nonterminal_law(law)
        self._cases: tuple[tuple[object, object], ...] = tuple(
            (_freeze_selector(selector), law) for selector, law in cases.items()
        )
        self._default = default
        self._until = _until

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
        return cls(
            cases={}, _until=(stop_age_exclusive, law, then, start_age_inclusive)
        )

    @property
    def laws(self) -> tuple[object, ...]:
        """Every law the schedule may select, in declaration order."""
        if self._until is not None:
            return (self._until[1], self._until[2])
        return (
            *(law for _, law in self._cases),
            *(() if self._default is _MISSING else (self._default,)),
        )

    def resolve(self, ages: AgeGrid) -> ResolvedSchedule:
        """Resolve the selectors against `ages` without evaluating any law."""
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
            if self._default is not _MISSING:
                for period in range(ages.n_periods):
                    law_by_period.setdefault(period, self._default)
        return ResolvedSchedule(
            ages=ages,
            law_by_period=MappingProxyType(dict(sorted(law_by_period.items()))),
        )


@dataclass(frozen=True, kw_only=True)
class ResolvedSchedule:
    """A schedule resolved against one age grid; inspection only."""

    ages: AgeGrid
    """The grid the schedule was resolved against."""

    law_by_period: MappingProxyType[int, object]
    """The selected law at each covered period."""

    @property
    def covered_ages(self) -> tuple[UserAge, ...]:
        """The exact ages the schedule covers, ascending."""
        return tuple(self.ages.exact_values[period] for period in self.law_by_period)

    @property
    def periods(self) -> tuple[int, ...]:
        """The covered period indices, ascending."""
        return tuple(self.law_by_period)

    def at(self, age: UserAge | float) -> object:
        """Return the law selected at `age`; raise `KeyError` if it is uncovered."""
        for period, exact in enumerate(self.ages.exact_values):
            if exact == age and period in self.law_by_period:
                return self.law_by_period[period]
        raise KeyError(age)


def _declared_targets(*, targets: Sequence[str], owner: str) -> tuple[str, ...]:
    """Validate and freeze a declared regime support."""
    frozen = tuple(targets)
    if not frozen:
        raise RegimeInitializationError(f"`{owner}.targets` must name a regime.")
    if len(set(frozen)) != len(frozen):
        raise RegimeInitializationError(
            f"`{owner}.targets` names a regime more than once: {list(frozen)}."
        )
    return frozen


def _fail_if_not_a_nonterminal_law(law: object) -> None:
    """Reject terminality and nested schedules inside a schedule."""
    sides = (law.solve, law.simulate) if isinstance(law, Phased) else (law,)
    for side in sides:
        if side is None:
            raise RegimeInitializationError(
                "`None` marks a terminal regime only as the top-level "
                "`Regime.regime_transitions`; a schedule case must be a "
                "nonterminal law. "
                'Name the terminal regime to exit into it, e.g. `then="dead"`.'
            )
        if isinstance(side, ByAge):
            raise RegimeInitializationError(
                "`ByAge` cannot be nested inside `ByAge` or `Phased`. Put one "
                "`Phased` inside each `ByAge` case instead."
            )
        if isinstance(side, Mapping) and not side:
            raise RegimeInitializationError(
                "A per-target transition mapping must name at least one target."
            )


def _freeze_selector(selector: object) -> object:
    """Freeze a selector and reject values that can never be grid ages."""
    values = selector if isinstance(selector, tuple | range) else (selector,)
    if isinstance(selector, AgeRange):
        values = tuple(v for v in (selector.start, selector.stop) if v is not None)
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
        and selector.stop is not None
        and selector.start >= selector.stop
    ):
        raise RegimeInitializationError(
            f"`AgeRange` start {selector.start} must be below stop {selector.stop}."
        )
    return selector


def _select_periods(
    *, selector: object, ages: AgeGrid, period_by_age: Mapping[object, int]
) -> tuple[int, ...]:
    """Return the periods an exact selector names; off-grid points raise."""
    if isinstance(selector, AgeRange):
        return tuple(
            period
            for period, age in enumerate(ages.exact_values)
            if (selector.start is None or age >= selector.start)
            and (selector.stop is None or age < selector.stop)
        )
    values = selector if isinstance(selector, tuple | range) else (selector,)
    periods = set()
    for value in values:
        if value not in period_by_age:
            raise RegimeInitializationError(
                f"Age {value} in selector {selector!r} is not an age of the model; "
                f"valid ages are {list(ages.exact_values)}."
            )
        periods.add(period_by_age[value])
    return tuple(sorted(periods))


def _resolve_until(
    *,
    until: tuple[object, object, object, object],
    ages: AgeGrid,
    period_by_age: Mapping[object, int],
) -> dict[int, object]:
    """Resolve `ByAge.until` into per-period laws."""
    boundary, law, then, start = until
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
    return dict.fromkeys(range(first, stop - 1), law) | {stop - 1: then}
