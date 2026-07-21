"""User-facing vocabulary: `fixed_transition`, `MarkovTransition`,
`AgeSpecializedFunction`, `AgeSpecializedGrid`.

A thin leaf module with no dependency on `Regime`, the validators, or the
regime-building code. Keeping the vocabulary here lets the user-facing
`Regime`, the engine-internal regime-building code, and the regime validators
all import it without an import cycle.

"""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.grids.continuous import ContinuousGrid
from _lcm.identity_transition import _IdentityTransition
from _lcm.typing import StateName
from lcm.typing import FloatND, UserFunction


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
    return _IdentityTransition(state_name)


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

    **What `signature(age)` means differs by marker**, so read the concrete class:
    - `AgeSpecializedFunction` — `signature` **is** the dedup key and a *correctness
      precondition*: equal signature must imply an identical resolved closure. A
      function's closure cannot be inspected, so pylcm has to take the author's word.
    - `AgeSpecializedGrid` — `signature` is **not** the dedup key and **not**
      load-bearing. Grids dedup on their *resolved nodes*: a grid can be asked what it
      actually is, so nothing hand-written is trusted for correctness.

    The two concrete markers are `AgeSpecializedFunction` (a function whose closure
    varies with age) and `AgeSpecializedGrid` (a continuous-state grid whose
    bounds/nodes vary with age at a fixed shape). A marker is resolved before
    it is used, so calling it directly is a loud error.
    """

    build: Callable[[float], Any]
    """Factory returning the concrete object (function or grid) for a given age. Must be
    deterministic and side-effect-free; it is called more than once per age."""

    signature: Callable[[float], Hashable]
    """Hashable identity of the age's object. The dedup key (and a correctness
    precondition) for `AgeSpecializedFunction`; not load-bearing for
    `AgeSpecializedGrid`, which dedups on resolved nodes. See the class docstrings."""

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
    `transition`, a regime transition whose dependency graph reads an
    `AgeSpecializedFunction`, a `MarkovTransition(AgeSpecializedFunction(...))`, and
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

    Unlike `AgeSpecializedFunction.signature`, this `signature` is **not** the dedup
    key for grids, and it is not a correctness precondition: grids are deduplicated on
    their resolved nodes, which cannot disagree with the grid the way a hand-written
    signature can. It is retained for API symmetry.

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
    """Returns a hashable identity of the age's grid. Retained for symmetry with
    `AgeSpecializedFunction`; grid dedup keys on the resolved nodes, not on this."""
