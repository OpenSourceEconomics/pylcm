"""User-facing vocabulary: `fixed_transition`, `MarkovTransition`, `AgeSpecialized`.

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


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True)
class AgeSpecialized:
    """Wrapper marking a function whose closure is bound per age at build time.

    Wrap a function *factory* to indicate that its closure depends on the agent's
    age — for example a tax-transfer system pinned to a policy date that moves with
    calendar time as the agent ages. At build time pylcm calls `build(age)` for each
    period's age to obtain that period's concrete function, and uses `signature(age)`
    as a dedup key so ages resolving to the same closure share a single compiled
    program.

    Usable in `functions` and `constraints` of non-terminal regimes. A
    policy-dependent law of motion is expressed as a plain state transition that
    reads an `AgeSpecialized` entry of `functions`; a direct `AgeSpecialized`
    state-transition value, a specialized regime `transition`, a regime
    transition whose dependency graph reads an `AgeSpecialized` function, a
    `MarkovTransition(AgeSpecialized(...))`, and any `AgeSpecialized` in a
    terminal regime are rejected at `Regime` construction. Every concrete
    function returned by `build` must expose the same call signature — only the
    constants it closes over may differ across ages.

        functions={"tax": AgeSpecialized(build=make_tax, signature=policy_key)}

    `signature` is a **correctness precondition**, not a performance hint: ages
    with equal signatures share one compiled program, so an equal signature must
    imply identical closure behavior (policy date, price level, overrides, and
    every other closed-over constant). An incomplete signature silently shares a
    wrong program across ages.

    A bare callable (without the wrapper) is age-invariant, as before. `AgeSpecialized`
    is a build-time marker: it is resolved to a concrete function via `build(age)`
    before the DAG is traced, so calling it directly is an error.
    """

    build: Callable[[float], UserFunction]
    """Factory returning the concrete function for a given age."""

    signature: Callable[[float], Hashable]
    """Returns a hashable identity of the age's closure; used as the dedup key."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401, ARG002
        msg = (
            "AgeSpecialized is a build-time marker and must be resolved to a concrete "
            "function via build(age) before it is called."
        )
        raise TypeError(msg)
