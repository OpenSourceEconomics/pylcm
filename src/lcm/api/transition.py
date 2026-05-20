"""User-facing transition wrappers: `MarkovTransition`, `SolveSimulateFunctionPair`.

A thin leaf module — class definitions only, with no dependency on `Regime`,
the validators, or the regime-building code. Keeping these types here lets the
user-facing `Regime`, the engine-internal regime-building code, and the
regime validators all import them without an import cycle.

"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from beartype import beartype

from lcm._beartype_conf import REGIME_CONF
from lcm.typing import FloatND


class SolveSimulateFunctionPair[S, T]:
    """Container for phase-specific function variants.

    Use this to provide different implementations of a function for the solve
    and simulate phases.  For example, naive beta-delta discounting uses
    exponential discounting during backward induction (solve) but
    present-biased discounting for action selection (simulate).

    Variants may have different parameter signatures.  The params template is
    the union of both variants' parameters; each variant receives only the
    kwargs it expects.

    """

    __slots__ = ("simulate", "solve")

    def __init__(self, *, solve: S, simulate: T) -> None:
        self.solve = solve
        self.simulate = simulate


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
