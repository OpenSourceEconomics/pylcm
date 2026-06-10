"""User-facing transition wrappers.

`MarkovTransition`, `SolveSimulateFunctionPair`, and `SolveSimulateStatePair`.

A thin leaf module — class definitions only, with no dependency on `Regime`,
the validators, or the regime-building code. Keeping these types here lets the
user-facing `Regime`, the engine-internal regime-building code, and the
regime validators all import them without an import cycle.

"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
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


class SolveSimulateStatePair[S, G, T]:
    """Container for a quantity that is a derived function in solve but a state in
    simulate.

    Place this in `Regime.states` to give a quantity two phase-specific roles:

    - In the **solve** phase the name is a derived DAG function (`solve`): it is
      computed from other states/actions and never becomes a grid dimension, so
      the solve grid (and its cost) is unchanged.
    - In the **simulate** phase the name is a genuine state: seeded from the
      initial conditions on `grid` and carried forward each period via
      `transition`. Simulate-phase functions read this carried-forward value
      instead of the solve-phase imputation.

    The canonical use is pension wealth: imputed from AIME during backward
    induction, but seeded to its true value and evolved in simulation so the
    realized budget reflects the actual carried-forward wealth.

    In the simulate phase every published consumer — state transitions,
    feasibility checks, `to_dataframe` additional targets — reads the carried
    value; only the decision (the argmax over the solved policy and the
    regime-transition probabilities) reads the imputation. To expose the
    imputed value as a simulate output as well, declare the `solve` callable
    under a second name in `Regime.functions`.

    """

    __slots__ = ("grid", "solve", "transition")

    def __init__(self, *, solve: S, grid: G, transition: T) -> None:
        self.solve = solve
        self.grid = grid
        self.transition = transition


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
