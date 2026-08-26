"""The `Phased` container for phase-specific variants of regime-slot values.

A thin leaf module — the class definition only, with no dependency on
`Regime`, the validators, or the regime-building code, so the user-facing
`Regime`, the engine-internal normalizer, and the regime validators can all
import it without an import cycle.
"""

from lcm.exceptions import RegimeInitializationError


class Phased[S, T]:
    """Phase-specific variants of a regime-slot value.

    Wherever a regime slot admits phase variance, a bare value broadcasts to
    both the solve and simulate phases; `Phased` specifies each phase
    explicitly. Which value types a slot accepts per phase is governed by the
    slot's grammar (see `Regime`); the container itself is value-agnostic.

    Both variants are required keyword arguments. Nesting `Phased` inside
    `Phased` is rejected — phase is a single broadcast dimension.

    **What the two phases mean.** The container is value-agnostic, but regime
    building gives them a fixed economic meaning: *the agent acts on its beliefs
    about the future and lives in the truth now*. The agent is naive — it does not
    anticipate that the world will differ from the model it solved.

    - `solve` is the **perceived** law. It prices the **continuation** and nothing
      else: the next-period state kernels, the regime-transition probabilities,
      and every helper those read. The value function the agent optimizes against
      is the one solved under these beliefs.
    - `simulate` is the **truth**. It governs what is realized as the simulation
      moves forward, and it supplies the **current-period** primitives of the
      decision: period utility, feasibility, the Koopmans aggregator, and any
      deterministic `next_<state>` those read.

    The split follows the agent's information set at the moment of choice. Today's
    utility, today's feasible set, and the deterministic consequence of the agent's
    own action are known when the action is taken; only the future is perceived. So
    a period utility reading a chosen deterministic `next_<state>` reads its
    `simulate` variant, and misperception enters only at the continuation boundary.
    Pricing today's utility under a perceived law would be a separate primitive,
    not a reading of `Phased`.

    Two consequences are enforced rather than merely documented:

    - Period utility and feasibility may not read a `next_<state>` that is
      stochastic in that phase — its value is not known when the action is chosen.
    - Constraints are phase-invariant through their whole dependency chain, so the
      simulated agent never chooses an action its value function was not computed
      for.

    One deliberate exception to "the current period is the `simulate` truth": a
    carried-only state — one the simulation tracks but the solve grid does not
    carry — enters the decision at its solve imputation rather than its realized
    value, because the imputation is what the continuation was solved at. The
    realized value still drives the forward transition.

    """

    __slots__ = ("simulate", "solve")

    def __init__(self, *, solve: S, simulate: T) -> None:
        if isinstance(solve, Phased) or isinstance(simulate, Phased):
            msg = (
                "Nested `Phased` is not supported: phase is a single broadcast "
                "dimension, so each variant must be a plain slot value."
            )
            raise RegimeInitializationError(msg)
        self.solve = solve
        self.simulate = simulate

    def __repr__(self) -> str:
        return f"Phased(solve={self.solve!r}, simulate={self.simulate!r})"
