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

    Semantics (the information-timing contract)
    -------------------------------------------
    The container is value-agnostic, but regime building gives the two phases a
    fixed meaning: **the agent acts on its beliefs about the future and lives in
    the truth now.**

    - ``solve`` is the **perceived** (belief) variant. It is used ONLY to price
      the **continuation** — next-period state kernels, regime-transition
      probabilities, and their helpers. The backward-induction value function
      the agent optimises against is the one solved under these beliefs.
    - ``simulate`` is the **truth**. It governs (a) what is actually realised as
      the simulation walks forward, and (b) the **current-period** flow of the
      decision: within-period utility, feasibility, the aggregator ``H``, and any
      chosen deterministic ``next_<state>`` those read.

    The load-bearing assumption is one of **information timing**: the simulate-side
    current consequence, utility technology, feasibility, and ``H`` are known when
    the action is chosen; only the *future* (the next-period kernels) is perceived.
    So a within-period utility reading a chosen deterministic ``next_<state>`` (the
    NEGM service-flow pattern) reads its ``simulate`` variant — the agent knows the
    true deterministic consequence of *its own action*. Misperception enters only
    at the continuation boundary. This is *not* implied by the container; it is the
    contract regime building imposes, and belief-flow (pricing current utility
    under the perceived law) would be a separate primitive, not a reading of
    ``Phased``.

    Two consequences of the contract are enforced, not merely documented:

    - A within-period utility or feasibility may not read a ``next_<state>`` that
      is **stochastic** in that phase: its value is not known at choice time.
    - **Constraints are phase-invariant through their whole dependency ancestry.**
      A phase-specific feasible set would let the simulated argmax range over
      actions the value function was never computed for; a direct ``Phased``
      constraint is rejected, and so is a bare constraint that reaches a ``Phased``
      helper or ``Phased`` ``next_<state>`` transitively.

    One deliberate exception to "current flow is the ``simulate`` truth":
    a **carried-only** state (present in simulate but not on the solve grid) enters
    the decision under its solve **imputation**, not its realised carried value,
    because the continuation was solved at that imputation — the decision stays
    consistent with the policy it re-optimises (policy-consistency). Its realised
    value is still used for the forward transition. See
    ``regime_building/processing.py``.

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
