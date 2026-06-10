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
