"""User-facing outer-search strategy configuration (re-export façade).

A nested solver's `outer_search` field selects how the scalar continuous
outer action is searched:

- `FiniteOuterGrid(...)`: the historical finite-candidate behavior — exact
  relative to a fixed exogenous grid, result grid-snapped.
- `AdaptiveOuterMesh(...)`: the canonical continuous-outer approximation —
  exact solves on an adaptive shared mesh, a validated interpolant between
  nodes, and globally safeguarded bracket-local refinement.
- `LegacyGoldenSection(...)`: historical-algorithm compatibility with
  source-specific endpoint and tie rules; never the canonical paper mode.

The strategies are defined engine-side in `_lcm.egm.outer_search`; this
module is a thin re-export so user code can name them, and the abstract
`OuterSearch` marker, without eagerly importing the numerical engine.
"""

from _lcm.egm.outer_search import (
    AdaptiveOuterMesh,
    FiniteOuterGrid,
    LegacyGoldenSection,
    OuterSearch,
)

__all__ = [
    "AdaptiveOuterMesh",
    "FiniteOuterGrid",
    "LegacyGoldenSection",
    "OuterSearch",
]
