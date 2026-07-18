"""Solver-published numerical diagnostics.

A continuous-outer solve is only inference-grade when its numerical error is
*observable*: how far the interpolant strays from exact solves, how wide the
final refinement brackets are, how much population-relevant state mass sits at
bounds or branch ties. Solvers publish those observations through
`SolverDiagnostics` on their `KernelResult`; the engine treats the payload as
opaque cargo (no solver-type switch), and downstream release gates read it.

The finite-grid solvers publish nothing (`None`); the continuous-outer solver
fills the fields it can measure. Every field is optional-by-shape rather than
optional-by-`None` inside one payload: a period that measures nothing simply
does not publish a payload at all.
"""

from dataclasses import dataclass
from typing import Literal

from lcm.typing import BoolND, FloatND, IntND

# How much diagnostic cargo a solve retains: "none" keeps no diagnostic
# arrays; "summary" keeps scalar maxima/quantiles/counts; "full" keeps
# state-array diagnostics (research/debugging).
type DiagnosticLevel = Literal["none", "summary", "full"]


@dataclass(frozen=True, kw_only=True)
class SolverDiagnostics:
    """Per-period numerical diagnostics a solver publishes with its solve.

    Shapes are the solver's choice per its declared `DiagnosticLevel`:
    state-shaped arrays at `"full"`, 0-d summaries at `"summary"`. Consumers
    must not assume a particular shape — only that larger is worse for error
    fields and that masks mark cells needing attention.
    """

    max_outer_interpolation_error: FloatND
    """Largest validated interpolant-vs-exact-solve gap, in value units."""

    max_outer_bracket_width: FloatND
    """Widest refinement bracket around a selected outer optimum."""

    outer_nodes_used: IntND
    """Number of exact outer candidate nodes the solve ended with."""

    outer_at_lower_bound: BoolND
    """Where the selected outer action sits at the domain's lower bound."""

    outer_at_upper_bound: BoolND
    """Where the selected outer action sits at the domain's upper bound."""

    keeper_adjuster_margin: FloatND
    """Winning-branch margin `|V_keeper - V_adjuster|` at the optimum."""

    best_second_best_margin: FloatND
    """Margin between the best and second-best outer candidates."""

    policy_fallback_mask: BoolND
    """Where a policy read fell back rather than reading the optimum."""

    unresolved_mask: BoolND
    """Where the solve could not certify its outer optimum to tolerance."""
