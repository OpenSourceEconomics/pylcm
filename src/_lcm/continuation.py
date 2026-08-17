"""Current EGM continuation representation and its declared layout.

The engine treats `ContinuationPayload` as opaque. Solver and EGM modules use the
concrete `EGMContinuationSpec`, which keeps the all-finite template and every
layout property needed by a reading parent in one immutable object.
"""

from dataclasses import dataclass

from _lcm.egm.carry import EGMCarry

# Backward induction stores and forwards this channel without reading its fields.
# A future non-EGM representation should introduce an operation protocol rather
# than widening this alias independently in several engine modules.
type ContinuationPayload = EGMCarry


@dataclass(frozen=True, kw_only=True)
class EGMContinuationLayout:
    """Static interpretation of the leading axes in one EGM carry."""

    retains_discrete_action_rows: bool = True
    """Whether a parent must aggregate child discrete-action rows."""

    rows_share_state_grid: bool = False
    """Whether every row uses the child's own state grid as abscissae."""

    n_stacked_candidates: int = 0
    """Length of a stacked outer-candidate axis; zero means no such axis."""


@dataclass(frozen=True, kw_only=True)
class EGMContinuationSpec:
    """Concrete EGM continuation template bundled with its static layout."""

    template: EGMCarry
    """All-finite payload with the exact shapes used for rolling and lowering."""

    layout: EGMContinuationLayout = EGMContinuationLayout()
    """How a reading parent interprets the template and produced payloads."""
