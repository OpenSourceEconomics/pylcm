"""The four ways a solver can meet a constraint, and nothing else.

A solver either evaluates a constraint, relies on its own construction to
enforce it, compiles it into a boundary it splits its candidate grid on, or
refuses it. Those four are exhaustive on purpose. The state they exclude is a
solver holding a constraint it neither honours nor refuses, which is the one
outcome with no symptom: a dropped constraint surfaces as a wrong policy rather
than as an error.

Pure data. A disposition records what was decided and why, and carries no
policy of its own — which solver reaches which verdict is declared by that
solver's capabilities and assigned by `_lcm.constraints.capabilities`.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from _lcm.constraints.ir import Compare
from _lcm.constraints.processed import ProcessedConstraint
from _lcm.grids import Grid
from _lcm.typing import FunctionName, RegimeName, StateOrActionName

# Where in a solve a constraint is evaluated. The stages differ in what is in
# scope, which is what decides whether a constraint can be evaluated at all:
# - `"state_action"` — over the whole state-action product, as grid search does
# - `"discrete_combo"` — per discrete combination, before a continuous inner stage
# - `"savings_stage"` — inside an endogenous-grid inversion
# - `"simulation"` — against the simulate-phase feasibility array
type EvaluationStage = Literal[
    "state_action", "discrete_combo", "savings_stage", "simulation"
]


@dataclass(frozen=True, kw_only=True)
class ConstraintContext:
    """What a structural proof or a boundary compiler may read.

    Deliberately smaller than the solver's full build context: a disposition
    has to be decidable from the declaration plus the solver's own
    configuration, so that the solve phase and the simulate phase cannot reach
    different verdicts about the same constraint.
    """

    regime_name: RegimeName
    """Name of the regime whose constraints are being disposed of."""

    phase: Literal["solve", "simulate"]
    """Phase this disposition is being decided for."""

    grids: MappingProxyType[StateOrActionName, Grid]
    """Immutable mapping of the regime's state and action grids."""

    function_names: frozenset[FunctionName]
    """Names the regime's own functions produce."""

    param_names: frozenset[str]
    """Names supplied as parameters rather than computed."""


@dataclass(frozen=True, eq=False)
class Proof:
    """Why a solver's own construction already enforces a constraint."""

    reason: str
    """A sentence naming what enforces it, for a diagnostic to quote."""

    surface: Compare | None = None
    """The comparison the construction enforces, when the proof identified one."""


@dataclass(frozen=True, eq=False)
class BoundaryProgram:
    """A compiled boundary a solver splits its candidate grid on."""

    surfaces: tuple[Compare, ...]
    """The comparisons separating the admitted region from its complement."""

    payload: object
    """Whatever the compiling solver needs to act on the boundary; private
    to that solver, and never interpreted here."""


@dataclass(frozen=True, eq=False)
class Evaluate:
    """The solver calls the constraint at a stage where it can read its inputs."""

    constraint: ProcessedConstraint
    """The constraint to evaluate."""

    stage: EvaluationStage
    """Where the solver evaluates it."""


@dataclass(frozen=True, eq=False)
class ProvedByConstruction:
    """The solver's own construction enforces the constraint, so it is not called."""

    constraint: ProcessedConstraint
    """The constraint the proof discharges."""

    proof: Proof
    """Why the construction already enforces it."""


@dataclass(frozen=True, eq=False)
class CompileBoundary:
    """The solver splits its candidate grid on the constraint's boundary."""

    constraint: ProcessedConstraint
    """The constraint whose boundary was compiled."""

    program: BoundaryProgram
    """The compiled boundary."""


@dataclass(frozen=True, eq=False)
class Reject:
    """The solver can neither evaluate nor discharge the constraint."""

    constraint: ProcessedConstraint
    """The constraint being refused."""

    reason: str
    """A complete user-facing sentence, raised verbatim as the error message."""


type ConstraintDisposition = Evaluate | ProvedByConstruction | CompileBoundary | Reject
