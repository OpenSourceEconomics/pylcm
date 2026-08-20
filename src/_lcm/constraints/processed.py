"""One normalized form for every constraint a regime declares.

Whatever a user writes — a condition built from references, a declared
post-decision bound, or a bare predicate — normalization produces a
`ProcessedConstraint` carrying the same four things: its name, what it says,
how to evaluate it, and which names it reads. A solver therefore never has to
ask what kind of constraint it is holding before it can evaluate one, and never
has to work out for itself which values it must have available.

What varies is how much structure survives. A bare predicate normalizes to an
opaque condition, which a solver needing structure can recognise and refuse
rather than accept and silently ignore. A declared post-decision bound
normalizes to the comparison it stands for, which is what lets a solver prove
it against its own savings grid.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from _lcm.constraints.ir import (
    Compare,
    Condition,
    Const,
    Ref,
    conjuncts,
    dependencies_of,
)
from _lcm.post_decision_bound import _PostDecisionLowerBound
from _lcm.typing import ConstraintFunction, FunctionName
from lcm.typing import BoolND, UserFunction, ValueND

# What a regime's `constraints` slot accepts, before normalization.
type ConstraintLike = Condition | UserFunction


@dataclass(frozen=True, eq=False)
class ProcessedConstraint:
    """A constraint in the one form every solver consumes.

    Holds the condition rather than a separately built predicate: the evaluator
    is generated from the condition on demand, so there is no second object to
    fall out of step with what the constraint says.
    """

    name: FunctionName
    """The name the constraint was declared under."""

    condition: Condition
    """What the constraint says, as an inspectable expression."""

    declaration: ConstraintLike
    """The object the user handed over, kept as written.

    Age specialization and the pruning walk both recognise a declaration by its
    own type, so reading a constraint must not cost the caller access to what
    was declared.
    """

    @property
    def dependencies(self) -> frozenset[str]:
        """Frozenset of names that must be available to evaluate the constraint."""
        return dependencies_of(self.condition.expression)

    @property
    def arg_names(self) -> tuple[str, ...]:
        """Tuple of the names the constraint is called with, in a stable order."""
        return self.condition.arg_names

    @property
    def boundary_surfaces(self) -> tuple[Compare, ...] | None:
        """Tuple of comparisons bounding the admitted region, or `None`.

        `None` says the condition does not decompose into surfaces — it is an
        `or`, a negation, an implication, or an opaque predicate — and must
        never be read as "this constraint has no boundaries".
        """
        return conjuncts(self.condition.expression)

    def as_function(
        self, *, pool: Mapping[FunctionName, UserFunction] | None = None
    ) -> ConstraintFunction:
        """Build the DAG-composable predicate this constraint is evaluated through.

        Args:
            pool: The regime's functions, whose annotations the result adopts.
                Without one every argument is annotated as a continuous value,
                which is enough wherever the result is only walked for its
                dependencies rather than composed.

        Returns:
            A callable whose signature names the constraint's dependencies.

        """
        from _lcm.constraints.materialize import (  # noqa: PLC0415
            as_constraint_function,
        )

        return as_constraint_function(constraint=self, pool=pool or {})

    @property
    def is_opaque(self) -> bool:
        """Whether the constraint offers a solver no structure to reason about."""
        return self.condition.is_opaque

    def evaluate(self, **values: ValueND) -> BoolND:
        """Evaluate the constraint on the supplied values.

        Args:
            **values: One entry per dependency.

        Returns:
            The elementwise feasibility the constraint admits.

        """
        return self.condition.evaluate(**values)


def normalize_constraints(
    *,
    constraints: Mapping[FunctionName, ConstraintLike],
) -> MappingProxyType[FunctionName, ProcessedConstraint]:
    """Bring every declared constraint into the form solvers consume.

    Args:
        constraints: The regime's constraints as the user declared them.

    Returns:
        Immutable mapping of constraint name to its normalized form.

    """
    return MappingProxyType(
        {
            name: ProcessedConstraint(
                name=name,
                condition=_as_condition(declaration),
                declaration=declaration,
            )
            for name, declaration in constraints.items()
        }
    )


def _as_condition(declaration: ConstraintLike) -> Condition:
    """Recover as much structure from a declaration as it carries."""
    if isinstance(declaration, Condition):
        return declaration
    if isinstance(declaration, _PostDecisionLowerBound):
        # The declared bound stands for a comparison, and a solver can only
        # prove it against its savings grid if it arrives as one.
        return Condition(
            expression=Compare(
                left=Ref(declaration.post_decision),
                op=">=",
                right=Const(declaration.lower_bound),
            )
        )
    return Condition.from_callable(declaration)
