"""Normalize schedule thresholds onto the shared condition representation.

A piecewise-affine bracket edge says that one variable stands on one side of a
threshold. That is the same comparison shape used by a structured constraint
or case boundary, so the schedule normalization emits the shared condition IR.

The normalization moves the exact-equality point out of a metadata field and
into the comparison operator, which is where a reader already looks for it:
`equality="otherwise"` is `<` and `equality="when"` is `<=`. A solver that
reads a declared schedule threshold by its comparison shape rather than a
second boundary representation.
"""

from _lcm.constraints.ir import Compare, ComparisonOperator, Condition, Ref
from lcm.case_piece import AffineBreakpoint, EqualityOwner
from lcm.exceptions import ModelInitializationError

# Which comparison leaves the exact threshold point on the `when` side.
_OPERATOR_OWNING_EQUALITY: dict[EqualityOwner, ComparisonOperator] = {
    "when": "<=",
    "otherwise": "<",
}


def condition_of_breakpoint(*, edge: AffineBreakpoint, variable: str) -> Condition:
    """State one threshold of a piecewise-affine schedule as a condition.

    A bracket edge partitions the schedule's monotone variable exactly as a
    case boundary partitions the liquid axis, so it yields the same shape. The
    segment below the threshold owns the edge's open side, matching the
    `otherwise`-owned convention case boundaries use.

    Args:
        edge: The declared threshold.
        variable: Name of the monotone variable the schedule compares against.

    Returns:
        The below-the-threshold segment, as an inspectable condition.

    Raises:
        ModelInitializationError: If the threshold is read out of a table
            rather than named directly.

    """
    _fail_if_threshold_is_not_a_named_value(edge=edge)
    return _threshold_condition(
        variable=variable,
        threshold=edge.threshold,
        equality_owner=("when" if edge.equality_owner == "below" else "otherwise"),
    )


def _threshold_condition(
    *, variable: str, threshold: str, equality_owner: EqualityOwner
) -> Condition:
    """Compare a variable against a threshold, with the declared owner of equality."""
    return Condition(
        expression=Compare(
            left=Ref(variable),
            op=_OPERATOR_OWNING_EQUALITY[equality_owner],
            right=Ref(threshold),
        )
    )


def _fail_if_threshold_is_not_a_named_value(*, edge: AffineBreakpoint) -> None:
    """Refuse a threshold the condition language has no way to refer to.

    An indexed, sub-keyed, or column-selected threshold is a read out of a
    table at a position, not a named value. Rendering it as a bare reference to
    the table would compare the variable against the whole table and read, to
    every consumer, like an ordinary scalar boundary.

    This is the settled arrangement rather than a gap. A condition earns its
    keep by being *provable*: a threshold known when the model is built can be
    compared against a grid and either proved or refused. A table entry is
    resolved from params at solve time, so a comparison against one could never
    be proved — it would only add a second right-operand kind that every shape
    test has to tell apart from a constant, and the first test that forgot
    would prove something against a value nobody knew yet.

    Such a schedule keeps its own route, which reads the table at the ride-along
    cell and the selected column rather than receiving it flattened into a name.
    The case for revisiting is therefore not that the language could be more
    expressive; it is a schedule that route cannot consume.
    """
    table_valued = {
        "indexed_by": edge.indexed_by,
        "static_index": edge.static_index,
        "threshold_subkey": edge.threshold_subkey,
    }
    declared = sorted(name for name, value in table_valued.items() if value is not None)
    if declared:
        msg = (
            f"The threshold '{edge.threshold}' declares {declared}, so it "
            "is read out of a table rather than named directly. The condition "
            "language refers to values by name and has no spelling for a read "
            "at a position, so this threshold is not statable as a condition. "
            "The schedule stays on the boundary route, which resolves the "
            "table itself; only the constraint path treats it as opaque."
        )
        raise ModelInitializationError(msg)
