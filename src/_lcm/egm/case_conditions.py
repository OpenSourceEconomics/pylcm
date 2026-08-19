"""Case boundaries and schedule thresholds, as conditions.

A case boundary declares that one variable stands on one side of a threshold,
and so does a piecewise-affine bracket edge. That is the same assertion a
constraint makes, so both normalize onto the shared condition IR and a solver
reads one kind of object rather than three.

The normalization moves the exact-equality point out of a metadata field and
into the comparison operator, which is where a reader already looks for it:
`equality="otherwise"` is `<` and `equality="when"` is `<=`. A solver that
proves a declared boundary therefore proves it by its shape, and a boundary
written by hand as a condition gets the same treatment as a decorated one.
"""

from _lcm.constraints.ir import Compare, ComparisonOperator, Condition, Ref
from lcm.case_piece import AffineBreakpoint, BoundarySurface, EqualityOwner
from lcm.exceptions import ModelInitializationError

# Which comparison leaves the exact threshold point on the `when` side.
_OPERATOR_OWNING_EQUALITY: dict[EqualityOwner, ComparisonOperator] = {
    "when": "<=",
    "otherwise": "<",
}


def condition_of_boundary_surface(*, surface: BoundarySurface) -> Condition:
    """State one equality surface of a case boundary as a condition.

    The condition is the `when` side of the split: it holds where the boundary
    predicate holds, with the exact threshold point included exactly when the
    surface declares `when` owns it.

    Args:
        surface: The declared equality surface.

    Returns:
        The `when` side, as an inspectable condition.

    """
    return _threshold_condition(
        variable=surface.variable,
        threshold=surface.threshold,
        equality_owner=surface.equality_owner,
    )


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
        equality_owner="otherwise",
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
    """Refuse a threshold the IR has no way to refer to.

    An indexed, sub-keyed, or column-selected threshold is a read out of a
    table at a position, not a named value. Rendering it as a bare reference to
    the table would compare the variable against the whole table and read, to
    every consumer, like an ordinary scalar boundary.
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
            "language refers to values by name and has no spelling for that "
            "read, so the threshold cannot be stated as a condition; a solver "
            "that needs one must take this schedule through the route that "
            "resolves the table itself."
        )
        raise ModelInitializationError(msg)
