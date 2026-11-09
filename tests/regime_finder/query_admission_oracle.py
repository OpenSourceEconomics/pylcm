"""Structural oracle for physical reachability, separate from query demand.

This oracle takes literal finite relation sets, not the normalizer's output.
It neither imports production analysis nor evaluates an economic callback.
"""


def physical_closure(initial_nodes, scheduled_moves, settlements=()):
    """Least closure of physical moves from explicitly admitted initial nodes.

    Each move is ((source_period, source_owner), (target_period, target_owner)).
    Policy-read relations deliberately are not an argument to this operation.
    """
    reached = set(initial_nodes)
    moves = tuple(scheduled_moves) + tuple(settlements)
    changed = True
    while changed:
        additions = {target for source, target in moves if source in reached}
        changed = not additions <= reached
        reached.update(additions)
    return reached


def demanded_owners(physical, reads):
    """Least phase-specific owner closure of literal read edges.

    A read is ((period, owner, phase), (period, owner, phase)). An owner that is
    already physical is available without manufacturing a query-only cell.
    """
    available = {(period, owner, phase) for period, owner in physical
                 for phase in ("solve", "simulation")}
    demanded = set()
    changed = True
    while changed:
        additions = {target for source, target in reads
                     if source in available or source in demanded}
        additions.difference_update(available)
        changed = not additions <= demanded
        demanded.update(additions)
    return demanded


if __name__ == "__main__":
    physical = physical_closure({(0, "A")}, [((0, "A"), (1, "B"))])
    queries = demanded_owners(physical, [((0, "A", "solve"), (0, "Q", "solve")),
                                        ((0, "Q", "solve"), (0, "R", "solve"))])
    assert physical == {(0, "A"), (1, "B")}
    assert queries == {(0, "Q", "solve"), (0, "R", "solve")}
    print("Physical visits and recursive query demands remain distinct.")
