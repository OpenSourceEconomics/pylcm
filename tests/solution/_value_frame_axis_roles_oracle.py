"""Standard-library scalar reference for collective value-frame rows.

The reference reads no solution result, metadata, coordinate builder or serializer.
"""


def expected_collective_rows(
    *, stakeholder_names: tuple[str, ...]
) -> tuple[tuple[float, str, float], ...]:
    """Enumerate the two feasible actions at each literal wealth coordinate.

    The household objective is the arithmetic mean of `f`'s utility
    `wealth + 3 * action` and `m`'s utility `2 * wealth + 5 * action`. Both strictly
    increase with the action, so action 1 uniquely wins; the tie key keeps the rule
    explicit although no tie occurs. Stakeholder order is supplied by the caller and
    never inferred from result metadata.
    """
    rows = []
    for wealth in (1.0, 2.0):
        candidates = []
        for action in (0, 1):
            payoffs = {"f": wealth + 3.0 * action, "m": 2.0 * wealth + 5.0 * action}
            household_value = (payoffs["f"] + payoffs["m"]) / 2.0
            candidates.append((household_value, -action, payoffs))
        chosen = max(candidates, key=lambda candidate: candidate[:2])[2]
        rows.extend(
            (wealth, stakeholder, chosen[stakeholder])
            for stakeholder in stakeholder_names
        )
    return tuple(rows)
