"""Regime activity must be evaluated once, through one canonical schedule.

Promoted from the Pro-review reachability audit: the construction-time
reachability graph evaluated `Regime.active` on exact `Fraction` ages
(`AgeGrid.exact_values`), while every other consumer (age specialization,
model-input validation, broadcast pruning) evaluated the same predicate
through `AgeGrid.get_periods_where`, which first rounds each age through a
float32 round-trip. For a sub-annual `AgeGrid` step, the two representations
can disagree on which periods a regime is active in, silently desynchronizing
the reachability graph from every other activity-derived structure.
"""

from jax import numpy as jnp

from lcm import AgeGrid, Model, Regime, categorical
from lcm.typing import ScalarFloat, ScalarInt

# Period 5 of `AgeGrid(start=20, stop=21, step="M")` has exact age
# `Fraction(245, 12)` = 20.416666666666668, but float32(Fraction(245, 12)) =
# 20.41666603088379. This threshold sits strictly between the two, so a
# `>=` predicate classifies period 5 differently depending on which
# representation evaluates it.
_THRESHOLD = 20.41666634877523


@categorical(ordered=False)
class RegimeId:
    solo: ScalarInt
    term: ScalarInt


def _zero_utility() -> ScalarFloat:
    return jnp.float32(0)


def _next_term() -> ScalarInt:
    return RegimeId.term


def _active_from_threshold(age: float) -> bool:
    return age >= _THRESHOLD


def test_subannual_activity_uses_one_canonical_schedule(x64_disabled: None) -> None:
    """The reachability graph's active-period set must equal `get_periods_where`'s.

    Both must classify period 5 the same way; the reachability graph must not
    derive its own, Fraction-based activity set that disagrees with the
    canonical `AgeGrid.get_periods_where`-based set every other subsystem
    (age specialization, model-input validation, broadcast pruning) uses. The
    disagreement only manifests once ages round-trip through float32, so this
    test forces `jax_enable_x64=False` regardless of the suite's `--precision`
    flag.
    """
    del x64_disabled
    ages = AgeGrid(start=20, stop=21, step="M")
    assert float(ages.exact_values[5]) == 245 / 12  # confirms the age of interest

    model = Model(
        regimes={
            "solo": Regime(
                transition=_next_term,
                active=_active_from_threshold,
                functions={"utility": _zero_utility},
            ),
            "term": Regime(transition=None, functions={"utility": _zero_utility}),
        },
        ages=ages,
        regime_id_class=RegimeId,
        enable_jit=False,
    )

    expected_periods = ages.get_periods_where(_active_from_threshold)
    graph_periods = tuple(
        period
        for period, regimes in enumerate(
            model.reachability.solution.active_regimes_by_period
        )
        if "solo" in regimes
    )

    assert graph_periods == expected_periods
