"""A gated edge's target DAG pool holds concrete functions, never age markers.

An edge's gate and projections are compiled against the target regime's nodes.
The mapping handed over is the target's PUBLISHED function set, which is
resolved at the regime's representative age — so every node in the pool is one
concrete callable, the same one at every period the edge folds at.

That is what makes it safe to group compiled folds by grid signature alone: no
node in the pool varies with age, so nothing but the grids can. A pool carrying
an unresolved `PeriodizedEconFunction` would break that silently rather than
loudly, because the fold's grouping has no reason to consult a function's
signature and a frozen representative-age closure returns a number like any
other. Pinned here: publication resolves the marker.
"""

from _lcm.regime_building.age_normalization import PeriodizedEconFunction
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.typing import ContinuousState, FloatND, ScalarInt

_AGES = AgeGrid(start=0, inclusive_stop=3, step="Y")
_WEALTH = LinSpacedGrid(start=0.0, stop=4.0, n_points=2)

# The bonus scale changes once over the worker's active ages, so the marker
# resolves to two distinct closures and is genuinely periodized rather than
# collapsing to a single shared program.
_BONUS_SCALE_EARLY = 1.0
_BONUS_SCALE_LATE = 2.0


@categorical(ordered=False)
class RegimeId:
    """Regime ids of the worker / dead model in this module."""

    worker: ScalarInt  # code 0
    dead: ScalarInt  # code 1


def test_published_functions_resolve_an_age_specialized_marker():
    """A regime's published solve functions hold a concrete callable per name.

    This is the property the pool guard relies on staying true, and it is why
    that guard stays quiet on every model pylcm can build today.
    """
    canonical = _build_model()._regimes["worker"]

    periodized = sorted(
        name
        for name, func in canonical.solution.functions.items()
        if isinstance(func, PeriodizedEconFunction)
    )
    assert periodized == []


def test_the_marker_really_was_periodized_in_that_build():
    """The control for the pin above: `bonus` did vary by age in that model.

    Without it, `test_published_functions_resolve_an_age_specialized_marker`
    would also pass on a build where nothing was ever age-specialized, which
    would make it a test of the model rather than of publication.
    """
    canonical = _build_model()._regimes["worker"]

    assert canonical.simulation.age_specialized_function_names == frozenset({"bonus"})


def _build_model() -> Model:
    """Build a worker whose `bonus` helper is bound per age."""
    worker = Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": {"dead": _keep_wealth}},
        functions={
            "utility": _utility,
            "bonus": AgeSpecializedFunction(build=_make_bonus, signature=_bonus_scale),
        },
    )
    dead = Regime(
        states={"wealth": _WEALTH},
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"worker": worker, "dead": dead},
        edges={"worker": {"dead": (0, 1, 2)}},
        ages=_AGES,
        regime_id_class=RegimeId,
        initial_nodes={0: "worker"},
    )


def _bonus_scale(age: float) -> float:
    """The factor the bonus pays at this age; also the marker's dedup key."""
    return _BONUS_SCALE_EARLY if age <= 1 else _BONUS_SCALE_LATE


def _make_bonus(age: float):
    """Build the bonus closure for one age."""
    scale = _bonus_scale(age)

    def bonus(wealth: ContinuousState) -> FloatND:
        return scale * wealth

    return bonus


def _keep_wealth(wealth: ContinuousState) -> ContinuousState:
    """Wealth carries over unchanged into the target regime."""
    return wealth


def _utility(*, wealth: ContinuousState, bonus: FloatND) -> FloatND:
    """The worker is paid its wealth plus the age's bonus."""
    return wealth + bonus


def _terminal_utility(wealth: ContinuousState) -> FloatND:
    """The terminal regime pays out its wealth."""
    return wealth
