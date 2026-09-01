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
other. Both halves are pinned here: that publication resolves the marker, and
that the pool refuses one if it ever stops.
"""

import re
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.regime_building.age_normalization import PeriodizedEconFunction
from _lcm.regime_building.gated_edges import _build_target_dag_pool
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import ContinuousState, FloatND, ScalarInt

_AGES = AgeGrid(start=0, stop=3, step="Y")
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


def test_target_dag_pool_refuses_an_age_specialized_node():
    """A pool node that still varies by age is refused, naming it.

    Publication resolves every marker, so this fires only if that stops — at
    which point a fold would freeze one age's closure into every period.
    """
    with pytest.raises(
        ModelInitializationError,
        match=re.escape(
            "the DAG pool of target regime 'dead' carries age-specialized "
            "node(s) ['bonus']"
        ),
    ):
        _build_target_dag_pool(
            target_functions=MappingProxyType({"bonus": _periodized_bonus()}),
            target_deterministic_transitions=MappingProxyType({}),
            edge_target="dead",
        )


def test_target_dag_pool_accepts_concrete_nodes():
    """The control: the same pool of plain callables is built without complaint.

    Without it, a guard that rejected every pool would pass the test above.
    """
    pool = _build_target_dag_pool(
        # A published node is an ordinary model function; the engine's
        # `EconFunction` protocol is wider than any one of them.
        target_functions=MappingProxyType({"bonus": _plain_bonus}),  # ty: ignore[invalid-argument-type]
        target_deterministic_transitions=MappingProxyType({}),
        edge_target="dead",
    )

    assert sorted(pool) == ["bonus"]


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


def _periodized_bonus() -> PeriodizedEconFunction:
    """An unresolved marker of the kind publication is supposed to remove."""
    return PeriodizedEconFunction(
        representative=_plain_bonus,  # ty: ignore[invalid-argument-type]
        function_by_signature=MappingProxyType({_BONUS_SCALE_EARLY: _plain_bonus}),  # ty: ignore[invalid-argument-type]
        signature_by_period=MappingProxyType({0: _BONUS_SCALE_EARLY}),
    )


def _plain_bonus(wealth: ContinuousState) -> FloatND:
    """A concrete node of the kind a published function set holds."""
    return _BONUS_SCALE_EARLY * wealth


def _build_model() -> Model:
    """Build a worker whose `bonus` helper is bound per age."""
    worker = Regime(
        transition={"dead": MarkovTransition(_prob_one)},
        active=lambda age: age < 3,
        states={"wealth": _WEALTH},
        state_transitions={"wealth": {"dead": _keep_wealth}},
        functions={
            "utility": _utility,
            "bonus": AgeSpecializedFunction(build=_make_bonus, signature=_bonus_scale),
        },
    )
    dead = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": _WEALTH},
        functions={"utility": _terminal_utility},
    )
    return Model(
        regimes={"worker": worker, "dead": dead},
        ages=_AGES,
        regime_id_class=RegimeId,
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


def _prob_one(age: FloatND) -> FloatND:
    """Regime transition taken with certainty."""
    return jnp.ones_like(age, dtype=float)


def _keep_wealth(wealth: ContinuousState) -> ContinuousState:
    """Wealth carries over unchanged into the target regime."""
    return wealth


def _utility(*, wealth: ContinuousState, bonus: FloatND) -> FloatND:
    """The worker is paid its wealth plus the age's bonus."""
    return wealth + bonus


def _terminal_utility(wealth: ContinuousState) -> FloatND:
    """The terminal regime pays out its wealth."""
    return wealth
