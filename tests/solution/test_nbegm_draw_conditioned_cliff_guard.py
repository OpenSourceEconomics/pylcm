"""NB-EGM refuses a draw-reading liquid law whose child cliffs move with the draw.

The save-to-cliff candidates invert the liquid law at every node of the draws it
reads, against the cliff breakpoints of the source cell. When the child's
breakpoints are indexed by (or derived from) the state that the draw realizes,
each node lands on a different child row with its own cliffs, so the source
cell's breakpoints are the wrong ones. Model construction rejects that
combination; a draw-reading law whose breakpoints do not depend on the drawn
state still builds.
"""

from collections.abc import Callable

import jax.numpy as jnp
import pytest

import lcm
from lcm import DiscreteGrid, LinSpacedGrid, Model
from lcm.exceptions import ModelInitializationError
from lcm.typing import ContinuousState, DiscreteState, FloatND
from tests.test_models.nbegm_common import (
    make_alive_dead_model,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_indexed_threshold_toy import (
    ConsumerKind,
    gross_income,
    resources,
)
from tests.test_models.nbegm_indexed_threshold_toy import (
    subsidy as kind_indexed_subsidy,
)


def next_liquid(*, savings: FloatND, next_kind: DiscreteState) -> ContinuousState:
    """Liquid wealth next period: savings plus a shift set by the drawn kind."""
    return savings + 0.1 * next_kind


def kind_probabilities(*, kind: DiscreteState) -> FloatND:
    """Next-period kind is a fair coin, whatever the current kind."""
    return jnp.asarray(((0.5, 0.5), (0.5, 0.5)))[kind]


def zero_bequest(*, liquid: ContinuousState) -> FloatND:
    """The terminal regime values remaining wealth at zero."""
    return jnp.zeros_like(liquid)


@lcm.piecewise_affine(
    output="subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint(threshold="fpl_cliff", kind="jump"),),
)
def kind_invariant_subsidy(
    *,
    gross_income: FloatND,
    subsidy_low: float,
    subsidy_high: float,
    fpl_cliff: float,
) -> FloatND:
    """Lump-sum subsidy with one income cliff shared by both kinds."""
    return jnp.where(gross_income < fpl_cliff, subsidy_high, subsidy_low)


def _build_model(*, subsidy: Callable[..., object]) -> Model:
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=31,
        liquid_max=30.0,
        n_consumption=31,
        liquid_grid=LinSpacedGrid(start=0.0, stop=30.0, n_points=31),
        alive_functions={
            "utility": utility,
            "gross_income": gross_income,
            "subsidy": subsidy,
            "resources": resources,
            "savings": savings,
        },
        liquid_law=next_liquid,
        alive_solver=resolve_solver(
            variant="nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=28.0, n_points=16),
        ),
        constraints={},
        extra_states={"kind": DiscreteGrid(category_class=ConsumerKind)},
        extra_state_transitions={
            "kind": lcm.StochasticTransition(func=kind_probabilities)
        },
        dead_functions={"utility": zero_bequest},
    )


def test_nbegm_rejects_draw_reading_law_with_draw_indexed_child_cliffs() -> None:
    """A liquid law reading `next_kind` with a `kind`-indexed cliff is refused."""
    with pytest.raises(
        ModelInitializationError,
        match=r"(?s)'alive'.*'kind'.*'next_kind'.*not supported yet",
    ):
        _build_model(subsidy=kind_indexed_subsidy)


def test_nbegm_builds_draw_reading_law_with_draw_invariant_cliffs() -> None:
    """A liquid law reading `next_kind` with a kind-invariant cliff still builds."""
    model = _build_model(subsidy=kind_invariant_subsidy)
    assert set(model.user_regimes) == {"alive", "dead"}
