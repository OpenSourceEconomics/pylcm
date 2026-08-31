"""A joint lottery's probabilities must have exactly one row per source cell.

A probability function that reads no grid variable describes one lottery, so it
owes exactly one vector over the support. One that reads grid variables owes one
vector per cell of the mesh those variables span. An extra leading axis has no
declared source variable, so nothing downstream can say which cell each of its
rows belongs to — the preflight refuses it rather than letting the sampler pick
an axis by shape.
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    JointTransition,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.exceptions import InvalidStateTransitionProbabilitiesError
from lcm.regime import Regime
from lcm.typing import DiscreteAction, FloatND, ScalarInt

_SUPPORT = jnp.asarray([1.0, 2.0])


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Effort:
    low: ScalarInt
    high: ScalarInt


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 61, RegimeId.working, RegimeId.dead)


def _utility(wealth: float, effort: DiscreteAction) -> FloatND:
    return jnp.asarray(wealth) - 0.1 * effort


def _next_wealth_from_match(match: dict[str, FloatND]) -> FloatND:
    return match["value"]


def _one_vector() -> FloatND:
    """One lottery over the two support nodes: the well-formed no-grid case."""
    return jnp.asarray([0.5, 0.5])


def _undeclared_event_axis() -> FloatND:
    """Rows summing to one, but with a leading axis no argument declares."""
    return jnp.asarray([[0.5, 0.5], [0.3, 0.7]])


def _one_vector_per_effort(effort: DiscreteAction) -> FloatND:
    """One lottery per cell of the mesh spanned by the arguments it reads."""
    return jnp.where(
        effort == Effort.high,
        jnp.asarray([0.3, 0.7]),
        jnp.asarray([0.5, 0.5]),
    )


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "working": {"koopmans_aggregator": {"discount_factor": 0.95}},
        "dead": {},
    }


def _build_model(*, probabilities) -> Model:
    working = Regime(
        transition=_next_regime,
        active=lambda age: age < 64,
        states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=3)},
        actions={"effort": DiscreteGrid(Effort)},
        functions={"utility": _utility},
        joint_transitions={
            "working": {
                "match": JointTransition(
                    support_size=2,
                    support={"value": _SUPPORT},
                    probabilities=probabilities,
                    outputs={"wealth": _next_wealth_from_match},
                )
            }
        },
    )
    dead = Regime(transition=None, functions={"utility": lambda: jnp.asarray(0.0)})
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=RegimeId,
    )


def test_a_probability_function_reading_no_grid_owes_exactly_one_vector() -> None:
    """A zero-argument probability function returning a matrix is refused."""
    model = _build_model(probabilities=_undeclared_event_axis)

    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="shape"):
        model.solve(params=_params(), log_level="debug")


@pytest.mark.parametrize(
    "probabilities", [_one_vector, _one_vector_per_effort], ids=["no_grid", "per_cell"]
)
def test_a_correctly_shaped_probability_function_is_accepted(probabilities) -> None:
    """One vector without grid arguments, one per cell with them, both solve."""
    model = _build_model(probabilities=probabilities)

    period_to_regime_to_V_arr = model.solve(params=_params(), log_level="debug")

    assert set(period_to_regime_to_V_arr[0]) == {"working", "dead"}
