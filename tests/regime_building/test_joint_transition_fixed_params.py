"""A joint lottery whose probabilities read a fixed parameter is validated with it.

`log_level="debug"` evaluates every transition-local lottery before solving. A
probability function reading a value supplied through `Model(fixed_params=...)`
is evaluated with that value, exactly as with the same value supplied at solve
time, so a valid table solves and an invalid one is refused for its rows.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DeterministicTransition,
    JointTransition,
    LinSpacedGrid,
    Model,
    Transition,
    categorical,
)
from lcm.exceptions import InvalidStateTransitionProbabilitiesError
from lcm.regime import Regime
from lcm.typing import FloatND, ScalarInt, UserParamsNode

_VALID_TABLE = jnp.asarray([0.25, 0.75])
_INVALID_TABLE = jnp.asarray([1.2, -0.2])
_PARAMS = {"working": {"koopmans_aggregator": {"discount_factor": 0.95}}, "dead": {}}


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 62, RegimeId.working, RegimeId.dead)


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _consumption_leq_wealth(*, consumption: float, wealth: float) -> bool:
    return consumption <= wealth


def _probabilities_from_table(match_table: FloatND) -> FloatND:
    return match_table


def _next_wealth_from_match(match: dict[str, FloatND]) -> FloatND:
    return match["value"]


def _table_path(table: FloatND) -> dict[str, UserParamsNode]:
    """The table at its declaration path below the source regime."""
    return {"working": {"working": {"match": _probabilities(table)}}}


def _probabilities(table: FloatND) -> dict[str, UserParamsNode]:
    return {"probabilities": {"match_table": table}}


def _build_model(*, fixed_table: FloatND | None) -> Model:
    working = Regime(
        states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=3)},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=3)},
        constraints={"feasible_consumption": _consumption_leq_wealth},
        functions={"utility": _utility},
        joint_transitions={
            "working": {
                "match": JointTransition(
                    support_size=2,
                    support={"value": jnp.asarray([2.0, 6.0])},
                    probabilities=_probabilities_from_table,
                    outputs={"wealth": _next_wealth_from_match},
                )
            }
        },
    )
    dead = Regime(functions={"utility": lambda: jnp.asarray(0.0)})
    return Model(
        edges={
            "working": Transition(
                targets={"working": (60, 61), "dead": (60, 61, 62)},
                law=DeterministicTransition(func=_next_regime),
            )
        },
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=60, inclusive_stop=63, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={60: "working"},
        fixed_params={} if fixed_table is None else _table_path(fixed_table),
    )


def test_a_fixed_joint_probability_table_solves_like_a_free_one() -> None:
    """Debug validation reads the fixed table; both models give the same values."""
    fixed = _build_model(fixed_table=_VALID_TABLE).solve(
        params=_PARAMS, log_level="debug"
    )
    free = _build_model(fixed_table=None).solve(
        params={
            "working": {
                "koopmans_aggregator": {"discount_factor": 0.95},
                "working": {"match": _probabilities(_VALID_TABLE)},
            },
            "dead": {},
        },
        log_level="debug",
    )
    np.testing.assert_allclose(
        np.asarray(fixed.values[0]["working"]),
        np.asarray(free.values[0]["working"]),
        rtol=1e-6,
    )


def test_an_invalid_fixed_joint_probability_table_is_refused_for_its_rows() -> None:
    """A fixed table `[1.2, -0.2]` is refused as out of range in debug."""
    with pytest.raises(
        InvalidStateTransitionProbabilitiesError, match="out-of-range values"
    ):
        _build_model(fixed_table=_INVALID_TABLE).solve(
            params=_PARAMS, log_level="debug"
        )
