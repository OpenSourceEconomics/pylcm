"""A coarse state law read by within-period utility is not a target-dependent conflict.

A single BARE (coarse) `next_<state>` law carried by several targets is canonicalized
into one cell per target, and each cell is renamed with its own qualified parameter
names -- producing DISTINCT wrapper objects that are nonetheless the same user law.
The within-period conflict guard must compare the raw source (via `__wrapped__`), not
processed wrapper identity, or it falsely rejects the coarse-broadcast case. Genuinely
different per-target laws read by utility must still be rejected.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model, categorical
from lcm.regime import Regime
from lcm.typing import FloatND, ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    work: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _prob(age: int) -> ScalarFloat:  # noqa: ARG001
    return jnp.asarray(0.5)


_WEALTH = LinSpacedGrid(start=1.0, stop=100.0, n_points=8)
_CONS = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
_PARAMS = {
    "work": {"discount_factor": 0.95, "growth": 1.01},
    "retired": {"discount_factor": 0.95, "growth": 1.01},
}


def _utility_reads_next(consumption: float, next_wealth: float) -> FloatND:
    return jnp.log(consumption) + 0.0 * next_wealth


def _dead() -> Regime:
    return Regime(transition=None, functions={"utility": lambda: 0.0})


def test_coarse_parameterized_law_read_by_utility_is_accepted():
    """The F8 case: a PARAMETERIZED bare law carried by two targets (so it is
    canonicalized into two per-target renamed wrappers) read by utility. Same law,
    so no target-dependent conflict -- it must solve.
    """

    def next_wealth(wealth: float, consumption: float, growth: float) -> float:
        return (wealth - consumption) * growth

    def regime() -> Regime:
        # self-loop + retired: `wealth` is carried by two targets.
        return Regime(
            transition={
                "work": MarkovTransition(_prob),
                "retired": MarkovTransition(_prob),
            },
            active=lambda age: age < 2,
            states={"wealth": _WEALTH},
            actions={"consumption": _CONS},
            state_transitions={"wealth": next_wealth},
            functions={"utility": _utility_reads_next},
        )

    work = regime()
    retired = Regime(
        transition={
            "retired": MarkovTransition(_prob),
            "dead": MarkovTransition(_prob),
        },
        active=lambda age: age < 2,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONS},
        state_transitions={"wealth": next_wealth},
        functions={"utility": _utility_reads_next},
    )
    model = Model(
        regimes={"work": work, "retired": retired, "dead": _dead()},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )
    # Must not raise a "target-dependent deterministic state law" ValueError.
    model.solve(params=_PARAMS, log_level="off")


def test_genuinely_different_per_target_laws_read_by_utility_still_rejected():
    """The guard is preserved: two DIFFERENT per-target `next_wealth` laws read by
    utility must still be rejected, since the merged decision DAG would bind one
    while the simulate state-update uses the other.
    """

    def grow(wealth: float, consumption: float, growth: float) -> float:
        return (wealth - consumption) * growth

    def shrink(wealth: float, consumption: float, growth: float) -> float:
        return (wealth - consumption) / growth

    work = Regime(
        transition={
            "work": MarkovTransition(_prob),
            "retired": MarkovTransition(_prob),
        },
        active=lambda age: age < 2,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONS},
        # genuinely different laws per target
        state_transitions={"wealth": {"work": grow, "retired": shrink}},
        functions={"utility": _utility_reads_next},
    )
    retired = Regime(
        transition={
            "retired": MarkovTransition(_prob),
            "dead": MarkovTransition(_prob),
        },
        active=lambda age: age < 2,
        states={"wealth": _WEALTH},
        actions={"consumption": _CONS},
        state_transitions={"wealth": {"retired": grow, "dead": grow}},
        functions={"utility": _utility_reads_next},
    )
    # The conflict is caught while the Q-and-F decision DAG is assembled, at model
    # build (before any solve).
    with pytest.raises(ValueError, match="target-dependent deterministic state law"):
        Model(
            regimes={"work": work, "retired": retired, "dead": _dead()},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=RegimeId,
        )
