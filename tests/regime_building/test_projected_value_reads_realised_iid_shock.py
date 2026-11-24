"""A gate fallback that reads a shock's node position needs node draws.

A young single moves into a couple through a state-only gate that is open at every
landing, and the gate's fallback projects onto an older single's value. The projection
names the shock in either spelling:

- `wage_shock_f` receives the shock value the household holds;
- `wage_shock_f_code` receives the int32 position of the node it stands on.

A node position exists in simulation only when the shock is drawn on its nodes, so a
model whose projection reads the position of a continuously drawn shock is refused.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    MemberRoute,
    Model,
    NormalIIDProcess,
    ProjectedRegimeValue,
    Regime,
    Transition,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)

type Spelling = Literal["value", "code"]
type Draws = Literal["continuous", "nodes"]

_NODES = np.linspace(-2.0, 2.0, 5)
_BRACKET = 0.25


def _shock(*, spelling: Spelling, draws: Draws | None = None) -> NormalIIDProcess:
    """The shock; a node position exists in simulation only on node draws."""
    return NormalIIDProcess(
        n_points=5,
        gauss_hermite=False,
        mu=0.0,
        sigma=1.0,
        n_std=2.0,
        simulated_draws=draws or ("continuous" if spelling == "value" else "nodes"),
    )


@categorical(ordered=False)
class Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class RegimeId:
    young_f: ScalarInt
    married: ScalarInt
    single_f: ScalarInt


def _cubic(shock: FloatND | np.ndarray) -> FloatND | np.ndarray:
    return shock**3 + 2.0 * shock + 5.0


_NODE_VALUES = _cubic(_NODES)


def _single_f_utility(wage_shock_f: ContinuousState) -> FloatND:
    return jnp.asarray(_cubic(wage_shock_f))


def _married_f(*, wage_shock_f: ContinuousState, work: DiscreteAction) -> FloatND:
    reference = jnp.interp(wage_shock_f, jnp.asarray(_NODES), jnp.asarray(_NODE_VALUES))
    return reference + _BRACKET * (2.0 * work - 1.0)


def _married_m(*, wage_shock_f: ContinuousState, work: DiscreteAction) -> FloatND:
    return 100.0 * (1.0 - work) + 0.0 * wage_shock_f


def _identity(wage_shock_f: ContinuousState) -> ContinuousState:
    return wage_shock_f


def _node_at_code(wage_shock_f_code: DiscreteState) -> ContinuousState:
    """Name the reference node by its position; a non-integer index is refused."""
    return jnp.asarray(_NODES)[wage_shock_f_code]


_PROJECTIONS = {"value": _identity, "code": _node_at_code}


def _always_open(wage_shock_f: ContinuousState) -> BoolND:
    """Every landing enters the couple; the gate only routes the single's role."""
    return jnp.isfinite(wage_shock_f)


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _build_model(
    *, enable_jit: bool, spelling: Spelling, draws: Draws | None = None
) -> Model:
    shock = _shock(spelling=spelling, draws=draws)
    projection = {"wage_shock_f": _PROJECTIONS[spelling]}
    married = Regime(
        states={"wage_shock_f": shock},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _married_f, "m": _married_m})
        },
    )
    single_f = Regime(
        states={"wage_shock_f": shock}, functions={"utility": _single_f_utility}
    )
    young_f = Regime(
        states={"wage_shock_f": shock}, functions={"utility": _single_f_utility}
    )
    return Model(
        edges={
            "young_f": Transition(
                targets={"married": 0, "single_f": 0},
                law=ByAge(cases={0: {"married": StochasticTransition(func=_prob_one)}}),
                gates={
                    "married": Gate(
                        predicate=_always_open,
                        routes={
                            "only": MemberRoute(
                                target_member="f",
                                fallback=ProjectedRegimeValue(
                                    regime="single_f", projection=projection
                                ),
                            )
                        },
                    )
                },
            )
        },
        regimes={"young_f": young_f, "married": married, "single_f": single_f},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "young_f"},
        enable_jit=enable_jit,
    )


def test_a_code_read_of_a_process_drawn_between_nodes_is_refused() -> None:
    """A continuously drawn shock has no node position in simulation."""
    with pytest.raises(RegimeInitializationError, match="simulated_draws='nodes'"):
        _build_model(enable_jit=False, spelling="code", draws="continuous")
