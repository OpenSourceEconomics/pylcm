"""A projected reference value reads the IID shock a household realises.

A young single, a couple and an older single carry the same
`NormalIIDProcess` state `wage_shock_f`, and the young single moves into the
couple through a state-only gate that is open at every landing. The older
single's value is read through a projection in the couple's participation
constraint, at the couple's own cell. A value read at the arrival itself is an
`ArrivalDecision`, covered by
`tests/regime_building/test_continuous_arrival_physical_value.py`.

A projection names the shock in either spelling:

- `wage_shock_f` receives the shock value the household holds;
- `wage_shock_f_code` receives the int32 position of the node it stands on.

The single's value is a cubic of the shock, so reading it at any coordinate
other than the realised shock -- a node position read as a value, the shock's
mean, another node -- moves it by at least one unit. The oracle is
`numpy.interp` over the cubic at the quadrature nodes, which is what the
single's terminal value function holds.

The wife's household payoff brackets that oracle: working pays her the oracle
plus `_BRACKET`, staying home the oracle minus `_BRACKET`, and the husband
strongly prefers staying home. So the household works exactly when the reference
read lies within `_BRACKET` of the oracle; it stays home when the read is too
low; and it has no feasible action when the read is too high.
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
    Model,
    NormalIIDProcess,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    ValueDependentConstraint,
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
        simulate_draws=draws or ("continuous" if spelling == "value" else "nodes"),
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


def _oracle(shock: np.ndarray) -> np.ndarray:
    """The single's terminal value at a realised shock, linear between nodes."""
    return np.interp(shock, _NODES, _NODE_VALUES)


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


def _participation_f(*, Q_f: FloatND, V_single_f: FloatND) -> BoolND:
    return Q_f >= V_single_f


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
        constraints={
            "participation_f": ValueDependentConstraint(
                predicate=_participation_f,
                references={
                    "V_single_f": ProjectedRegimeValue(
                        regime="single_f", projection=projection
                    )
                },
            )
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
                            "only": StakeholderRoute(
                                target_stakeholder="f",
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


_PARAMS = {
    "young_f": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "married": {},
    "single_f": {},
}


def _solve(*, enable_jit: bool, spelling: Spelling):
    return _build_model(enable_jit=enable_jit, spelling=spelling).solve(
        params=_PARAMS, log_level="off"
    )


def _simulate(*, enable_jit: bool, spelling: Spelling, n_subjects: int = 64):
    model = _build_model(enable_jit=enable_jit, spelling=spelling)
    solution = model.solve(params=_PARAMS, log_level="off")
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wage_shock_f": jnp.zeros(n_subjects),
            "age": jnp.zeros(n_subjects),
            "regime_id": jnp.full(
                n_subjects, model.regime_names_to_ids["young_f"], jnp.int32
            ),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )
    return result.to_dataframe().query("period == 1").sort_values("subject_id")


_ENABLE_JIT = pytest.mark.parametrize("enable_jit", [True, False], ids=["jit", "eager"])
_SPELLING = pytest.mark.parametrize("spelling", ["value", "code"])


@_SPELLING
@_ENABLE_JIT
def test_value_constraint_reads_the_node_in_solve(
    *, enable_jit: bool, spelling: Spelling
):
    """At every node the couple works: the wife holds the oracle plus the bracket."""
    solution = _solve(enable_jit=enable_jit, spelling=spelling)
    V = np.asarray(solution.values.materialize()[1]["married"])
    expected = np.stack([_oracle(_NODES) + _BRACKET, np.zeros(5)], axis=-1)
    np.testing.assert_allclose(V, expected, atol=1e-4)


@_SPELLING
@_ENABLE_JIT
def test_simulated_couples_read_their_realised_shock(
    *, enable_jit: bool, spelling: Spelling
):
    """Every simulated couple works, so each read matched its own realised shock."""
    panel = _simulate(enable_jit=enable_jit, spelling=spelling).query(
        "regime_name == 'married'"
    )
    assert len(panel) > 0
    assert panel["work"].tolist() == ["on"] * len(panel)


def test_continuous_draws_are_off_node_and_span_both_signs():
    """The draws are realised values, not nodes, of both signs."""
    shocks = _simulate(enable_jit=True, spelling="value")["wage_shock_f"].to_numpy()
    assert (shocks < -0.5).any()
    assert (shocks > 0.5).any()
    assert not np.isin(shocks, _NODES).any()


def test_node_draws_reach_every_node():
    """Node draws land on each of the five nodes, so every position is read."""
    shocks = _simulate(enable_jit=True, spelling="code", n_subjects=512)[
        "wage_shock_f"
    ].to_numpy()
    np.testing.assert_array_equal(np.unique(shocks), _NODES)


def test_a_code_read_of_a_process_drawn_between_nodes_is_refused():
    """A continuously drawn shock has no node position in simulation."""
    with pytest.raises(RegimeInitializationError, match="simulate_draws='nodes'"):
        _build_model(enable_jit=False, spelling="code", draws="continuous")
