"""A projected reference value reads the IID shock a household realises.

A young single, a couple and an older single carry the same
`NormalIIDProcess` state `wage_shock_f`, and the young single moves into the
couple. The older single's value is read through a projection in two places:

- the couple's participation constraint, at the couple's own cell;
- a gate on the edge from the young single into the couple, at the couple's
  landing, whose closed branch puts the row into the older single regime.

A projection and a gate name the shock in either spelling, in every context:

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
from lcm.solver_api import DISSOLUTION_FLAG
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


def _reads_the_oracle(*, V_target_f: FloatND, V_single_f_ref: FloatND) -> BoolND:
    """A working wife's surplus over a correct read is `_BRACKET`."""
    return jnp.abs(V_target_f - V_single_f_ref - _BRACKET) <= 0.5 * _BRACKET


def _gate_on_value(
    *, V_target_f: FloatND, V_single_f_ref: FloatND, wage_shock_f: ContinuousState
) -> BoolND:
    """Open where the landing's shock is positive and the read is the oracle."""
    reads = _reads_the_oracle(V_target_f=V_target_f, V_single_f_ref=V_single_f_ref)
    return reads & (wage_shock_f > 0.0)


def _gate_on_code(
    *, V_target_f: FloatND, V_single_f_ref: FloatND, wage_shock_f_code: DiscreteState
) -> BoolND:
    """Open where the landing stands on a positive node and the read is the oracle."""
    reads = _reads_the_oracle(V_target_f=V_target_f, V_single_f_ref=V_single_f_ref)
    return reads & (wage_shock_f_code >= 3)


_GATES = {"value": _gate_on_value, "code": _gate_on_code}


def _no_dissolution(D_target: BoolND) -> BoolND:
    return ~D_target


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _build_model(
    *, gated: bool, enable_jit: bool, spelling: Spelling, draws: Draws | None = None
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
    references = (
        {
            "V_single_f_ref": ProjectedRegimeValue(
                regime="single_f", projection=projection
            )
        }
        if gated
        else {}
    )
    gates = {
        "married": Gate(
            predicate=_GATES[spelling] if gated else _no_dissolution,
            routes={
                "only": StakeholderRoute(
                    target_stakeholder="f",
                    fallback=ProjectedRegimeValue(
                        regime="single_f", projection=projection
                    ),
                )
            },
            references=references,
        )
    }
    return Model(
        edges={
            "young_f": Transition(
                targets={"married": 0, "single_f": 0},
                law=ByAge(cases={0: {"married": StochasticTransition(func=_prob_one)}}),
                gates=gates,
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


def _solve(*, gated: bool, enable_jit: bool, spelling: Spelling):
    return _build_model(gated=gated, enable_jit=enable_jit, spelling=spelling).solve(
        params=_PARAMS, log_level="off"
    )


def _simulate(
    *, gated: bool, enable_jit: bool, spelling: Spelling, n_subjects: int = 64
):
    model = _build_model(gated=gated, enable_jit=enable_jit, spelling=spelling)
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
_GATED = pytest.mark.parametrize("gated", [False, True], ids=["plain-gate", "ref-gate"])
_SPELLING = pytest.mark.parametrize("spelling", ["value", "code"])


@_SPELLING
@_GATED
@_ENABLE_JIT
def test_value_constraint_reads_the_node_in_solve(
    *, gated: bool, enable_jit: bool, spelling: Spelling
):
    """At every node the couple works: the wife holds the oracle plus the bracket."""
    solution = _solve(gated=gated, enable_jit=enable_jit, spelling=spelling)
    V = np.asarray(solution.values.materialize()[1]["married"])
    expected = np.stack([_oracle(_NODES) + _BRACKET, np.zeros(5)], axis=-1)
    np.testing.assert_allclose(V, expected, atol=1e-4)


@_SPELLING
@_GATED
@_ENABLE_JIT
def test_value_constraint_leaves_every_node_feasible(
    *, gated: bool, enable_jit: bool, spelling: Spelling
):
    """No node of the couple dissolves."""
    solution = _solve(gated=gated, enable_jit=enable_jit, spelling=spelling)
    D = solution.replay_artifacts.project(DISSOLUTION_FLAG)[1]["married"]
    assert not np.asarray(D).any()


@_SPELLING
@_ENABLE_JIT
def test_gate_reads_the_landing_node_in_solve(*, enable_jit: bool, spelling: Spelling):
    """The young single's value opens the gate exactly at the positive nodes.

    With discount one and the single's terminal value equal to the cubic, the
    period-0 value is the cubic today plus the expectation over landing nodes of
    the couple's wife value (oracle plus bracket) where the gate opens and the
    single's value where it closes.
    """
    solution = _solve(gated=True, enable_jit=enable_jit, spelling=spelling)
    V0 = np.asarray(solution.values.materialize()[0]["young_f"])
    weights = np.asarray(_shock(spelling=spelling).get_transition_probs())[0]
    continuation = _NODE_VALUES + _BRACKET * (_NODES > 0.0)
    expected = _NODE_VALUES + weights @ continuation
    np.testing.assert_allclose(V0, expected, atol=1e-4)


@_SPELLING
@_GATED
@_ENABLE_JIT
def test_simulated_couples_read_their_realised_shock(
    *, gated: bool, enable_jit: bool, spelling: Spelling
):
    """Every simulated couple works, so each read matched its own realised shock."""
    panel = _simulate(gated=gated, enable_jit=enable_jit, spelling=spelling).query(
        "regime_name == 'married'"
    )
    assert len(panel) > 0
    assert panel["work"].tolist() == ["on"] * len(panel)


@_SPELLING
@_ENABLE_JIT
def test_gate_opens_exactly_on_positive_landings(
    *, enable_jit: bool, spelling: Spelling
):
    """Couples hold positive shocks; routed singles hold the rest of the draws."""
    panel = _simulate(gated=True, enable_jit=enable_jit, spelling=spelling)
    in_couple = panel["regime_name"].to_numpy() == "married"
    positive = panel["wage_shock_f"].to_numpy() > 0.0
    np.testing.assert_array_equal(in_couple, positive)


@_GATED
def test_continuous_draws_are_off_node_and_span_both_signs(*, gated: bool):
    """The draws are realised values, not nodes, on both sides of the gate."""
    shocks = _simulate(gated=gated, enable_jit=True, spelling="value")[
        "wage_shock_f"
    ].to_numpy()
    assert (shocks < -0.5).any()
    assert (shocks > 0.5).any()
    assert not np.isin(shocks, _NODES).any()


@_GATED
def test_node_draws_reach_every_node(*, gated: bool):
    """Node draws land on each of the five nodes, so every position is read."""
    shocks = _simulate(gated=gated, enable_jit=True, spelling="code", n_subjects=512)[
        "wage_shock_f"
    ].to_numpy()
    np.testing.assert_array_equal(np.unique(shocks), _NODES)


def test_a_code_read_of_a_process_drawn_between_nodes_is_refused():
    """A continuously drawn shock has no node position in simulation."""
    with pytest.raises(RegimeInitializationError, match="simulate_draws='nodes'"):
        _build_model(gated=True, enable_jit=False, spelling="code", draws="continuous")
