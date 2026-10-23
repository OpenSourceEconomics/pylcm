"""A folded IID shock in a gated collective regime is routed per node, then averaged.

A young single woman moves into a couple, through a gate that keeps the couple
together unless it dissolves. The couple carries an IID shock `s` and declares
it `fold=True`. Her outside option is a single woman's value, read through a
projection of the couple's own shock, both in the couple's participation
constraint and as the gate's fallback. The single woman carries `s` unfolded.

The contract under test:

- every node of `s` stays an axis through the participation constraint and the
  gate, so the dissolution flag is decided node by node;
- only after routing is each stakeholder's value averaged with the quadrature
  weights, so a household that dissolves at one node is priced as dissolving
  at that node only;
- a next-period state law may read the shock its own regime folds, singleton
  and collective alike: the next state is formed at each node, the
  continuation is read there, the max is taken, and only then the average.

The oracle below is literal scalar loops over nodes and actions written from
the primitives, with nodes and weights from numpy's probabilists' Gauss-Hermite
rule rather than from the process. It shares no code with the engine.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.regime_building.processing import process_regimes
from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    ValueDependentConstraint,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.solver_api import DISSOLUTION_FLAG
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    StateName,
)
from tests.regime_building.test_fold_gate_guard import (
    _AGES_2P,
    _make_same_period_ref_regimes,
    _solve_kwargs,
)

_N_NODES = 3


def _oracle_nodes_and_weights() -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal Gauss-Hermite nodes and probability weights."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(_N_NODES)
    return nodes, weights / weights.sum()


def _shock(*, fold: bool) -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=_N_NODES,
        gauss_hermite=True,
        mu=0.0,
        sigma=1.0,
        fold=fold,
        simulate_draws="nodes",
    )


@categorical(ordered=False)
class Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class HouseholdRegimeId:
    young_f: ScalarInt
    married: ScalarInt
    single_f: ScalarInt


@categorical(ordered=False)
class CarryRegimeId:
    shocked: ScalarInt
    after: ScalarInt


def _tolerance() -> float:
    """Quadrature agreement owed at the working precision."""
    return 1e-12 if jnp.zeros(1).dtype == jnp.float64 else 1e-5


def _v_single(s: float) -> float:
    return 0.5 * s**3


def _u_married_f(*, s: float, work: int) -> float:
    return 1.0 - 0.5 * work + 0.0 * s


def _u_married_m(*, s: float, work: int) -> float:
    return 2.0 * work + 0.0 * s


def oracle_married_nodes() -> tuple[list[tuple[float, float]], list[bool]]:
    """Per node: each stakeholder's value at the household argmax, and D.

    An action is feasible where the wife's value clears her single value at
    the same node; the household maximizes the equally weighted sum.
    """
    nodes, _ = _oracle_nodes_and_weights()
    values: list[tuple[float, float]] = []
    dissolved: list[bool] = []
    for s in nodes:
        best: tuple[float, float] | None = None
        best_objective = -np.inf
        for work in (0, 1):
            q_f = _u_married_f(s=s, work=work)
            q_m = _u_married_m(s=s, work=work)
            if q_f < _v_single(s):
                continue
            objective = 0.5 * q_f + 0.5 * q_m
            if objective > best_objective:
                best_objective = objective
                best = (q_f, q_m)
        dissolved.append(best is None)
        values.append((-np.inf, -np.inf) if best is None else best)
    return values, dissolved


def oracle_route_then_average(
    *,
    node_values: list[float],
    dissolved: list[bool],
    fallbacks: list[float],
    weights: list[float],
) -> float:
    """Route each node through the gate, then take the weighted average."""
    total = 0.0
    for value, flag, fallback, weight in zip(
        node_values, dissolved, fallbacks, weights, strict=True
    ):
        total += weight * (fallback if flag else value)
    return total


def oracle_average_then_route(
    *,
    node_values: list[float],
    dissolved: list[bool],
    fallbacks: list[float],
    weights: list[float],
) -> float:
    """The refused order: average value and flag first, then route the average."""
    averaged_value = sum(w * v for w, v in zip(weights, node_values, strict=True))
    averaged_fallback = sum(w * f for w, f in zip(weights, fallbacks, strict=True))
    return averaged_fallback if any(dissolved) else averaged_value


def oracle_young_value() -> float:
    """The young single's period-0 value: zero today plus the routed average."""
    nodes, weights = _oracle_nodes_and_weights()
    values, dissolved = oracle_married_nodes()
    return oracle_route_then_average(
        node_values=[value_f for value_f, _ in values],
        dissolved=dissolved,
        fallbacks=[_v_single(s) for s in nodes],
        weights=list(weights),
    )


def _carry_u(work: int) -> float:
    return -0.5 * work


def oracle_carry_value(*, stakeholder_weight: tuple[float, float] | None) -> np.ndarray:
    """Period-0 value when the next state is `work * s` at each node.

    The terminal payoff is linear in the carried state, `3 * kept` for the
    singleton (and the wife), `kept` for the husband.
    """
    nodes, weights = _oracle_nodes_and_weights()
    if stakeholder_weight is None:
        total = 0.0
        for s, weight in zip(nodes, weights, strict=True):
            best = -np.inf
            for work in (0, 1):
                best = max(best, _carry_u(work) + 3.0 * work * s)
            total += weight * best
        return np.asarray(total)
    totals = np.zeros(2)
    for s, weight in zip(nodes, weights, strict=True):
        best_pair = (-np.inf, -np.inf)
        best_objective = -np.inf
        for work in (0, 1):
            pair = (_carry_u(work) + 3.0 * work * s, 0.0 + 1.0 * work * s)
            objective = (
                stakeholder_weight[0] * pair[0] + stakeholder_weight[1] * pair[1]
            )
            if objective > best_objective:
                best_objective = objective
                best_pair = pair
        totals += weight * np.asarray(best_pair)
    return totals


def _single_f_utility(s: ContinuousState) -> FloatND:
    return 0.5 * s**3


def _young_f_utility(s: ContinuousState) -> FloatND:
    return 0.0 * s


def _married_f(*, s: ContinuousState, work: DiscreteAction) -> FloatND:
    return 1.0 - 0.5 * work + 0.0 * s


def _married_m(*, s: ContinuousState, work: DiscreteAction) -> FloatND:
    return 2.0 * work + 0.0 * s


def _participation_f(*, Q_f: FloatND, V_single_f: FloatND) -> BoolND:
    return Q_f >= V_single_f


def _identity(s: ContinuousState) -> ContinuousState:
    return s


def _node_at_code(s_code: DiscreteState) -> ContinuousState:
    """Name the node by its position in the shock's grid."""
    return _shock(fold=False).to_jax()[s_code]


def _no_dissolution(D_target: BoolND) -> BoolND:
    return ~D_target


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _build_household(
    *,
    fold: bool,
    enable_jit: bool,
    single_fold: bool = False,
    single_state: StateName = "s",
    participation: bool = True,
    spelling: str = "value",
) -> Model:
    single_projection = {
        single_state: _identity if spelling == "value" else _node_at_code
    }
    married = Regime(
        states={"s": _shock(fold=fold)},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _married_f, "m": _married_m})
        },
        constraints=(
            {
                "participation_f": ValueDependentConstraint(
                    predicate=_participation_f,
                    references={
                        "V_single_f": ProjectedRegimeValue(
                            regime="single_f", projection=single_projection
                        )
                    },
                )
            }
            if participation
            else {}
        ),
    )
    single_f = Regime(
        states={single_state: _shock(fold=single_fold)},
        functions={
            "utility": (
                _single_f_utility if single_state == "s" else lambda t: 0.5 * t**3
            )
        },
    )
    young_f = Regime(
        states={"s": _shock(fold=False)}, functions={"utility": _young_f_utility}
    )
    return Model(
        edges={
            "young_f": Transition(
                targets={"married": 0, "single_f": 0},
                law=ByAge(cases={0: {"married": StochasticTransition(func=_prob_one)}}),
                gates={
                    "married": Gate(
                        predicate=_no_dissolution,
                        routes={
                            "only": StakeholderRoute(
                                target_stakeholder="f",
                                fallback=ProjectedRegimeValue(
                                    regime="single_f", projection=single_projection
                                ),
                            )
                        },
                    )
                },
            )
        },
        regimes={"young_f": young_f, "married": married, "single_f": single_f},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=HouseholdRegimeId,
        initial_nodes={0: "young_f"},
        enable_jit=enable_jit,
    )


_HOUSEHOLD_PARAMS = {
    "young_f": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "married": {},
    "single_f": {},
}


def _solve_household(*, fold: bool, enable_jit: bool):
    model = _build_household(fold=fold, enable_jit=enable_jit)
    return model, model.solve(params=_HOUSEHOLD_PARAMS, log_level="off")


def _simulate_household(*, fold: bool, enable_jit: bool, n_subjects: int = 96):
    model, solution = _solve_household(fold=fold, enable_jit=enable_jit)
    result = model.simulate(
        params=_HOUSEHOLD_PARAMS,
        initial_conditions={
            "s": jnp.zeros(n_subjects),
            "age": jnp.zeros(n_subjects),
            "regime_id": jnp.full(
                n_subjects, model.regime_names_to_ids["young_f"], jnp.int32
            ),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )
    return result.to_dataframe().sort_values(["period", "subject_id"])


_ENABLE_JIT = pytest.mark.parametrize("enable_jit", [True, False], ids=["jit", "eager"])
_FOLD = pytest.mark.parametrize("fold", [True, False], ids=["folded", "unfolded"])


def test_oracle_nodes_are_the_process_nodes():
    """The oracle integrates over the nodes and weights the process declares."""
    nodes, weights = _oracle_nodes_and_weights()
    process = _shock(fold=False)
    np.testing.assert_allclose(np.asarray(process.to_jax()), nodes, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(process.get_transition_probs())[0], weights, atol=1e-6
    )


def test_oracle_routes_the_documented_counterexample_to_one_half():
    """Values [-inf, 1], flags [True, False], fallback 0, equal weights give 0.5."""
    routed = oracle_route_then_average(
        node_values=[-np.inf, 1.0],
        dissolved=[True, False],
        fallbacks=[0.0, 0.0],
        weights=[0.5, 0.5],
    )
    assert routed == 0.5


def test_averaging_first_gets_the_documented_counterexample_wrong():
    """Averaging before routing collapses the counterexample to the fallback."""
    averaged = oracle_average_then_route(
        node_values=[-np.inf, 1.0],
        dissolved=[True, False],
        fallbacks=[0.0, 0.0],
        weights=[0.5, 0.5],
    )
    assert averaged == 0.0


def test_household_oracle_dissolves_at_exactly_one_node():
    """The fixture is informative: one node dissolves, the others do not."""
    _, dissolved = oracle_married_nodes()
    assert dissolved == [False, False, True]


def test_household_oracle_separates_the_two_orders():
    """Route-then-average and average-then-route differ on this household."""
    nodes, weights = _oracle_nodes_and_weights()
    values, dissolved = oracle_married_nodes()
    node_values = [value_f for value_f, _ in values]
    fallbacks = [_v_single(s) for s in nodes]
    routed = oracle_route_then_average(
        node_values=node_values,
        dissolved=dissolved,
        fallbacks=fallbacks,
        weights=list(weights),
    )
    averaged = oracle_average_then_route(
        node_values=node_values,
        dissolved=dissolved,
        fallbacks=fallbacks,
        weights=list(weights),
    )
    assert abs(routed - averaged) > 0.1


@_FOLD
@_ENABLE_JIT
def test_married_values_are_decided_per_node(*, fold: bool, enable_jit: bool):
    """Each node carries its own stakeholder values, `-inf` where it dissolves."""
    _, solution = _solve_household(fold=fold, enable_jit=enable_jit)
    values, _ = oracle_married_nodes()
    V = np.asarray(solution.values[1]["married"])
    np.testing.assert_array_equal(V, np.asarray(values, dtype=V.dtype))


@_FOLD
@_ENABLE_JIT
def test_married_dissolution_flag_is_decided_per_node(*, fold: bool, enable_jit: bool):
    """The dissolution flag is set at the dissolving node only, never reduced."""
    _, solution = _solve_household(fold=fold, enable_jit=enable_jit)
    _, dissolved = oracle_married_nodes()
    D = np.asarray(solution.replay_artifacts.project(DISSOLUTION_FLAG)[1]["married"])
    np.testing.assert_array_equal(D, np.asarray(dissolved))


@_FOLD
@_ENABLE_JIT
def test_young_value_routes_each_node_before_averaging(*, fold: bool, enable_jit: bool):
    """The young single's value is the oracle's route-then-average at every node."""
    _, solution = _solve_household(fold=fold, enable_jit=enable_jit)
    V0 = np.asarray(solution.values[0]["young_f"])
    np.testing.assert_allclose(
        V0, np.full(_N_NODES, oracle_young_value()), rtol=0.0, atol=_tolerance()
    )


@_ENABLE_JIT
def test_folded_solution_equals_unfolded_solution(*, enable_jit: bool):
    """Folding the couple's shock changes no stored value or flag."""
    _, folded = _solve_household(fold=True, enable_jit=enable_jit)
    _, unfolded = _solve_household(fold=False, enable_jit=enable_jit)
    folded_values = folded.values
    unfolded_values = unfolded.values
    for period, regime in ((0, "young_f"), (1, "married"), (1, "single_f")):
        np.testing.assert_array_equal(
            np.asarray(folded_values[period][regime]),
            np.asarray(unfolded_values[period][regime]),
        )


@_ENABLE_JIT
def test_folded_simulation_equals_unfolded_simulation(*, enable_jit: bool):
    """The same seed simulates the same regimes, shocks and choices either way."""
    columns = ["period", "subject_id", "regime_name", "s", "work"]
    folded = _simulate_household(fold=True, enable_jit=enable_jit)
    unfolded = _simulate_household(fold=False, enable_jit=enable_jit)
    pd.testing.assert_frame_equal(
        folded[columns].reset_index(drop=True),
        unfolded[columns].reset_index(drop=True),
        check_exact=True,
    )


def test_simulated_households_dissolve_exactly_at_the_dissolving_node():
    """A row stays married where its realised node keeps the couple together."""
    panel = _simulate_household(fold=True, enable_jit=True).query("period == 1")
    nodes, _ = _oracle_nodes_and_weights()
    in_couple = panel["regime_name"].to_numpy() == "married"
    dissolving = np.isclose(panel["s"].to_numpy(), nodes[-1], atol=1e-5)
    assert dissolving.any()
    np.testing.assert_array_equal(in_couple, ~dissolving)


_NOT_CARRIED = r"\['s'\].*onto regime 'single_f', which does not carry"
_READERS = pytest.mark.parametrize(
    "participation", [True, False], ids=["constraint-and-gate", "gate-only"]
)
_SPELLINGS = pytest.mark.parametrize("spelling", ["value", "code"])


@_SPELLINGS
@_READERS
def test_projection_onto_a_target_that_folds_the_shock_is_refused(
    *, participation: bool, spelling: str
):
    """A folded single cannot be read at the couple's node."""
    with pytest.raises(ModelInitializationError, match=_NOT_CARRIED):
        _build_household(
            fold=True,
            enable_jit=False,
            single_fold=True,
            participation=participation,
            spelling=spelling,
        )


@_SPELLINGS
@_READERS
def test_projection_onto_a_target_without_the_shock_is_refused(
    *, participation: bool, spelling: str
):
    """A projection reading a folded shock needs the shock as a target axis."""
    with pytest.raises(ModelInitializationError, match=_NOT_CARRIED):
        _build_household(
            fold=True,
            enable_jit=False,
            single_state="t",
            participation=participation,
            spelling=spelling,
        )


def test_folded_collective_regime_read_as_a_reference_is_refused():
    """A couple that folds its shock cannot be read node by node by another regime."""
    with pytest.raises(ModelInitializationError, match=r"(?s)ref_target.*REFERENCE"):
        process_regimes(
            **_solve_kwargs(
                regimes_and_laws=_make_same_period_ref_regimes(fold=True),
                ages=_AGES_2P,
            )
        )


def _carry_utility(work: DiscreteAction) -> FloatND:
    return -0.5 * work


def _carry_utility_m(work: DiscreteAction) -> FloatND:
    return 0.0 * work


def _keep(*, s: ContinuousState, work: DiscreteAction) -> ContinuousState:
    return work * s


def _after_utility(kept: ContinuousState) -> FloatND:
    return 3.0 * kept


def _after_utility_f(*, kept: ContinuousState, work: DiscreteAction) -> FloatND:
    return 3.0 * kept + 0.0 * work


def _after_utility_m(*, kept: ContinuousState, work: DiscreteAction) -> FloatND:
    return 1.0 * kept + 0.0 * work


def _build_carry(*, fold: bool, collective: bool) -> Model:
    if collective:
        shocked_utility = CollectiveUtility(
            utilities={"f": _carry_utility, "m": _carry_utility_m}
        )
        after_utility = CollectiveUtility(
            utilities={"f": _after_utility_f, "m": _after_utility_m}
        )
    else:
        shocked_utility = _carry_utility
        after_utility = _after_utility
    shocked = Regime(
        states={"s": _shock(fold=fold)},
        state_transitions={"kept": {"after": _keep}},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": shocked_utility},
    )
    after = Regime(
        states={"kept": LinSpacedGrid(start=-2.0, stop=2.0, n_points=5)},
        functions={"utility": after_utility},
        **(
            {"actions": {"work": DiscreteGrid(category_class=Work)}}
            if collective
            else {}
        ),
    )
    return Model(
        regimes={"shocked": shocked, "after": after},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=CarryRegimeId,
        initial_nodes={0: "shocked"},
        edges={"shocked": {"after": 0}},
    )


_CARRY_PARAMS = {
    "shocked": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "after": {},
}

_COLLECTIVE = pytest.mark.parametrize(
    "collective", [False, True], ids=["singleton", "collective"]
)


def _period_0_value(*, fold: bool, collective: bool) -> np.ndarray:
    solution = _build_carry(fold=fold, collective=collective).solve(
        params=_CARRY_PARAMS, log_level="off"
    )
    return np.asarray(solution.values[0]["shocked"])


@_COLLECTIVE
def test_transition_reading_the_folded_shock_solves_per_node(*, collective: bool):
    """The value is the average over nodes of the per-node max, next state at node."""
    V0 = _period_0_value(fold=True, collective=collective)
    expected = oracle_carry_value(stakeholder_weight=(0.5, 0.5) if collective else None)
    if collective:
        # A collective regime keeps the node axis; the reader averages it.
        _, weights = _oracle_nodes_and_weights()
        V0 = np.tensordot(weights, V0, axes=(0, 0))
    np.testing.assert_allclose(V0, expected, rtol=0.0, atol=_tolerance())


@_COLLECTIVE
def test_transition_reading_the_folded_shock_equals_unfolded_average(
    *, collective: bool
):
    """Folding equals solving unfolded and averaging the node axis by hand."""
    _, weights = _oracle_nodes_and_weights()
    unfolded = _period_0_value(fold=False, collective=collective)
    folded = _period_0_value(fold=True, collective=collective)
    if collective:
        folded = np.tensordot(weights, folded, axes=(0, 0))
    np.testing.assert_allclose(
        folded,
        np.tensordot(weights, unfolded, axes=(0, 0)),
        rtol=0.0,
        atol=_tolerance(),
    )


@_COLLECTIVE
def test_simulated_next_state_reads_the_realised_node(*, collective: bool):
    """Each row's carried state is its own choice times its own realised node."""
    model = _build_carry(fold=True, collective=collective)
    n_subjects = 96
    result = model.simulate(
        params=_CARRY_PARAMS,
        initial_conditions={
            "s": jnp.asarray(_oracle_nodes_and_weights()[0])[
                jnp.arange(n_subjects) % _N_NODES
            ],
            "age": jnp.zeros(n_subjects),
            "regime_id": jnp.full(
                n_subjects, model.regime_names_to_ids["shocked"], jnp.int32
            ),
        },
        log_level="off",
        seed=0,
    )
    panel = result.to_dataframe().sort_values(["subject_id", "period"])
    first = panel.query("period == 0")
    second = panel.query("period == 1")
    worked = (first["work"].astype(str) == "on").to_numpy()
    shocks = first["s"].to_numpy()
    assert len(np.unique(shocks)) == _N_NODES
    np.testing.assert_allclose(
        second["kept"].to_numpy(), worked * shocks, rtol=0.0, atol=1e-6
    )
