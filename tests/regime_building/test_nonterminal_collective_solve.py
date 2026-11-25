"""Integration tests for the NON-terminal collective-regime solve.

Extends the terminal machinery with per-member continuation values: a
non-terminal `Regime(members=("f", "m"))` computes
`Q^s = u^s + beta * E[V'^s]` per member, argmaxes the household
scalarization `O = sum_s lambda_s Q^s` over the feasible actions, and reads off
each member's OWN `Q^s` at that shared argmax. No gated edges here; those
are covered by `test_gated_edges_collective_solve.py`.

The economics mirror the work/leisure setup of
`test_terminal_collective_solve.py` (same utility functions) so the hand
computations carry over; only the wage grid's low point moves from 10 to 8 so
every argmax is strict (at wage 10 the myopic household objective ties).

Hand computation, wage grid {8, 40}, beta = 0.95, next_wage = 40*work +
8*(1-work) (working today yields the high wage tomorrow):

Terminal period (t=1), myopic household argmax of O = (u_f + u_m)/2:
  wage=8 : leisure (30, 0) O=15  vs work (8, 16)  O=12  -> leisure, V=(30, 0)
  wage=40: leisure (30, 0) O=15  vs work (40, 80) O=60  -> work,    V=(40, 80)

Period 0, Q^s = u^s + 0.95 * V'^s(next_wage):
  wage=8 : leisure Q=(30+28.5, 0)      =(58.5, 0)  O=29.25
           work    Q=(8+0.95*40, 16+0.95*80)=(46, 92)  O=69  -> work, V=(46, 92)
  wage=40: leisure Q=(58.5, 0)  O=29.25
           work    Q=(40+38, 80+76)=(78, 156)      O=117    -> work, V=(78, 156)

At wage=8 the period-0 household argmax (work) DIFFERS from the myopic
terminal-period one (leisure): only the continuation through tomorrow's high
wage makes work jointly optimal.
"""

from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.regime_building.transition_support import (
    _SupportedDeterministicTransition,
)
from _lcm.regime_law import RegimeLaws
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import (
    AgeRange,
    ByAge,
    CESAggregator,
    CollectiveUtility,
    Decoupled,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    ParetoObjective,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.certainty_equivalent import PowerMean
from lcm.koopmans_aggregation import LinearAggregator
from lcm.regime import Regime
from lcm.transition import StochasticTransition
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    Period,
    ScalarInt,
    UserFunction,
    UserParamsNode,
)
from lcm_examples import epstein_zin
from tests.conftest import assert_agrees_to_ulp, bind_laws, build_prepared_structure


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class RegimeId:
    couple: ScalarInt  # code 0
    couple_terminal: ScalarInt  # code 1


def _utility_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife: values her own leisure highly, also sees household consumption."""
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _utility_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband: values household consumption, indifferent to leisure."""
    consumption = wage * work
    return 2.0 * consumption


def _next_wage(work: DiscreteAction) -> ContinuousState:
    """Deterministic wage law: working today yields the high wage tomorrow."""
    return 40.0 * work + 8.0 * (1.0 - work)


def _next_regime() -> ScalarInt:
    return RegimeId.couple_terminal


_GRAPH_BOUND_LAW = _SupportedDeterministicTransition(
    func=_next_regime, targets=("couple_terminal",)
)


_WAGE_GRID = LinSpacedGrid(start=8.0, stop=40.0, n_points=2)

_EXPECTED_V_TERMINAL = np.array([[30.0, 0.0], [40.0, 80.0]])
_EXPECTED_V_PERIOD_0 = np.array([[46.0, 92.0], [78.0, 156.0]])


def _make_couple_regimes() -> dict[str, Regime]:
    """Build the couple regimes; `couple` leaves for `couple_terminal` at age 0.

    The regimes carry no law: a model declares the single edge in
    `Model(edges=...)`, and kernel-level tests bind the graph-bound law through
    `_couple_laws`.
    """
    couple = Regime(
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
    )
    couple_terminal = Regime(
        states={"wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
    )
    return {"couple": couple, "couple_terminal": couple_terminal}


def _couple_laws() -> RegimeLaws:
    """The couple regimes' laws, bound as a model binds them."""
    return bind_laws(
        {
            "couple": ByAge(cases={AgeRange(exclusive_stop=1): _GRAPH_BOUND_LAW}),
            "couple_terminal": None,
        }
    )


def test_nonterminal_collective_regime_solves_with_continuation() -> None:
    """Kernel-level: finalize -> process -> backward induction, two periods."""
    laws = _couple_laws()
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes = process_regimes(
        prepared_structure=build_prepared_structure(
            user_regimes=finalize_regimes(
                user_regimes=_make_couple_regimes(),
                derived_categoricals={},
                koopmans_aggregator=LinearAggregator(),
                certainty_equivalent=LinearExpectation(),
                laws=laws,
            ),
            ages=ages,
            laws=laws,
        ),
        user_regimes=finalize_regimes(
            user_regimes=_make_couple_regimes(),
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
            laws=laws,
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {"couple": jnp.int32(0), "couple_terminal": jnp.int32(1)}
        ),
        enable_jit=False,
    )

    _bi_result = solve(
        program_fingerprint="test_nonterminal_collective_solve",
        flat_params=MappingProxyType(
            {
                "couple": MappingProxyType(
                    {"koopmans_aggregator__discount_factor": jnp.asarray(0.95)}
                ),
                "couple_terminal": MappingProxyType({}),
            }
        ),
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies

    V_terminal = solution[1]["couple_terminal"]
    V_0 = solution[0]["couple"]

    # Both V arrays carry the trailing member axis: (wage grid = 2) x (S = 2).
    assert V_terminal.shape == (2, 2)
    assert V_0.shape == (2, 2)

    np.testing.assert_allclose(np.asarray(V_terminal), _EXPECTED_V_TERMINAL, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(V_0), _EXPECTED_V_PERIOD_0, rtol=1e-6)

    # Per-member values genuinely differ (not the scalarization).
    assert not np.allclose(np.asarray(V_0[:, 0]), np.asarray(V_0[:, 1]))

    # The continuation flips the argmax at the low wage: the myopic (terminal)
    # household choice there is leisure, whose period-0 member values would
    # be (30 + 0.95*30, 0 + 0.95*0) = (58.5, 0). The solved V is the work
    # branch's (46, 92) instead.
    assert not np.allclose(np.asarray(V_0[0]), [58.5, 0.0])


def test_nonterminal_collective_full_model_solve_matches_kernel_level() -> None:
    """Model-level: the same two regimes through public Model(...) + solve()."""
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    model = Model(
        regimes=_make_couple_regimes(),
        edges={"couple": {"couple_terminal": 0}},
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "couple"},
    )

    solution = model.solve(params={"discount_factor": 0.95}, log_level="off").values

    V_terminal = solution[1]["couple_terminal"]
    V_0 = solution[0]["couple"]
    assert V_terminal.shape == (2, 2)
    assert V_0.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(V_terminal), _EXPECTED_V_TERMINAL, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(V_0), _EXPECTED_V_PERIOD_0, rtol=1e-6)


@categorical(ordered=True)
class Mood:
    grumpy: ScalarInt  # code 0
    cheerful: ScalarInt  # code 1


def _utility_f_mood(
    *, wage: ContinuousState, work: DiscreteAction, mood: DiscreteState
) -> FloatND:
    """As `_utility_f`, plus a mood bonus only the wife's felicity carries."""
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work) + 5.0 * mood


def _next_mood(mood: DiscreteState) -> FloatND:  # noqa: ARG001
    """IID coin-flip mood: probabilities (0.5, 0.5) regardless of today's mood."""
    return jnp.array([0.5, 0.5])


def test_nonterminal_collective_stochastic_state_expectation_is_per_member() -> None:
    """A Markov state's expectation is taken per member slice.

    Adds an IID binary `mood` state (probabilities 1/2 each) whose realization
    enters ONLY the wife's felicity (`+ 5 * mood`). Hand computation on top of
    the module-level one:

    Terminal V (mood shifts u_f additively, so argmax is unchanged):
      V(m, wage=8)  = (30 + 5m, 0)      V(m, wage=40) = (40 + 5m, 80)

    Period 0: E over mood' hits the member slices differently —
    E[V'_f](wage') = base_f(wage') + 2.5 while E[V'_m](wage') is unchanged:
      (m, wage=8) : leisure Q=(30+5m+0.95*32.5, 0)=(60.875+5m, 0)   O=30.4375+2.5m
                    work    Q=(8+5m+0.95*42.5, 16+76)=(48.375+5m, 92) O=70.1875+2.5m
                    -> work, V=(48.375+5m, 92)
      (m, wage=40): work analogously -> V=(80.375+5m, 156)
    """
    couple = Regime(
        states={"mood": DiscreteGrid(category_class=Mood), "wage": _WAGE_GRID},
        state_transitions={
            "mood": StochasticTransition(func=_next_mood),
            "wage": _next_wage,
        },
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _utility_f_mood, "m": _utility_m}
            )
        },
    )
    laws = bind_laws(
        {
            "couple": ByAge(
                cases={
                    AgeRange(exclusive_stop=1): _SupportedDeterministicTransition(
                        func=_next_regime, targets=("couple_terminal",)
                    )
                }
            ),
            "couple_terminal": None,
        }
    )
    couple_terminal = Regime(
        states={"mood": DiscreteGrid(category_class=Mood), "wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _utility_f_mood, "m": _utility_m}
            )
        },
    )
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes = process_regimes(
        prepared_structure=build_prepared_structure(
            user_regimes=finalize_regimes(
                user_regimes={"couple": couple, "couple_terminal": couple_terminal},
                derived_categoricals={},
                koopmans_aggregator=LinearAggregator(),
                certainty_equivalent=LinearExpectation(),
                laws=laws,
            ),
            ages=ages,
            laws=laws,
        ),
        user_regimes=finalize_regimes(
            user_regimes={"couple": couple, "couple_terminal": couple_terminal},
            derived_categoricals={},
            koopmans_aggregator=LinearAggregator(),
            certainty_equivalent=LinearExpectation(),
            laws=laws,
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {"couple": jnp.int32(0), "couple_terminal": jnp.int32(1)}
        ),
        enable_jit=False,
    )

    _bi_result = solve(
        program_fingerprint="test_nonterminal_collective_solve",
        flat_params=MappingProxyType(
            {
                "couple": MappingProxyType(
                    {"koopmans_aggregator__discount_factor": jnp.asarray(0.95)}
                ),
                "couple_terminal": MappingProxyType({}),
            }
        ),
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies

    V_terminal = solution[1]["couple_terminal"]
    V_0 = solution[0]["couple"]

    # (mood = 2) x (wage grid = 2) x (members = 2).
    assert V_terminal.shape == (2, 2, 2)
    assert V_0.shape == (2, 2, 2)

    expected_terminal = np.array(
        [
            [[30.0, 0.0], [40.0, 80.0]],  # mood = 0
            [[35.0, 0.0], [45.0, 80.0]],  # mood = 1
        ]
    )
    expected_period_0 = np.array(
        [
            [[48.375, 92.0], [80.375, 156.0]],  # mood = 0
            [[53.375, 92.0], [85.375, 156.0]],  # mood = 1
        ]
    )
    np.testing.assert_allclose(np.asarray(V_terminal), expected_terminal, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(V_0), expected_period_0, rtol=1e-6)


def test_collective_model_simulates_end_to_end_via_public_model_api() -> None:
    """A collective regime simulates end to end through the public model API.

    The full recomputed-argmax mechanics (both members tracked, values
    matching the hand computation above) are covered in depth by
    `tests/regime_building/test_collective_regime_simulate.py`; this pin just
    confirms the PUBLIC `Model.simulate()` path — auto-solving, then
    simulating — runs end to end instead of raising, for a collective regime
    with no gated edges (so `period_to_regime_to_V_arr`/dissolution flags need no
    special threading).
    """
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    model = Model(
        regimes=_make_couple_regimes(),
        edges={"couple": {"couple_terminal": 0}},
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "couple"},
    )
    result = model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions={
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
        },
        log_level="off",
        seed=0,
    )
    period_0 = result.raw_results["couple"][0]
    np.testing.assert_allclose(
        np.asarray(period_0.V_arr), _EXPECTED_V_PERIOD_0, rtol=1e-6
    )


def test_nonterminal_collective_regime_with_singleton_target_is_rejected() -> None:
    """Routing a collective regime toward a singleton target needs a gated edge."""

    def _utility_single(wage: ContinuousState) -> FloatND:
        return wage

    couple = Regime(
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m})
        },
    )
    laws = bind_laws(
        {
            "couple": ByAge(
                cases={
                    AgeRange(exclusive_stop=1): _SupportedDeterministicTransition(
                        func=_next_regime, targets=("single_terminal",)
                    )
                }
            ),
            "single_terminal": None,
        }
    )
    single_terminal = Regime(
        states={"wage": _WAGE_GRID},
        functions={"utility": _utility_single},
    )
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")

    with pytest.raises(NotImplementedError, match="identical `members`"):
        process_regimes(
            prepared_structure=build_prepared_structure(
                user_regimes=finalize_regimes(
                    user_regimes={"couple": couple, "single_terminal": single_terminal},
                    derived_categoricals={},
                    koopmans_aggregator=LinearAggregator(),
                    certainty_equivalent=LinearExpectation(),
                    laws=laws,
                ),
                ages=ages,
                laws=laws,
            ),
            user_regimes=finalize_regimes(
                user_regimes={"couple": couple, "single_terminal": single_terminal},
                derived_categoricals={},
                koopmans_aggregator=LinearAggregator(),
                certainty_equivalent=LinearExpectation(),
                laws=laws,
            ),
            ages=ages,
            regime_names_to_ids=MappingProxyType(
                {"couple": jnp.int32(0), "single_terminal": jnp.int32(1)}
            ),
            enable_jit=False,
        )


def _epstein_zin(
    *,
    alive_utility: UserFunction | CollectiveUtility,
    dead_utility: UserFunction | CollectiveUtility,
    koopmans_aggregator: CESAggregator
    | LinearAggregator
    | dict[str, CESAggregator | LinearAggregator],
    certainty_equivalent: PowerMean
    | LinearExpectation
    | dict[str, PowerMean | LinearExpectation],
) -> Model:
    """The Epstein-Zin example over four periods, with the given preferences."""
    alive = Regime(
        states={
            "wealth": epstein_zin.WEALTH_GRID,
            "health": DiscreteGrid(category_class=epstein_zin.HealthStatus),
        },
        state_transitions={
            "wealth": epstein_zin.next_wealth,
            "health": {"alive": StochasticTransition(func=epstein_zin.health_probs)},
        },
        actions={"consumption": epstein_zin.CONSUMPTION_GRID},
        constraints={"budget_constraint": epstein_zin.budget_constraint},
        functions={"utility": alive_utility},
        koopmans_aggregator=koopmans_aggregator,
        certainty_equivalent=certainty_equivalent,
    )
    dead = Regime(
        states={"wealth": epstein_zin.DEAD_WEALTH_GRID},
        functions={"utility": dead_utility},
    )
    return Model(
        edges={
            "alive": Transition(
                targets={"dead": (25, 26, 27), "alive": (25, 26)},
                law=StochasticTransition(func=epstein_zin.next_regime),
            )
        },
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=25, inclusive_stop=28, step="Y"),
        regime_id_class=epstein_zin.EZRegimeId,
        initial_nodes={25: "alive"},
    )


@pytest.mark.parametrize(
    ("member", "weights", "koopmans_aggregator", "certainty_equivalent"),
    [
        ("f", {"f": 1.0, "m": 0.0}, CESAggregator(), PowerMean()),
        ("m", {"f": 0.0, "m": 1.0}, LinearAggregator(), LinearExpectation()),
    ],
)
def test_each_member_aggregates_with_its_own_w_and_certainty_equivalent(
    *,
    member: str,
    weights: dict[str, float],
    koopmans_aggregator: CESAggregator | LinearAggregator,
    certainty_equivalent: PowerMean | LinearExpectation,
) -> None:
    """The deciding member's values are the single's under that member's W and CE.

    `f` aggregates with `CESAggregator` and `PowerMean`, `m` with the linear
    aggregator and expectation; the Pareto weights make one of them decide.
    Per-member parameters live under `koopmans_aggregator_<m>` and
    `certainty_equivalent_<m>`.
    """
    params = epstein_zin.get_params(risk_aversion=4.0)
    alive = cast("dict[str, UserParamsNode]", params["alive"])
    linear = {"discount_factor": 0.9}
    single = _epstein_zin(
        alive_utility=epstein_zin.utility_alive,
        dead_utility=epstein_zin.utility_dead,
        koopmans_aggregator=koopmans_aggregator,
        certainty_equivalent=certainty_equivalent,
    ).solve(
        params={
            **params,
            "alive": {
                "next_wealth": alive["next_wealth"],
                **(
                    {
                        "koopmans_aggregator": alive["koopmans_aggregator"],
                        "certainty_equivalent": alive["certainty_equivalent"],
                    }
                    if member == "f"
                    else {"koopmans_aggregator": linear}
                ),
            },
        },
        log_level="off",
    )
    household = _epstein_zin(
        alive_utility=CollectiveUtility(
            utilities={"f": epstein_zin.utility_alive, "m": epstein_zin.utility_alive},
            objective=ParetoObjective(weights=weights),
        ),
        dead_utility=CollectiveUtility(
            utilities={"f": epstein_zin.utility_dead, "m": epstein_zin.utility_dead}
        ),
        koopmans_aggregator={"f": CESAggregator(), "m": LinearAggregator()},
        certainty_equivalent={"f": PowerMean(), "m": LinearExpectation()},
    ).solve(
        params={
            **params,
            "alive": {
                "next_wealth": alive["next_wealth"],
                "koopmans_aggregator_f": alive["koopmans_aggregator"],
                "koopmans_aggregator_m": linear,
                "certainty_equivalent_f": alive["certainty_equivalent"],
            },
            "dead": {f"utility_{name}": {"bequest_scale": 1.0} for name in ("f", "m")},
        },
        log_level="off",
    )
    index = ("f", "m").index(member)

    for period, values in single.values.items():
        for regime, expected in values.items():
            assert_agrees_to_ulp(
                got=np.asarray(household.values[period][regime])[..., index],
                expected=np.asarray(expected)[..., 0],
                n_ulp=16,
                operand_magnitude=float(np.max(np.abs(expected))),
                err_msg=f"period {period}, regime {regime!r}",
            )


def _flow_two(work: DiscreteAction) -> FloatND:
    return 2.0 - work


def _flow_four(work: DiscreteAction) -> FloatND:
    return 4.0 - work


def _still_two() -> FloatND:
    return jnp.asarray(2.0)


def _still_four() -> FloatND:
    return jnp.asarray(4.0)


def _terminal_three() -> FloatND:
    return jnp.asarray(3.0)


def _terminal_six() -> FloatND:
    return jnp.asarray(6.0)


def _summed_W(*, utility: FloatND, CE: FloatND, beta: float) -> FloatND:
    """The additive aggregator, written as a sum over its two terms."""
    return jnp.sum(jnp.stack((utility, beta * CE)))


def _euclidean_W(*, utility: FloatND, CE: FloatND, beta: float) -> FloatND:
    """A nonlinear aggregator, written as a norm over its two terms."""
    return jnp.sqrt(jnp.sum(jnp.stack((utility**2, (beta * CE) ** 2))))


# Per member: flow with and without the action, and terminal value.
_SCALAR_W_FUNCTIONS: dict[str, tuple[UserFunction, UserFunction, UserFunction]] = {
    "a": (_flow_two, _still_two, _terminal_three),
    "b": (_flow_four, _still_four, _terminal_six),
}
# Per member: the period-0 value under `_summed_W` and `_euclidean_W` at beta = 0.5,
# where leisure is chosen.
_SCALAR_W_VALUES = {"a": (3.5, 2.5), "b": (7.0, 5.0)}


def _scalar_W_model(
    *,
    member_names: tuple[str, ...],
    koopmans_aggregator: UserFunction | Decoupled,
    with_action: bool,
    with_ce: bool,
) -> Model:
    flow_index = 0 if with_action else 1

    def declare(index: int) -> UserFunction | CollectiveUtility:
        if member_names == ("self",):
            return _SCALAR_W_FUNCTIONS["a"][index]
        return CollectiveUtility(
            utilities={
                member: _SCALAR_W_FUNCTIONS[member][index] for member in member_names
            }
        )

    return Model(
        regimes={
            "couple": Regime(
                actions={"work": DiscreteGrid(Work)} if with_action else {},
                functions={"utility": declare(flow_index)},
                koopmans_aggregator=koopmans_aggregator,
                certainty_equivalent=LinearExpectation() if with_ce else None,
            ),
            "couple_terminal": Regime(functions={"utility": declare(2)}),
        },
        edges={"couple": {"couple_terminal": 0}},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "couple"},
    )


@pytest.mark.parametrize("with_ce", [False, True], ids=["default_ce", "declared_ce"])
@pytest.mark.parametrize("with_action", [False, True], ids=["no_action", "action"])
@pytest.mark.parametrize("decoupled", [False, True], ids=["plain", "decoupled"])
@pytest.mark.parametrize(
    "aggregator", [_summed_W, _euclidean_W], ids=["linear", "nonlinear"]
)
@pytest.mark.parametrize(
    "member_names",
    [("self",), ("a",), ("a", "b"), ("b", "a")],
    ids=["single", "one_named", "pair", "reversed_pair"],
)
def test_a_scalar_aggregator_is_applied_to_each_members_own_values(
    *,
    member_names: tuple[str, ...],
    aggregator: UserFunction,
    decoupled: bool,
    with_action: bool,
    with_ce: bool,
) -> None:
    """A bare `W` sees one member's utility and continuation at a time.

    `W` is written as a reduction over its two terms, so applying it to the
    stacked member values instead would mix the members or drop their axis. A
    `Decoupled` declaration chooses with `aggregator` and stores the additive value.
    """
    solution = _scalar_W_model(
        member_names=member_names,
        koopmans_aggregator=(
            Decoupled(objective=aggregator, value=_summed_W)
            if decoupled
            else aggregator
        ),
        with_action=with_action,
        with_ce=with_ce,
    ).solve(params={"couple": {"koopmans_aggregator": {"beta": 0.5}}}, log_level="off")
    column = 0 if decoupled or aggregator is _summed_W else 1
    expected = [
        _SCALAR_W_VALUES["a" if name == "self" else name][column]
        for name in member_names
    ]
    np.testing.assert_array_equal(
        np.asarray(solution.values[0]["couple"]), np.array(expected)
    )


def _invest(*, investment: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return -investment + 0.0 * wealth


def _keep_wealth(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _grow(*, investment: ContinuousAction) -> ContinuousState:
    return 2.0 * investment


def _mood_W(
    *, utility: FloatND, CE: FloatND, mood: DiscreteState, beta: FloatND
) -> FloatND:
    return utility + beta[mood] * CE


def _age_W(*, utility: FloatND, CE: FloatND, period: Period, beta: FloatND) -> FloatND:
    return utility + beta[period] * CE


_INVESTMENT_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
_INVESTED_WEALTH_GRID = LinSpacedGrid(start=0.0, stop=2.0, n_points=2)
# Rows in the reverse of the label codes: label alignment, not row position,
# decides which beta each mood and age reads.
_BETA_BY_LABEL: dict[UserFunction, pd.Series] = {
    _mood_W: pd.Series(
        [0.75, 0.25], index=pd.Index(["cheerful", "grumpy"], name="mood")
    ),
    _age_W: pd.Series([0.75, 0.25], index=pd.Index([1, 0], name="age")),
}
_BETA_BY_CODE: dict[UserFunction, tuple[float, float]] = {
    _mood_W: (0.25, 0.75),
    _age_W: (0.25, 0.75),
}


@pytest.mark.parametrize(
    ("aggregators", "weights"),
    [
        ({"a": _mood_W}, None),
        ({"a": _mood_W, "b": _mood_W}, None),
        ({"b": _mood_W, "a": _mood_W}, None),
        ({"a": _mood_W, "b": _age_W}, {"a": 0.75, "b": 0.25}),
    ],
    ids=["one_named", "pair", "reversed_pair", "distinct_consumers"],
)
def test_a_labelled_parameter_of_a_member_aggregator_is_read_by_label(
    *,
    aggregators: dict[str, UserFunction],
    weights: dict[str, float] | None,
) -> None:
    """Each member's `W` binds a `pd.Series` parameter's rows by mood or age label.

    With `Q(a) = -a + beta * 2a` the household invests iff its weighted gain
    `sum_s w_s (2 beta_s - 1)` is positive; each member then values `2 beta_s - 1`.
    """
    members = tuple(aggregators)
    household = CollectiveUtility(
        utilities=dict.fromkeys(members, _invest),
        objective=None if weights is None else ParetoObjective(weights=weights),
    )
    model = Model(
        regimes={
            "couple": Regime(
                states={"wealth": _INVESTED_WEALTH_GRID, "mood": DiscreteGrid(Mood)},
                state_transitions={
                    "wealth": _grow,
                    "mood": fixed_transition("mood"),
                },
                actions={"investment": _INVESTMENT_GRID},
                functions={"utility": household},
                koopmans_aggregator=aggregators,
            ),
            "couple_terminal": Regime(
                states={"wealth": _INVESTED_WEALTH_GRID},
                functions={
                    "utility": CollectiveUtility(
                        utilities=dict.fromkeys(members, _keep_wealth)
                    )
                },
            ),
        },
        edges={"couple": {"couple_terminal": (0, 1)}},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=RegimeId,
        initial_nodes={0: "couple", 1: "couple"},
    )
    solution = model.solve(
        params={
            "couple": {
                f"koopmans_aggregator_{member}": {"beta": _BETA_BY_LABEL[W]}
                for member, W in aggregators.items()
            }
        },
        log_level="off",
    )
    share = {
        member: (weights or dict.fromkeys(members, 1.0))[member] for member in members
    }
    for period in (0, 1):
        for mood in (Mood.grumpy, Mood.cheerful):
            gains = {
                member: 2.0 * _BETA_BY_CODE[W][mood if W is _mood_W else period] - 1.0
                for member, W in aggregators.items()
            }
            invests = sum(share[member] * gains[member] for member in members) > 0
            np.testing.assert_array_equal(
                np.take(
                    np.asarray(solution.values[period]["couple"]),
                    mood,
                    axis=model.state_names(regime_name="couple").index("mood"),
                ),
                np.array([[gains[member] * invests for member in members]] * 2),
                err_msg=f"period {period}, mood {mood}",
            )
