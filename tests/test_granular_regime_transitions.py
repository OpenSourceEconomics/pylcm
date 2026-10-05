"""Per-target stochastic regime transitions.

The granular form declares each target regime's transition probability as its
own function; the key set IS the regime's reachability declaration — omitted
targets are structurally unreachable. The coarse forms (one callable over all
regimes) remain valid and reach every regime.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Phased,
    StochasticTransition,
    categorical,
)
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarFloat, ScalarInt
from tests.test_models.graph import with_fixture_graph
from tests.test_models.schedules import until_exit


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _TerminalRegimeId:
    dead: ScalarInt


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _next_wealth(*, wealth: float, consumption: float) -> float:
    return wealth - consumption


def _prob_work(age: int) -> ScalarFloat:
    return jnp.where(age < 1, 0.5, 0.0)


def _prob_retired(age: int) -> ScalarFloat:
    return jnp.where(age < 1, 0.5, 1.0) * 0.5


def _prob_dead(age: int) -> ScalarFloat:
    return jnp.asarray(1.0 - _prob_work(age) - _prob_retired(age))


def _granular_transition() -> dict[str, StochasticTransition]:
    return {
        "work": StochasticTransition(func=_prob_work),
        "retired": StochasticTransition(func=_prob_retired),
        "dead": StochasticTransition(func=_prob_dead),
    }


def _build_regime(**overrides: Any) -> UserRegime:
    spec: dict[str, Any] = {
        "regime_transitions": until_exit(
            2, law=_granular_transition(), exits=("retired", "dead")
        ),
        "states": {"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility},
    }
    spec.update(overrides)
    return UserRegime(**spec)


def _build_model(*, work: UserRegime, retired: UserRegime | None = None) -> Model:
    # `retired` outlives `work` by one age so that the mass `work` sends it in
    # its final transition lands on an active regime, and hands everything to
    # `dead` in its own final transition.
    if retired is None:
        retired = _build_regime(
            regime_transitions=until_exit(
                3,
                law={
                    "retired": StochasticTransition(
                        func=lambda age: jnp.where(age < 2, 0.5, 0.0),
                    ),
                    "dead": StochasticTransition(
                        func=lambda age: jnp.where(age < 2, 0.5, 1.0),
                    ),
                },
                exits=("dead",),
            ),
        )
    dead = UserRegime(
        regime_transitions=None,
        functions={"utility": lambda: 0.0},
    )
    return with_fixture_graph(
        regimes={"work": work, "retired": retired, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "work"},
    )


def test_granular_transition_solves_and_simulates() -> None:
    """A model with per-target regime transitions solves and simulates; the
    realized memberships are drawn from the declared distribution."""
    model = _build_model(work=_build_regime())
    result = model.simulate(
        params={
            "work": {"discount_factor": 0.95},
            "retired": {"discount_factor": 0.95},
        },
        initial_conditions={
            "age": jnp.zeros(40),
            "wealth": jnp.full(40, 50.0),
            "regime_id": jnp.full(40, _RegimeId.work),
        },
        log_level="debug",
        seed=7,
    )
    df = result.to_dataframe()
    last = df.loc[df["period"] == 2, "regime_name"]
    # From period 1 on, work assigns zero probability to itself.
    assert set(last) <= {"retired", "dead"}


def test_template_has_per_target_regime_transition_keys() -> None:
    """Each granular cell's parameters surface under the target's branch."""

    def _prob_dead_with_param(*, age: int, hazard: float) -> ScalarFloat:
        return jnp.clip(hazard * (1.0 + age), 0.0, 1.0)

    work = _build_regime(
        regime_transitions=until_exit(
            3,
            law={
                "work": StochasticTransition(
                    func=lambda age, hazard: (
                        1.0 - _prob_dead_with_param(age=age, hazard=hazard)
                    )
                ),
                "dead": StochasticTransition(func=_prob_dead_with_param),
            },
            exits=("dead",),
        ),
    )
    model = _build_model(work=work)
    template = model.get_params_template()
    assert "hazard" in template["work"]["dead"]["next_regime"]
    assert "hazard" in template["work"]["work"]["next_regime"]


def test_plain_callable_cell_is_rejected() -> None:
    """Granular cells must be `StochasticTransition`-wrapped."""
    with pytest.raises(RegimeInitializationError, match=r"StochasticTransition"):
        _build_regime(
            regime_transitions={
                "work": lambda age: 1.0,  # noqa: ARG005
            },
        )


def test_empty_granular_dict_is_rejected() -> None:
    """`regime_transitions={}` is not the terminal spelling; terminality is `None`."""
    with pytest.raises(RegimeInitializationError, match=r"regime_transitions=None"):
        _build_regime(regime_transitions={})


def test_unknown_target_in_granular_dict_raises() -> None:
    """Every granular key must name a regime of the model."""
    work = _build_regime(
        regime_transitions={
            "work": StochasticTransition(func=lambda age: jnp.asarray(0.5)),  # noqa: ARG005
            "valhalla": StochasticTransition(func=lambda age: jnp.asarray(0.5)),  # noqa: ARG005
        },
    )
    with pytest.raises(ModelInitializationError, match=r"valhalla"):
        _build_model(work=work)


@pytest.mark.parametrize("phase_local_handoffs", [False, True])
def test_disjoint_phase_targets_price_perceived_choice_and_realize_exit(
    *,
    phase_local_handoffs: bool,
) -> None:
    """Price investment using work's payoff and realize retired's opposite payoff."""
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)

    def price_handoff(*, investment: float, perceived_scale: float) -> float:
        return investment * perceived_scale

    def realized_handoff(*, investment: float, realized_scale: float) -> float:
        return investment * realized_scale

    state_law = (
        Phased(
            solve={"dead": price_handoff},
            simulate={"retired": realized_handoff},
        )
        if phase_local_handoffs
        else lambda investment: investment
    )
    model = with_fixture_graph(
        regimes={
            "work": UserRegime(
                regime_transitions=Phased(
                    solve={
                        "dead": StochasticTransition(
                            func=lambda perceived_mass: perceived_mass
                        )
                    },
                    simulate={
                        "retired": StochasticTransition(
                            func=lambda realized_mass: realized_mass
                        )
                    },
                ),
                states={"wealth": grid},
                actions={"investment": grid},
                state_transitions={"wealth": state_law},
                functions={"utility": lambda wealth: 0.0 * wealth},
            ),
            "dead": UserRegime(
                regime_transitions=None,
                states={"wealth": grid},
                functions={"utility": lambda wealth: wealth},
            ),
            "retired": UserRegime(
                regime_transitions=None,
                states={"wealth": grid},
                functions={"utility": lambda wealth: -wealth},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "work"},
        enable_jit=False,
    )
    template = model.get_params_template()["work"]
    assert "perceived_mass" in template["dead"]["next_regime"]
    assert "realized_mass" in template["retired"]["next_regime"]
    params = {"discount_factor": 1.0, "perceived_mass": 1.0, "realized_mass": 1.0}
    if phase_local_handoffs:
        params |= {"perceived_scale": 1.0, "realized_scale": 1.0}
    solution = model.solve(params=params, log_level="debug")
    np.testing.assert_array_equal(solution.values[0]["work"], jnp.ones(2))
    assert model.reachability.solution.targets(period=0, source="work") == ("dead",)
    assert model.reachability.simulation.targets(period=0, source="work") == (
        "retired",
    )
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.array([0.0]),
            "wealth": jnp.array([0.0]),
            "regime_id": jnp.array([_RegimeId.work]),
        },
        seed=7,
        log_level="debug",
    ).to_dataframe()
    assert result.loc[result["period"] == 0, "investment"].tolist() == [1.0]
    assert result.loc[result["period"] == 1, "regime_name"].tolist() == ["retired"]
    assert result.loc[result["period"] == 1, "wealth"].tolist() == [1.0]


@pytest.mark.parametrize("name", ["DeterministicTransition", "StochasticTransition"])
def test_public_transition_names_describe_the_law_kind(name: str) -> None:
    """Expose paired deterministic and stochastic transition declarations."""
    assert hasattr(lcm, name)


def test_model_accepts_initial_nodes_as_age_regime_pair_rules() -> None:
    """Admit exactly the initial age-regime pair named by the node selector."""
    model = with_fixture_graph(
        regimes={
            "dead": UserRegime(
                regime_transitions=None,
                functions={"utility": lambda: 0.0},
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_TerminalRegimeId,
        initial_nodes={0: "dead"},
        enable_jit=False,
    )
    assert model.initial_nodes == frozenset({(0, "dead")})


def test_fixed_zero_probability_does_not_hide_an_invalid_empty_distribution() -> None:
    """Refuse zero total probability after retaining an all-zero transition law."""
    terminal = UserRegime(
        regime_transitions=None,
        functions={"utility": lambda: 0.0},
    )
    model = with_fixture_graph(
        regimes={
            "work": _build_regime(
                regime_transitions={
                    "dead": StochasticTransition(func=lambda probability: probability)
                },
            ),
            "retired": terminal,
            "dead": terminal,
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "work"},
        fixed_params={"probability": 0.0},
        enable_jit=False,
    )
    assert model.reachability.solution.targets(period=0, source="work") == ("dead",)
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params={"discount_factor": 1.0}, log_level="off")


def test_simulation_only_regime_prices_its_perceived_continuation() -> None:
    """Solve the perceived continuation needed by a simulation-only decision node."""
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    certain = StochasticTransition(func=lambda: jnp.asarray(1.0))
    model = with_fixture_graph(
        regimes={
            "work": UserRegime(
                regime_transitions=Phased(
                    solve={"dead": certain},
                    simulate={"retired": certain},
                ),
                states={"wealth": grid},
                state_transitions={"wealth": lambda wealth: wealth},
                functions={"utility": lambda wealth: 0.0 * wealth},
            ),
            "retired": UserRegime(
                regime_transitions={"dead": certain},
                states={"wealth": grid},
                actions={"investment": grid},
                state_transitions={"wealth": lambda investment: investment},
                functions={"utility": lambda wealth: 0.0 * wealth},
            ),
            "dead": UserRegime(
                regime_transitions=None,
                states={"wealth": grid},
                functions={"utility": lambda wealth: wealth},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "work"},
        enable_jit=False,
    )
    assert model.reachability.nodes == frozenset(
        {(0, "work"), (1, "retired"), (1, "dead"), (2, "dead")}
    )
    assert model.reachability.visited_nodes == frozenset(
        {(0, "work"), (1, "retired"), (2, "dead")}
    )
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="debug")
    np.testing.assert_array_equal(solution.values[1]["retired"], jnp.ones(2))
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.array([0.0]),
            "wealth": jnp.array([0.0]),
            "regime_id": jnp.array([_RegimeId.work]),
        },
        seed=7,
        log_level="debug",
    ).to_dataframe()
    assert result["regime_name"].tolist() == ["work", "retired", "dead"]
    assert result.loc[result["period"] == 1, "investment"].tolist() == [1.0]
    assert result.loc[result["period"] == 2, "wealth"].tolist() == [1.0]


def test_uncovered_reachable_target_raises_with_remedy() -> None:
    """A per-target state law must cover every reachable target carrying the
    state; the error points to the granular transition spelling."""
    work = _build_regime(
        state_transitions={"wealth": {"work": _next_wealth}},
        # Declares retired as reachable.
        regime_transitions=until_exit(
            2, law=_granular_transition(), exits=("retired", "dead")
        ),
    )
    with pytest.raises(
        ModelInitializationError, match=r"retired.*wealth|wealth.*retired"
    ):
        _build_model(work=work)


def test_granular_keys_narrow_reachability() -> None:
    """A per-target state law covering exactly the declared targets is valid —
    the granular key set, not coverage inference, decides reachability."""
    work = _build_regime(
        regime_transitions=until_exit(
            3,
            law={
                "retired": StochasticTransition(func=lambda age: jnp.asarray(0.7)),  # noqa: ARG005
                "dead": StochasticTransition(func=lambda age: jnp.asarray(0.3)),  # noqa: ARG005
            },
            exits=("dead",),
        ),
        state_transitions={"wealth": {"retired": _next_wealth}},
    )
    model = _build_model(work=work)
    assert "work" in model.user_regimes


def test_per_target_regime_transition_template_nests_under_target() -> None:
    """Granular cells' params nest under the target in the template."""
    work = _build_regime()
    model = _build_model(work=work)
    template = model.get_params_template()
    assert "next_regime" in template["work"]["retired"]
