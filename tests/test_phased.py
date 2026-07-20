"""The `Phased` container and the phase-broadcast grammar of regime slots.

Phase is a broadcast dimension of the regime spec: a bare slot value applies
to both the solve and simulate phases; `Phased(solve=..., simulate=...)`
specifies each phase explicitly. `normalize_regime_phases` expands every slot
into per-phase specs and rejects combinations without defined semantics.
"""

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
from dags import rename_arguments

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.phases import normalize_regime_phases
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Phased,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt


def _solve_variant(wealth: float) -> FloatND:
    return jnp.asarray(wealth)


def _simulate_variant(wealth: float) -> FloatND:
    return jnp.asarray(wealth * 2.0)


def test_phased_stores_both_variants() -> None:
    """`Phased` exposes its variants under `.solve` and `.simulate`."""
    phased = Phased(solve=_solve_variant, simulate=_simulate_variant)
    assert phased.solve is _solve_variant
    assert phased.simulate is _simulate_variant


def test_phased_requires_keyword_arguments() -> None:
    """Both variants must be passed by keyword."""
    with pytest.raises(TypeError):
        Phased(_solve_variant, _simulate_variant)  # ty: ignore[missing-argument, too-many-positional-arguments]


def test_phased_rejects_nested_phased() -> None:
    """A `Phased` variant cannot itself be a `Phased`."""
    inner = Phased(solve=_solve_variant, simulate=_simulate_variant)
    with pytest.raises(RegimeInitializationError, match=r"[Nn]ested"):
        Phased(solve=inner, simulate=_simulate_variant)


def test_phased_accepts_grid_and_callable() -> None:
    """The container is value-agnostic: a grid/callable mix is stored as-is.

    Admissibility of the combination is the per-slot grammar's job, not the
    container's.
    """
    grid = LinSpacedGrid(start=0.0, stop=20.0, n_points=4)
    phased = Phased(solve=_solve_variant, simulate=grid)
    assert phased.solve is _solve_variant
    assert phased.simulate is grid


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _next_regime(age: float) -> ScalarInt:  # noqa: ARG001
    return jnp.asarray(0, dtype=jnp.int32)


def _next_regime_probs(age: float) -> FloatND:  # noqa: ARG001
    return jnp.asarray([1.0, 0.0])


def _next_wealth(wealth: float, consumption: float) -> float:
    return wealth - consumption


def _impute_pension_wealth(aime: float) -> float:
    return aime * 0.1


def _evolve_pension_wealth(pension_wealth: float) -> float:
    return pension_wealth * 1.03


def _evolve_pension_wealth_probs(pension_wealth: float) -> FloatND:
    return jnp.asarray(pension_wealth)


def _pension_grid() -> LinSpacedGrid:
    return LinSpacedGrid(start=0.0, stop=20.0, n_points=4)


def _build_regime(**overrides: Any) -> UserRegime:
    """A small valid regime; tests override individual slots."""
    spec: dict[str, Any] = {
        "transition": _next_regime,
        "states": {
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
        },
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility},
    }
    spec.update(overrides)
    return UserRegime(**spec)


def _carried_states() -> dict[str, Any]:
    return {
        "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
        "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
        "pension_wealth": Phased(
            solve=_impute_pension_wealth, simulate=_pension_grid()
        ),
    }


def _carried_state_transitions() -> dict[str, Any]:
    return {
        "wealth": _next_wealth,
        "aime": lambda aime: aime,
        "pension_wealth": _evolve_pension_wealth,
    }


def test_bare_slot_values_broadcast_to_both_phases() -> None:
    """A regime without `Phased` values normalizes to two identical phase specs."""
    spec = normalize_regime_phases(_build_regime())
    assert spec.solution.functions["utility"] is _utility
    assert spec.simulation.functions["utility"] is _utility
    assert dict(spec.solution.grid_states) == dict(spec.simulation.grid_states)
    assert spec.solution.state_transitions["wealth"] is _next_wealth
    assert spec.simulation.state_transitions["wealth"] is _next_wealth
    assert spec.solution.regime_transition is _next_regime
    assert spec.simulation.regime_transition is _next_regime
    assert spec.carried_only_state_names == frozenset()


def test_phased_function_splits_into_phase_variants() -> None:
    """`Phased` in `functions` assigns each variant to its phase."""
    regime = _build_regime(
        functions={
            "utility": _utility,
            "bonus": Phased(solve=_solve_variant, simulate=_simulate_variant),
        }
    )
    spec = normalize_regime_phases(regime)
    assert spec.solution.functions["bonus"] is _solve_variant
    assert spec.simulation.functions["bonus"] is _simulate_variant


def test_carried_state_derivation() -> None:
    """`Phased(solve=callable, simulate=Grid)` declares a carried state.

    The solve phase computes the name as a derived function (no grid axis);
    the simulate phase carries it as a genuine state whose law of motion is
    the regular `state_transitions` entry, consumed only in simulation.
    """
    regime = _build_regime(
        states=_carried_states(),
        state_transitions=_carried_state_transitions(),
    )
    spec = normalize_regime_phases(regime)
    assert spec.carried_only_state_names == frozenset({"pension_wealth"})
    assert spec.solution.functions["pension_wealth"] is _impute_pension_wealth
    assert "pension_wealth" not in spec.solution.grid_states
    assert "pension_wealth" not in spec.solution.state_transitions
    assert "pension_wealth" not in spec.simulation.functions
    assert isinstance(spec.simulation.grid_states["pension_wealth"], LinSpacedGrid)
    assert spec.simulation.state_transitions["pension_wealth"] is _evolve_pension_wealth


def test_phased_state_transition_splits_into_phase_variants() -> None:
    """`Phased` in `state_transitions` assigns each law to its phase."""

    def _belief_law(wealth: float) -> float:
        return wealth

    def _true_law(wealth: float) -> float:
        return wealth * 1.01

    regime = _build_regime(
        state_transitions={"wealth": Phased(solve=_belief_law, simulate=_true_law)}
    )
    spec = normalize_regime_phases(regime)
    assert spec.solution.state_transitions["wealth"] is _belief_law
    assert spec.simulation.state_transitions["wealth"] is _true_law


def test_phased_regime_transition_splits_into_phase_variants() -> None:
    """`Phased` in `transition` assigns each variant to its phase."""

    def _planned(age: float) -> ScalarInt:  # noqa: ARG001
        return jnp.asarray(0, dtype=jnp.int32)

    regime = _build_regime(transition=Phased(solve=_planned, simulate=_next_regime))
    spec = normalize_regime_phases(regime)
    assert spec.solution.regime_transition is _planned
    assert spec.simulation.regime_transition is _next_regime
    assert spec.solution.stochastic_regime_transition is False
    assert spec.simulation.stochastic_regime_transition is False
    assert spec.terminal is False


def test_phased_markov_regime_transition_sets_stochastic_flags() -> None:
    """Markov variants on both sides mark both phases stochastic."""
    regime = _build_regime(
        transition=Phased(
            solve=MarkovTransition(_next_regime_probs),
            simulate=MarkovTransition(_next_regime_probs),
        )
    )
    spec = normalize_regime_phases(regime)
    assert spec.solution.stochastic_regime_transition is True
    assert spec.simulation.stochastic_regime_transition is True


@pytest.mark.parametrize(
    ("solve_side", "simulate_side", "match"),
    [
        (
            LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
            LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
            "bare Grid",
        ),
        (
            LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
            _impute_pension_wealth,
            "not yet supported",
        ),
        (_impute_pension_wealth, _impute_pension_wealth, "functions"),
    ],
)
def test_invalid_phased_state_combinations_are_rejected(
    solve_side: Any, simulate_side: Any, match: str
) -> None:
    """Of the states matrix, only `Phased(solve=callable, simulate=Grid)` is valid."""
    with pytest.raises(RegimeInitializationError, match=match):
        _build_regime(
            states={
                "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
                "pension_wealth": Phased(solve=solve_side, simulate=simulate_side),
            },
            state_transitions={
                "wealth": _next_wealth,
                "pension_wealth": _evolve_pension_wealth,
            },
        )


@categorical(ordered=False)
class _CoverageStatus:
    uncovered: ScalarInt
    covered: ScalarInt


@pytest.mark.parametrize(
    "grid",
    [
        LinSpacedGrid(start=0.0, stop=20.0, n_points=4, batch_size=1),
        DiscreteGrid(_CoverageStatus, distributed=True),
    ],
)
def test_carried_state_grid_with_solve_only_knobs_is_rejected(grid: Any) -> None:
    """A carried state's grid is simulate metadata; `batch_size`/`distributed`
    apply only to solve grid axes and must not be set on it."""
    with pytest.raises(RegimeInitializationError, match="carried"):
        _build_regime(
            states={
                "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
                "pension_wealth": Phased(solve=_impute_pension_wealth, simulate=grid),
            },
            state_transitions={
                "wealth": _next_wealth,
                "pension_wealth": _evolve_pension_wealth,
            },
        )


def test_process_grid_inside_phased_is_rejected() -> None:
    """Stochastic-process grids have intrinsic transitions and cannot be
    phase-variant."""
    process = NormalIIDProcess(
        n_points=5, batch_size=0, distributed=False, gauss_hermite=True
    )
    with pytest.raises(RegimeInitializationError, match="process"):
        _build_regime(
            states={
                "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
                "shock": Phased(solve=_impute_pension_wealth, simulate=process),
            },
            state_transitions={
                "wealth": _next_wealth,
                "shock": _evolve_pension_wealth,
            },
        )


def test_carried_state_without_law_of_motion_is_rejected() -> None:
    """A carried state is a genuine simulate-phase state and needs a
    `state_transitions` entry like any other state. Coverage is validated
    when the model finalizes its regimes."""
    regime = _build_regime(
        states=_carried_states(),
        state_transitions={"wealth": _next_wealth, "aime": lambda aime: aime},
    )
    with pytest.raises(RegimeInitializationError, match="state_transitions"):
        finalize_regimes(user_regimes={"regime": regime}, derived_categoricals={})


def test_carried_state_with_markov_law_is_rejected() -> None:
    """A carried state's law of motion must be deterministic."""
    with pytest.raises(RegimeInitializationError, match="not yet supported"):
        _build_regime(
            states=_carried_states(),
            state_transitions={
                **_carried_state_transitions(),
                "pension_wealth": MarkovTransition(_evolve_pension_wealth_probs),
            },
        )


def test_carried_state_name_colliding_with_function_is_rejected() -> None:
    """A carried state registers its imputation under the state's name, so a
    regime function of the same name has no unambiguous meaning."""
    with pytest.raises(RegimeInitializationError, match="collid"):
        _build_regime(
            states=_carried_states(),
            state_transitions=_carried_state_transitions(),
            functions={
                "utility": _utility,
                "pension_wealth": _impute_pension_wealth,
            },
        )


def test_terminal_regime_with_carried_state_is_rejected() -> None:
    """Terminal regimes have no next period to carry a state into."""
    with pytest.raises(RegimeInitializationError, match=r"[Tt]erminal"):
        UserRegime(
            transition=None,
            states={
                "pension_wealth": Phased(
                    solve=_impute_pension_wealth, simulate=_pension_grid()
                ),
            },
            functions={"utility": lambda pension_wealth: pension_wealth},
        )


def test_phased_in_constraints_is_rejected() -> None:
    """Constraints cannot be phase-variant: a phase-specific feasible set would
    let the simulated argmax range over actions the value function was never
    computed for."""
    with pytest.raises(RegimeInitializationError, match="feasible"):
        _build_regime(
            constraints={
                "cap": Phased(solve=_solve_variant, simulate=_simulate_variant)
            }
        )


def test_phased_in_actions_is_rejected() -> None:
    """Actions cannot be phase-variant: the simulated argmax must range over
    the menu the value function was computed for."""
    with pytest.raises(RegimeInitializationError):
        _build_regime(
            actions={
                "consumption": Phased(
                    solve=LinSpacedGrid(start=1.0, stop=10.0, n_points=5),
                    simulate=LinSpacedGrid(start=1.0, stop=20.0, n_points=5),
                )
            }
        )


def test_phased_in_derived_categoricals_is_rejected() -> None:
    """Derived categoricals are phase-invariant grid metadata."""
    with pytest.raises(RegimeInitializationError):
        _build_regime(
            derived_categoricals={
                "coverage": Phased(
                    solve=DiscreteGrid(_CoverageStatus),
                    simulate=DiscreteGrid(_CoverageStatus),
                )
            }
        )


def test_phased_inside_per_target_dict_is_rejected() -> None:
    """`Phased` is outermost-only: it wraps a whole slot value, never a
    per-target entry."""
    with pytest.raises(RegimeInitializationError, match="outermost"):
        _build_regime(
            state_transitions={
                "wealth": {"working": Phased(solve=_next_wealth, simulate=_next_wealth)}
            }
        )


def test_phased_regime_transition_with_none_side_is_rejected() -> None:
    """Terminality is phase-invariant: a regime is terminal in both phases or
    neither, so `None` cannot be a `Phased` variant."""
    with pytest.raises(RegimeInitializationError, match=r"[Tt]erminal"):
        _build_regime(transition=Phased(solve=None, simulate=_next_regime))


def test_phased_regime_transition_with_mixed_stochasticity_is_rejected() -> None:
    """Both regime-transition variants must agree on stochasticity."""
    with pytest.raises(RegimeInitializationError, match="stochastic"):
        _build_regime(
            transition=Phased(
                solve=MarkovTransition(_next_regime_probs), simulate=_next_regime
            )
        )


def test_phased_function_with_non_callable_variant_is_rejected() -> None:
    """Each `Phased` variant in `functions` must be a callable."""
    with pytest.raises(RegimeInitializationError, match="callable"):
        _build_regime(
            functions={
                "utility": _utility,
                "bonus": Phased(solve=_solve_variant, simulate="not callable"),
            }
        )


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _next_regime_working(age: float) -> ScalarInt:
    return jnp.where(age < 62, _RegimeId.working, _RegimeId.dead)


def _belief_next_wealth(wealth: float, consumption: float) -> float:
    return wealth - consumption


def _true_next_wealth(wealth: float, consumption: float) -> float:
    return (wealth - consumption) * 1.1


def _consumption_leq_wealth(consumption: float, wealth: float) -> bool:
    return consumption <= wealth


def _build_phased_law_model(*, phased_law: bool) -> Model:
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    working = UserRegime(
        transition=_next_regime_working,
        active=lambda age: age < 64,
        states={"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        state_transitions={
            "wealth": Phased(solve=_belief_next_wealth, simulate=_true_next_wealth)
            if phased_law
            else _belief_next_wealth
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible_consumption": _consumption_leq_wealth},
        functions={"utility": _utility},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=_RegimeId,
    )


def _solve_params(model: Model) -> dict:
    params = cast("dict", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    return params


def test_phased_law_simulation_evolves_state_under_simulate_law() -> None:
    """With `Phased` laws, the panel evolves the state under the simulate law.

    The agent decides under the solved policy (computed with the solve law),
    but the realized next state follows the simulate law.
    """
    model = _build_phased_law_model(phased_law=True)
    params = _solve_params(model)
    result = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions={
            "wealth": jnp.asarray([50.0]),
            "age": jnp.asarray([60.0]),
            "regime_id": jnp.asarray([_RegimeId.working]),
        },
    )
    sim = (
        result.to_dataframe()
        .query('regime_name == "working"')
        .set_index(["subject_id", "period"])
        .sort_index()
    )
    wealth_0 = float(cast("float", sim.loc[(0, 0), "wealth"]))
    consumption_0 = float(cast("float", sim.loc[(0, 0), "consumption"]))
    np.testing.assert_allclose(
        float(cast("float", sim.loc[(0, 1), "wealth"])),
        (wealth_0 - consumption_0) * 1.1,
        rtol=1e-6,
    )


def test_phased_law_solution_matches_bare_solve_law() -> None:
    """The value function only sees the solve law: a `Phased` law's simulate
    variant leaves V identical to the bare solve-law model."""
    phased_solution = _build_phased_law_model(phased_law=True).solve(
        params=_solve_params(_build_phased_law_model(phased_law=True)),
        log_level="debug",
    )
    bare_solution = _build_phased_law_model(phased_law=False).solve(
        params=_solve_params(_build_phased_law_model(phased_law=False)),
        log_level="debug",
    )
    for period, regime_to_V in bare_solution.items():
        for regime_name, expected_V in regime_to_V.items():
            assert bool(jnp.allclose(phased_solution[period][regime_name], expected_V))


def _belief_drift_law(wealth: float, belief_drift: float) -> float:
    return wealth * belief_drift


def _true_drift_law(wealth: float, true_drift: float) -> float:
    return wealth * true_drift


def test_phased_law_params_template_unions_both_variants() -> None:
    """The params template lists both laws' parameters under `next_<state>`."""
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    working = UserRegime(
        transition=_next_regime_working,
        active=lambda age: age < 64,
        states={"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        state_transitions={
            "wealth": Phased(solve=_belief_drift_law, simulate=_true_drift_law)
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        functions={"utility": _utility},
    )
    model = Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=_RegimeId,
    )
    template = model.get_params_template()
    assert "belief_drift" in template["working"]["next_wealth"]
    assert "true_drift" in template["working"]["next_wealth"]


def _income_law(income: float, rho: float, sigma: float) -> float:
    return rho * income + sigma


def _build_wrong_beliefs_model() -> Model:
    """One shared law, `rho` renamed apart per phase, `sigma` shared."""
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    working = UserRegime(
        transition=_next_regime_working,
        active=lambda age: age < 64,
        states={"income": LinSpacedGrid(start=0.0, stop=10.0, n_points=11)},
        state_transitions={
            "income": Phased(
                solve=rename_arguments(_income_law, mapper={"rho": "rho_belief"}),
                simulate=rename_arguments(_income_law, mapper={"rho": "rho_true"}),
            ),
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=3)},
        functions={
            "utility": lambda consumption, income: jnp.log(consumption + 0.1 * income)
        },
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=_RegimeId,
    )


def _simulate_income_panel(
    model: Model, *, rho_belief: float, rho_true: float
) -> np.ndarray:
    result = model.simulate(
        params={
            "discount_factor": 0.95,
            "working": {
                "next_income": {
                    "rho_belief": rho_belief,
                    "rho_true": rho_true,
                    "sigma": 0.5,
                },
            },
        },
        period_to_regime_to_V_arr=None,
        initial_conditions={
            "age": jnp.array([60.0, 60.0]),
            "income": jnp.array([2.0, 4.0]),
            "regime_id": jnp.array([_RegimeId.working] * 2),
        },
        log_level="off",
    )
    df = result.to_dataframe().query("regime_name == 'working'")
    return df.sort_values(["subject_id", "period"])["income"].to_numpy()


def test_renamed_phased_law_params_union_with_shared_name() -> None:
    """Per-side `rename_arguments` splits a param across phases while a name
    kept on both sides stays one shared parameter: the template holds
    `{rho_belief, rho_true, sigma}` under the law's single key."""
    model = _build_wrong_beliefs_model()
    template = model.get_params_template()
    assert set(template["working"]["next_income"]) == {
        "rho_belief",
        "rho_true",
        "sigma",
    }


def test_renamed_phased_law_realized_path_follows_rho_true() -> None:
    """The simulated income path follows the simulate-side persistence:
    `next_income = rho_true * income + sigma`."""
    model = _build_wrong_beliefs_model()
    income = _simulate_income_panel(model, rho_belief=0.95, rho_true=0.8)
    np.testing.assert_allclose(
        income,
        [2.0, 0.8 * 2.0 + 0.5, 4.0, 0.8 * 4.0 + 0.5],
        atol=1e-6,
    )


def test_renamed_phased_law_realized_path_ignores_rho_belief() -> None:
    """`rho_belief` enters only the solve side: changing it leaves the
    realized income path untouched."""
    model = _build_wrong_beliefs_model()
    income_high = _simulate_income_panel(model, rho_belief=0.95, rho_true=0.8)
    income_low = _simulate_income_panel(model, rho_belief=0.2, rho_true=0.8)
    np.testing.assert_allclose(income_high, income_low, atol=0)


def _markov_law(wealth: float) -> FloatND:  # noqa: ARG001
    return jnp.asarray([0.5, 0.5])


def test_markov_variant_in_phased_law_is_accepted() -> None:
    """A `Phased` law may be stochastic: solve = perceived law, simulate = true law.

    The solve variant supplies the probabilities Q integrates the continuation over; the
    simulate variant is the law the next state is actually drawn from. See
    `tests/regime_building/test_perceived_stochastic_transitions.py` for the behavioural
    split.
    """
    _build_regime(
        state_transitions={
            "wealth": Phased(
                solve=MarkovTransition(_markov_law),
                simulate=MarkovTransition(_markov_law),
            )
        }
    )


def test_mixed_stochastic_and_deterministic_phased_law_is_accepted() -> None:
    """The two phases need NOT agree on whether the law is stochastic.

    A deterministic law is a degenerate kernel, so the state has the same domain either
    way, and the two phase cores classify their stochastic names independently. A
    perceived kernel with a point-valued truth is a coherent model, not a kind error.
    `tests/regime_building/test_mixed_stochasticity_phases.py` pins the behaviour in
    both directions.
    """
    _build_regime(
        state_transitions={
            "wealth": Phased(solve=MarkovTransition(_markov_law), simulate=_next_wealth)
        }
    )
    _build_regime(
        state_transitions={
            "wealth": Phased(solve=_next_wealth, simulate=MarkovTransition(_markov_law))
        }
    )


def test_mixed_per_target_dict_phased_law_is_accepted() -> None:
    """Same for the per-target-dict form, where the cells carry the stochastic-ness."""
    _build_regime(
        state_transitions={
            "wealth": Phased(
                solve={"working": MarkovTransition(_markov_law)},
                simulate={"working": _next_wealth},
            )
        }
    )


def test_per_target_dict_with_bare_other_phase_is_rejected() -> None:
    """A per-target dict on one side and a bare law on the other is REJECTED.

    Although the bare law broadcasts cleanly at the function level, the params template
    is a union across phases and keys a param's coarseness by whether a
    `template[target]` branch exists. The per-target side fabricates such branches, so
    the bare side's coarse param is silently collapsed onto the per-target leaves and
    the two phases can no longer be parameterized independently (belief ≠ truth
    becomes unexpressible). The supported spelling is two per-target dicts over the
    same targets.
    """
    with pytest.raises(RegimeInitializationError, match="different shapes"):
        _build_regime(
            state_transitions={
                "wealth": Phased(
                    solve={"working": MarkovTransition(_markov_law)},
                    simulate=_next_wealth,
                )
            }
        )


def test_bare_solve_per_target_simulate_is_rejected() -> None:
    """The mirror spelling — bare solve, per-target simulate dict — is REJECTED too."""
    with pytest.raises(RegimeInitializationError, match="different shapes"):
        _build_regime(
            state_transitions={
                "wealth": Phased(
                    solve=_next_wealth,
                    simulate={"working": _next_wealth},
                )
            }
        )


def test_per_target_dicts_with_different_targets_are_rejected() -> None:
    """Both phases' per-target dicts must cover the same targets.

    Otherwise normalization leaves a target with a law in one phase and none in the
    other.
    """
    with pytest.raises(RegimeInitializationError, match="different targets"):
        _build_regime(
            state_transitions={
                "wealth": Phased(
                    solve={"working": _next_wealth},
                    simulate={"retired": _next_wealth},
                )
            }
        )


def _plan_next_regime(age: float) -> ScalarInt:
    """Solve-phase plan: stay working until 62."""
    return jnp.where(age < 62, _RegimeId.working, _RegimeId.dead)


def _realized_next_regime(age: float) -> ScalarInt:  # noqa: ARG001
    """Simulate-phase realization: death after the first period."""
    return jnp.asarray(_RegimeId.dead, dtype=jnp.int32)


def _build_phased_transition_model(*, phased_transition: bool) -> Model:
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    working = UserRegime(
        transition=Phased(solve=_plan_next_regime, simulate=_realized_next_regime)
        if phased_transition
        else _plan_next_regime,
        active=lambda age: age < 64,
        states={"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible_consumption": _consumption_leq_wealth},
        functions={"utility": _utility},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=_RegimeId,
    )


def test_phased_regime_transition_realized_draw_follows_simulate_variant() -> None:
    """The realized regime draw follows the simulate variant of a `Phased`
    regime transition, while the policy was solved under the solve variant."""
    model = _build_phased_transition_model(phased_transition=True)
    params = _solve_params(model)
    result = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions={
            "wealth": jnp.asarray([50.0]),
            "age": jnp.asarray([60.0]),
            "regime_id": jnp.asarray([_RegimeId.working]),
        },
    )
    sim = result.to_dataframe().set_index(["subject_id", "period"]).sort_index()
    assert sim.loc[(0, 0), "regime_name"] == "working"
    assert sim.loc[(0, 1), "regime_name"] == "dead"


def test_phased_regime_transition_solution_matches_bare_solve_variant() -> None:
    """V only sees the solve variant of a `Phased` regime transition."""
    phased_solution = _build_phased_transition_model(phased_transition=True).solve(
        params=_solve_params(_build_phased_transition_model(phased_transition=True)),
        log_level="debug",
    )
    bare_solution = _build_phased_transition_model(phased_transition=False).solve(
        params=_solve_params(_build_phased_transition_model(phased_transition=False)),
        log_level="debug",
    )
    for period, regime_to_V in bare_solution.items():
        for regime_name, expected_V in regime_to_V.items():
            assert bool(jnp.allclose(phased_solution[period][regime_name], expected_V))


def _retire_when_pension_rich(pension_wealth: float, age: float) -> ScalarInt:
    return jnp.where(
        (pension_wealth > 10.0) | (age >= 62), _RegimeId.dead, _RegimeId.working
    )


def test_regime_draw_reads_carried_value() -> None:
    """The realized regime draw reads carried values, not solve imputations.

    Two subjects with equal AIME (equal imputation) but different carried
    pension wealth get different realized regime draws when the transition
    reads the carried state; the decision itself still follows the policy
    solved on the imputation.
    """
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    working = UserRegime(
        transition=_retire_when_pension_rich,
        active=lambda age: age < 64,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            "pension_wealth": Phased(
                solve=_impute_pension_wealth, simulate=_pension_grid()
            ),
        },
        state_transitions={
            "wealth": _next_wealth,
            "aime": lambda aime: aime,
            "pension_wealth": _evolve_pension_wealth,
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible_consumption": _consumption_leq_wealth},
        functions={"utility": _utility},
    )
    model = Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=_RegimeId,
    )
    params = _solve_params(model)
    result = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions={
            "wealth": jnp.full(2, 50.0),
            "aime": jnp.full(2, 20.0),  # imputation = 2.0 for both subjects
            "pension_wealth": jnp.asarray([5.0, 15.0]),
            "age": jnp.full(2, 60.0),
            "regime_id": jnp.asarray([_RegimeId.working] * 2),
        },
    )
    sim = result.to_dataframe().set_index(["subject_id", "period"]).sort_index()
    assert sim.loc[(0, 1), "regime_name"] == "working"
    assert sim.loc[(1, 1), "regime_name"] == "dead"
