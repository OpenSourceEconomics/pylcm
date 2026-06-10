"""Phase-variant state: imputed during solve, seeded and evolved during simulate.

A `SolveSimulateStatePair` placed in `Regime.states` gives a quantity two roles:
- solve: a derived function (here, pension wealth imputed from AIME) that never
  becomes a grid dimension, so the solve grid is unchanged;
- simulate: a genuine state seeded from the initial conditions and carried
  forward each period via its transition, read by simulate functions instead of
  the solve-phase imputation.
"""

from collections.abc import Mapping
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    SolveSimulateStatePair,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 62, RegimeId.working, RegimeId.dead)


def _impute_pension_wealth(aime: float) -> float:
    """Solve-phase pension wealth: imputed from AIME."""
    return aime * 0.1


def _evolve_pension_wealth(pension_wealth: float) -> float:
    """Simulate-phase pension wealth: grows by a fixed factor each period."""
    return pension_wealth * 1.03


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _next_wealth(wealth: float, consumption: float, pension_wealth: float) -> float:
    return wealth - consumption + pension_wealth


def _next_aime(aime: float) -> float:
    return aime


def _consumption_leq_wealth(consumption: float, wealth: float) -> bool:
    return consumption <= wealth


def _working_active(age: float) -> bool:
    return age < 64


def _build_pension_regime() -> UserRegime:
    """A non-terminal regime whose pension wealth is a `SolveSimulateStatePair`."""
    return UserRegime(
        transition=_next_regime,
        active=_working_active,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            "pension_wealth": SolveSimulateStatePair(
                solve=_impute_pension_wealth,
                grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                transition=_evolve_pension_wealth,
            ),
        },
        state_transitions={
            "wealth": _next_wealth,
            "aime": _next_aime,
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible_consumption": _consumption_leq_wealth},
        functions={"utility": _utility},
    )


_DEAD = UserRegime(transition=None, functions={"utility": lambda: 0.0})


def _build_pension_model(*, pension_as_pair: bool) -> Model:
    """A two-regime model with pension wealth as a state pair or a plain function.

    With `pension_as_pair=False` pension wealth is an ordinary derived function
    of AIME, present in both phases; this is the behavior the state-pair's solve
    phase must reproduce exactly.
    """
    if pension_as_pair:
        working = _build_pension_regime()
    else:
        working = UserRegime(
            transition=_next_regime,
            active=_working_active,
            states={
                "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
                "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            },
            state_transitions={"wealth": _next_wealth, "aime": _next_aime},
            actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
            constraints={"feasible_consumption": _consumption_leq_wealth},
            functions={"utility": _utility, "pension_wealth": _impute_pension_wealth},
        )
    return Model(
        regimes={"working": working, "dead": _DEAD},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=RegimeId,
    )


def test_regime_accepts_state_pair_in_states() -> None:
    """A `SolveSimulateStatePair` is a valid value in `Regime.states`."""
    regime = _build_pension_regime()
    assert isinstance(regime.states["pension_wealth"], SolveSimulateStatePair)


def test_get_all_functions_exposes_state_pair_transition() -> None:
    """`get_all_functions` exposes a state pair's transition under `next_<name>`.

    The params template lists the transition's parameters under `next_<name>`,
    so Series-parameter conversion (which looks each template function up by
    that name) requires the transition to be resolvable there as well.
    """
    regime = _build_pension_regime()
    funcs = regime.get_all_functions()
    assert funcs["next_pension_wealth"] is _evolve_pension_wealth


def test_state_pair_in_state_transitions_raises() -> None:
    """A pair carries its own transition; listing it in `state_transitions` errors."""
    with pytest.raises(RegimeInitializationError, match="carry their own transition"):
        UserRegime(
            transition=_next_regime,
            states={
                "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
                "pension_wealth": SolveSimulateStatePair(
                    solve=_impute_pension_wealth,
                    grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                    transition=_evolve_pension_wealth,
                ),
            },
            state_transitions={
                "aime": lambda aime: aime,
                "pension_wealth": _evolve_pension_wealth,
            },
            functions={"utility": lambda pension_wealth: pension_wealth},
        )


def test_solve_grid_excludes_state_pair() -> None:
    """A state pair is a function in solve, so it is not a solve grid dimension."""
    model = _build_pension_model(pension_as_pair=True)
    solve_state_names = model._regimes["working"].variables.state_names
    assert set(solve_state_names) == {"wealth", "aime"}


def _solve_pension_model(model: Model) -> Mapping[int, Mapping[str, FloatND]]:
    params = cast("dict[str, Any]", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    return model.solve(params=params, log_level="debug")


def test_state_pair_solves_like_plain_function() -> None:
    """Imputing pension wealth via a state pair gives the same value function as
    imputing it via an ordinary derived function."""
    pair_solution = _solve_pension_model(_build_pension_model(pension_as_pair=True))
    plain_solution = _solve_pension_model(_build_pension_model(pension_as_pair=False))

    for period, regime_to_V in plain_solution.items():
        for regime_name, expected_V in regime_to_V.items():
            assert bool(jnp.allclose(pair_solution[period][regime_name], expected_V))


def _simulate_pension(model: Model, *, pension_seed: list[float]) -> pd.DataFrame:
    params = cast("dict[str, Any]", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    n = len(pension_seed)
    result = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions={
            "wealth": jnp.full(n, 50.0),
            "aime": jnp.full(n, 20.0),
            "pension_wealth": jnp.asarray(pension_seed),
            "age": jnp.full(n, 60.0),
            "regime_id": jnp.array([RegimeId.working] * n),
        },
    )
    working = result.to_dataframe().query('regime_name == "working"')
    return working.set_index(["subject_id", "period"]).sort_index()


def test_simulate_seeds_and_evolves_true_pension_wealth() -> None:
    """Pension wealth is seeded from the initial conditions and evolved by its own
    transition each period (the true value, not the AIME imputation)."""
    sim = _simulate_pension(
        _build_pension_model(pension_as_pair=True), pension_seed=[5.0, 15.0]
    )
    pension = sim["pension_wealth"]
    np.testing.assert_allclose(pension.loc[(0, 0)], 5.0)
    np.testing.assert_allclose(pension.loc[(1, 0)], 15.0)
    np.testing.assert_allclose(pension.loc[(0, 1)], 5.0 * 1.03)
    np.testing.assert_allclose(pension.loc[(1, 1)], 15.0 * 1.03)


def test_simulate_decides_on_imputed_but_accounts_on_true_pension() -> None:
    """Two subjects with equal AIME impute equal pension, so they choose the same
    consumption; their realized next wealth differs by exactly the true-pension gap."""
    sim = _simulate_pension(
        _build_pension_model(pension_as_pair=True), pension_seed=[5.0, 15.0]
    )
    consumption = sim["consumption"]
    np.testing.assert_allclose(consumption.loc[(0, 0)], consumption.loc[(1, 0)])

    wealth = sim["wealth"]
    np.testing.assert_allclose(wealth.loc[(1, 1)] - wealth.loc[(0, 1)], 15.0 - 5.0)


def test_simulate_aot_compiled_carries_state_pair() -> None:
    """A state pair survives the AOT-compiled simulate path.

    Setting `n_subjects` AOT-compiles every simulate program for that batch.
    The compiled `next_state` program reads the pair's simulate-only state, so
    its lower-args must seed that state — otherwise compilation fails before
    the first period runs.
    """
    model = Model(
        regimes={"working": _build_pension_regime(), "dead": _DEAD},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=RegimeId,
        n_subjects=2,
    )
    params = cast("dict[str, Any]", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    result = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions={
            "wealth": jnp.full(2, 50.0),
            "aime": jnp.full(2, 20.0),
            "pension_wealth": jnp.asarray([5.0, 15.0]),
            "age": jnp.full(2, 60.0),
            "regime_id": jnp.array([RegimeId.working] * 2),
        },
    )
    sim = (
        result.to_dataframe()
        .query('regime_name == "working"')
        .set_index(["subject_id", "period"])
        .sort_index()
    )
    np.testing.assert_allclose(sim["pension_wealth"].loc[(0, 1)], 5.0 * 1.03)


@categorical(ordered=False)
class _ThreeRegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _next_regime_from_working(age: float) -> ScalarInt:
    return jnp.where(age < 62, _ThreeRegimeId.working, _ThreeRegimeId.retired)


def _next_regime_from_retired(age: float) -> ScalarInt:
    return jnp.where(age < 64, _ThreeRegimeId.retired, _ThreeRegimeId.dead)


def _retired_imputed_pension_wealth() -> float:
    return 12.0


def _retired_utility(pension_wealth: float) -> FloatND:
    return jnp.log(pension_wealth)


_DEAD3 = UserRegime(transition=None, functions={"utility": lambda: 0.0})


def _build_handover_model() -> Model:
    """Three regimes where retirement keeps only the carried pension wealth.

    The working regime hands over nothing but the pair state to `retired`
    (no ordinary state is shared), so the pair's transition is the only
    state hand-over on the crossing.
    """
    working = UserRegime(
        transition=_next_regime_from_working,
        active=lambda age: age < 64,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            "pension_wealth": SolveSimulateStatePair(
                solve=_impute_pension_wealth,
                grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                transition=_evolve_pension_wealth,
            ),
        },
        state_transitions={"wealth": _next_wealth, "aime": _next_aime},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible_consumption": _consumption_leq_wealth},
        functions={"utility": _utility},
    )
    retired = UserRegime(
        transition=_next_regime_from_retired,
        active=lambda age: 64 <= age < 66,
        states={
            "pension_wealth": SolveSimulateStatePair(
                solve=_retired_imputed_pension_wealth,
                grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                transition=_evolve_pension_wealth,
            ),
        },
        state_transitions={},
        functions={"utility": _retired_utility},
    )
    return Model(
        regimes={"working": working, "retired": retired, "dead": _DEAD3},
        ages=AgeGrid(start=60, stop=66, step="2Y"),
        regime_id_class=_ThreeRegimeId,
    )


def test_simulate_evolves_pair_across_pair_only_handover() -> None:
    """The carried pair value is evolved, not frozen, on a pair-only crossing.

    Retirement keeps only pension wealth; entering it, the working regime's
    pair transition must apply, so the carried value grows by its factor
    instead of being copied unchanged.
    """
    model = _build_handover_model()
    params = cast("dict[str, Any]", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    params["retired"]["H"]["discount_factor"] = 0.95
    result = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions={
            "wealth": jnp.full(1, 50.0),
            "aime": jnp.full(1, 20.0),
            "pension_wealth": jnp.asarray([10.0]),
            "age": jnp.full(1, 60.0),
            "regime_id": jnp.array([_ThreeRegimeId.working]),
        },
    )
    sim = result.to_dataframe().set_index(["subject_id", "period"]).sort_index()
    assert sim.loc[(0, 2), "regime_name"] == "retired"
    np.testing.assert_allclose(
        float(cast("float", sim.loc[(0, 2), "pension_wealth"])), 10.0 * 1.03**2
    )


@pytest.mark.parametrize(
    ("pair_kwargs", "match"),
    [
        (
            {
                "solve": "not callable",
                "grid": LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                "transition": _evolve_pension_wealth,
            },
            "solve",
        ),
        (
            {
                "solve": _impute_pension_wealth,
                "grid": 42,
                "transition": _evolve_pension_wealth,
            },
            "grid",
        ),
        (
            {
                "solve": _impute_pension_wealth,
                "grid": LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                "transition": None,
            },
            "transition",
        ),
    ],
)
def test_malformed_state_pair_is_rejected_at_regime_construction(
    pair_kwargs: dict[str, Any], match: str
) -> None:
    """A pair with a non-callable solve/transition or a non-grid grid errors loudly.

    Each field is part of the pair's contract: `solve` and `transition` must be
    callable, `grid` must be an LCM grid. A malformed field must surface at
    `Regime` construction, not as an opaque failure deep inside processing.
    """
    with pytest.raises(RegimeInitializationError, match=match):
        UserRegime(
            transition=_next_regime,
            states={
                "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
                "pension_wealth": SolveSimulateStatePair(**pair_kwargs),
            },
            state_transitions={"aime": _next_aime},
            functions={"utility": lambda pension_wealth: pension_wealth},
        )


def test_terminal_regime_with_state_pair_is_rejected() -> None:
    """A terminal regime cannot carry a state pair.

    Terminal regimes have no transitions, so a pair's carry-forward role is
    meaningless there — and silently registering its `next_<name>` would leak
    a transition into a regime that must not have one.
    """
    with pytest.raises(RegimeInitializationError, match=r"[Tt]erminal"):
        UserRegime(
            transition=None,
            states={
                "pension_wealth": SolveSimulateStatePair(
                    solve=_impute_pension_wealth,
                    grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                    transition=_evolve_pension_wealth,
                ),
            },
            functions={"utility": lambda pension_wealth: pension_wealth},
        )


def _evolve_pension_wealth_probs(pension_wealth: float) -> FloatND:
    return jnp.asarray(pension_wealth)


def test_markov_transition_as_pair_transition_is_rejected() -> None:
    """A pair's transition must be deterministic; `MarkovTransition` errors."""
    with pytest.raises(RegimeInitializationError, match="MarkovTransition"):
        UserRegime(
            transition=_next_regime,
            states={
                "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
                "pension_wealth": SolveSimulateStatePair(
                    solve=_impute_pension_wealth,
                    grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
                    transition=MarkovTransition(_evolve_pension_wealth_probs),
                ),
            },
            state_transitions={"aime": _next_aime},
            functions={"utility": lambda pension_wealth: pension_wealth},
        )


@categorical(ordered=False)
class _CoverageStatus:
    uncovered: ScalarInt
    covered: ScalarInt


def _make_pair_grid(grid_kwargs: dict[str, Any]) -> Any:
    if grid_kwargs.get("distributed"):
        # Continuous grids reject `distributed` at construction, so the
        # sharded case needs a discrete pair grid.
        return DiscreteGrid(_CoverageStatus, **grid_kwargs)
    return LinSpacedGrid(start=0.0, stop=20.0, n_points=4, **grid_kwargs)


@pytest.mark.parametrize(
    "grid_kwargs",
    [{"batch_size": 1}, {"distributed": True}],
)
def test_sharded_or_batched_pair_grid_is_rejected(grid_kwargs: dict[str, Any]) -> None:
    """A pair's grid cannot be sharded or batched.

    The pair's grid is the simulate-phase domain of a carried per-subject
    value, not a solve dimension — device sharding and chunked-vmap batching
    only apply to solve grid axes.
    """
    with pytest.raises(RegimeInitializationError, match="pair"):
        UserRegime(
            transition=_next_regime,
            states={
                "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
                "pension_wealth": SolveSimulateStatePair(
                    solve=_impute_pension_wealth,
                    grid=_make_pair_grid(grid_kwargs),
                    transition=_evolve_pension_wealth,
                ),
            },
            state_transitions={"aime": _next_aime},
            functions={"utility": lambda pension_wealth: pension_wealth},
        )
