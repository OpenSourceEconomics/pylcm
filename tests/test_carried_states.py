"""Carried states: imputed during solve, seeded and evolved during simulate.

A `Phased(solve=callable, simulate=Grid)` value in `Regime.states` gives a
quantity two roles:
- solve: a derived function (here, pension wealth imputed from AIME) that never
  becomes a grid dimension, so the solve grid is unchanged;
- simulate: a genuine state seeded from the initial conditions and carried
  forward each period via its regular `state_transitions` law, read by
  simulate functions instead of the solve-phase imputation.
"""

from collections.abc import Mapping
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Phased,
    categorical,
)
from lcm.exceptions import InvalidInitialConditionsError, ModelInitializationError
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
    """A non-terminal regime whose pension wealth is a carried state."""
    return UserRegime(
        transition=_next_regime,
        active=_working_active,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            "pension_wealth": Phased(
                solve=_impute_pension_wealth,
                simulate=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
            ),
        },
        state_transitions={
            "wealth": _next_wealth,
            "aime": _next_aime,
            "pension_wealth": _evolve_pension_wealth,
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


def test_regime_accepts_carried_state_in_states() -> None:
    """A `Phased(solve=callable, simulate=Grid)` is a valid value in `Regime.states`."""
    regime = _build_pension_regime()
    assert isinstance(regime.states["pension_wealth"], Phased)


def test_get_all_functions_exposes_carried_law() -> None:
    """`get_all_functions` exposes a carried state's law under `next_<name>`.

    The params template lists the law's parameters under `next_<name>`, so
    Series-parameter conversion (which looks each template function up by
    that name) requires the law to be resolvable there as well.
    """
    regime = _build_pension_regime()
    funcs = regime.get_all_functions()
    assert funcs["next_pension_wealth"] is _evolve_pension_wealth


def test_solve_grid_excludes_carried_state() -> None:
    """A carried state is a function in solve, so it is not a solve grid dimension."""
    model = _build_pension_model(pension_as_pair=True)
    solve_state_names = model._regimes["working"].solution.state_names
    assert set(solve_state_names) == {"wealth", "aime"}


def _solve_pension_model(model: Model) -> Mapping[int, Mapping[str, FloatND]]:
    params = cast("dict[str, Any]", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    return model.solve(params=params, log_level="debug")


def test_carried_state_solves_like_plain_function() -> None:
    """Imputing pension wealth via a carried state gives the same value function
    as imputing it via an ordinary derived function."""
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


def test_simulate_aot_compiled_carries_carried_state() -> None:
    """A carried state survives the AOT-compiled simulate path.

    Setting `n_subjects` AOT-compiles every simulate program for that batch.
    The compiled `next_state` program reads the carried simulate-only state,
    so its lower-args must seed that state — otherwise compilation fails
    before the first period runs.
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

    The working regime hands over nothing but the carried state to `retired`
    (no ordinary state is shared), so the carried law is the only state
    hand-over on the crossing.
    """
    working = UserRegime(
        transition=_next_regime_from_working,
        active=lambda age: age < 64,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            "pension_wealth": Phased(
                solve=_impute_pension_wealth,
                simulate=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
            ),
        },
        state_transitions={
            "wealth": _next_wealth,
            "aime": _next_aime,
            "pension_wealth": _evolve_pension_wealth,
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible_consumption": _consumption_leq_wealth},
        functions={"utility": _utility},
    )
    retired = UserRegime(
        transition=_next_regime_from_retired,
        active=lambda age: 64 <= age < 66,
        states={
            "pension_wealth": Phased(
                solve=_retired_imputed_pension_wealth,
                simulate=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
            ),
        },
        state_transitions={"pension_wealth": _evolve_pension_wealth},
        functions={"utility": _retired_utility},
    )
    return Model(
        regimes={"working": working, "retired": retired, "dead": _DEAD3},
        ages=AgeGrid(start=60, stop=66, step="2Y"),
        regime_id_class=_ThreeRegimeId,
    )


def test_simulate_evolves_carried_state_across_carried_only_handover() -> None:
    """The carried value is evolved, not frozen, on a carried-only crossing.

    Retirement keeps only pension wealth; entering it, the working regime's
    carried law must apply, so the carried value grows by its factor instead
    of being copied unchanged.
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


def test_solve_V_axis_order_follows_canonical_state_order() -> None:
    """V arrays are indexed by the solve states in canonical order.

    The productmap axis order is the engine's load-bearing invariant: V's
    axes are exactly the solve grid states (a carried state contributes none),
    ordered
    discrete-first then continuous in declaration order. Distinct grid sizes
    pin each axis to its state.
    """
    model = _build_pension_model(pension_as_pair=True)
    solution = _solve_pension_model(model)
    for regime_to_V in solution.values():
        if "working" in regime_to_V:
            # wealth has 10 points, aime 5; pension_wealth contributes no axis.
            assert regime_to_V["working"].shape == (10, 5)


def _pension_double(pension_wealth: float) -> float:
    return pension_wealth * 2.0


def test_additional_targets_read_carried_value() -> None:
    """`to_dataframe(additional_targets=...)` evaluates on the carried value.

    Simulate-phase consumers must see the carried state as the agent's true
    value, not the solve-phase imputation — only the decision (argmax over
    the solved policy) reads the imputed value.
    """
    regime = _build_pension_regime()
    regime = regime.replace(
        functions={**regime.functions, "pension_double": _pension_double}
    )
    model = Model(
        regimes={"working": regime, "dead": _DEAD},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=RegimeId,
    )
    params = cast("dict[str, Any]", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    result = model.simulate(
        log_level="debug",
        params=params,
        period_to_regime_to_V_arr=None,
        initial_conditions={
            "wealth": jnp.full(1, 50.0),
            "aime": jnp.full(1, 20.0),
            "pension_wealth": jnp.asarray([5.0]),
            "age": jnp.full(1, 60.0),
            "regime_id": jnp.array([RegimeId.working]),
        },
    )
    sim = (
        result.to_dataframe(additional_targets=["pension_double"])
        .query('regime_name == "working"')
        .set_index(["subject_id", "period"])
        .sort_index()
    )
    np.testing.assert_allclose(
        float(cast("float", sim.loc[(0, 0), "pension_double"])), 2.0 * 5.0
    )


def _pension_leq_four(pension_wealth: float) -> bool:
    return pension_wealth <= 4.0


def test_initial_feasibility_checks_seeded_carried_value() -> None:
    """A constraint on a carried state is checked against the seeded value.

    The solve-phase imputation may be feasible while the agent's true carried
    value violates the constraint; the initial-conditions feasibility check
    must catch that.
    """
    regime = _build_pension_regime()
    regime = regime.replace(
        constraints={**regime.constraints, "pension_cap": _pension_leq_four}
    )
    model = Model(
        regimes={"working": regime, "dead": _DEAD},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=RegimeId,
    )
    params = cast("dict[str, Any]", model.get_params_template())
    params["working"]["H"]["discount_factor"] = 0.95
    # Imputed pension is aime * 0.1 = 2.0 (feasible); the carried value 5.0
    # violates the cap and must be rejected.
    with pytest.raises(InvalidInitialConditionsError):
        model.simulate(
            log_level="debug",
            params=params,
            period_to_regime_to_V_arr=None,
            initial_conditions={
                "wealth": jnp.full(1, 50.0),
                "aime": jnp.full(1, 20.0),
                "pension_wealth": jnp.asarray([5.0]),
                "age": jnp.full(1, 60.0),
                "regime_id": jnp.array([RegimeId.working]),
            },
        )


def test_constraint_reading_next_carried_state_is_rejected_early() -> None:
    """F5: a carried state is imputed in solve, so its next value has no solve-phase
    producer -- the canonical solve slice omits the carried law of motion. A
    constraint reading `next_<carried>` would leave the solve feasibility DAG with an
    unsupplied argument and fail with a cryptic missing-argument error deep in the
    solve build. It must be rejected early at model construction, clearly naming the
    carried next-state. (Reading the CURRENT carried value stays valid -- covered by
    `test_initial_feasibility_checks_seeded_carried_value`.)
    """

    def _cap_on_next_pension(next_pension_wealth: float) -> bool:
        return next_pension_wealth >= 0.0

    working = UserRegime(
        transition=_next_regime,
        active=lambda age: age < 64,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=5),
            "pension_wealth": Phased(
                solve=_impute_pension_wealth,
                simulate=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
            ),
        },
        state_transitions={
            "wealth": _next_wealth,
            "aime": _next_aime,
            "pension_wealth": _evolve_pension_wealth,
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"cap_on_next_pension": _cap_on_next_pension},
        functions={"utility": _utility},
    )
    with pytest.raises(ModelInitializationError, match="next value of a carried state"):
        Model(
            regimes={"working": working, "dead": _DEAD},
            ages=AgeGrid(start=60, stop=64, step="2Y"),
            regime_id_class=RegimeId,
        )
