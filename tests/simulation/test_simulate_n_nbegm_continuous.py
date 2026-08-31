"""End-to-end continuous-outer NNBEGM simulation through the nested read.

Simulation must preserve off-grid outer actions without snapping them to the
finite action grid, recover consumption from the conditional inner policies,
and remain deterministic without taste shocks.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import config as jax_config

from lcm import IrregSpacedGrid, LinSpacedGrid
from lcm.solvers import AdaptiveOuterMesh
from lcm.typing import ContinuousState, FloatND
from tests.test_models import n_nbegm_toy as toy

_PARAMS = {"discount_factor": 0.95}
# The settings the solve battery converges under (its 120 cells spread their
# optimum basins over the whole outer axis).
_MESH = AdaptiveOuterMesh(
    initial_grid=toy.OUTER_GRID,
    max_nodes=513,
    max_refinement_rounds=10,
    value_atol=1e-4,
    value_rtol=1e-4,
    golden_iterations=40,
)
# Subjects strictly between grid nodes on both asset axes, plus two exactly
# on-grid corners: the off-grid subjects are the case the grid argmax cannot
# represent.
_INITIAL = {
    "wealth": jnp.array([4.3, 11.7, 19.9, 27.2727272727272727, 8.1]),
    "illiquid": jnp.array([1.37, 6.6, 13.2, 8.8888888888888889, 17.5]),
    "age": jnp.full(5, 20.0),
    "regime_id": jnp.zeros(5, dtype=jnp.int32),
}
_MIXED_INITIAL = {
    "wealth": jnp.array([4.3, 11.7]),
    "illiquid": jnp.array([1.37, 6.6]),
    "age": jnp.full(2, 20.0),
    "regime_id": jnp.array(
        [toy.RegimeId.alive, toy.RegimeId.dead],
        dtype=jnp.int32,
    ),
}


def _simulate(
    *,
    seed: int,
    initial_conditions=_INITIAL,
    terminal_active_from_start: bool = False,
) -> pd.DataFrame:
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=3,
        outer_search=_MESH,
        terminal_active_from_start=terminal_active_from_start,
    )
    return model.simulate(
        params=_PARAMS,
        initial_conditions=dict(initial_conditions),
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=seed,
    ).to_dataframe()


@pytest.fixture(scope="module")
def simulated() -> pd.DataFrame:
    if not jax_config.read("jax_enable_x64"):
        pytest.skip("x64 run only")
    return _simulate(seed=42)


def test_outer_actions_leave_the_finite_action_grid(simulated: pd.DataFrame) -> None:
    """No silent grid snapping: some accepted outer actions are off-grid.

    The legacy finite path can only record nodes of the investment action
    grid; the continuous read must produce interior refined actions for
    off-grid subjects.
    """
    alive = simulated[simulated["regime_name"] == "alive"]
    actions = np.asarray(alive["illiquid_investment"], dtype=float)
    action_grid = np.asarray(toy.ILLIQUID_INVESTMENT_GRID.to_jax())
    distance = np.min(np.abs(actions[:, None] - action_grid[None, :]), axis=1)
    assert np.any(distance > 1e-6), "every outer action snapped to the grid"


def test_consumption_is_continuous_or_fallback_is_explicit(
    simulated: pd.DataFrame,
) -> None:
    """A value-controlled refusal may keep the safe finite-grid consumption."""
    alive = simulated[simulated["regime_name"] == "alive"]
    consumption = np.asarray(alive["consumption"], dtype=float)
    grid = np.asarray(toy.CONSUMPTION_GRID.to_jax())
    distance = np.min(np.abs(consumption[:, None] - grid[None, :]), axis=1)
    fallback = np.asarray(alive["nested_policy_fallback"], dtype=bool)
    assert np.any(distance > 1e-6) or np.all(fallback)


def test_recorded_pairs_respect_the_intrinsic_budget(simulated: pd.DataFrame) -> None:
    """Consumption positive and within resources at the recorded outer action."""
    alive = simulated[simulated["regime_name"] == "alive"]
    wealth = np.asarray(alive["wealth"], dtype=float)
    illiquid = np.asarray(alive["illiquid"], dtype=float)
    investment = np.asarray(alive["illiquid_investment"], dtype=float)
    consumption = np.asarray(alive["consumption"], dtype=float)
    resources = wealth + toy.LABOUR_INCOME - investment
    assert np.all(consumption > 0.0)
    assert np.all(consumption <= resources + 1e-9)
    # The chosen next durable stays inside the outer search domain.
    next_illiquid = illiquid + investment
    assert np.all(next_illiquid >= toy.OUTER_GRID.start - 1e-9)
    assert np.all(next_illiquid <= toy.OUTER_GRID.stop + 1e-9)


def test_nested_policy_fallback_column_is_published(simulated: pd.DataFrame) -> None:
    """The refused-nested-read flag must be observable.

    A regime that can take the nested continuous-outer policy read must publish
    the per-subject flag saying whether that read was refused and the grid
    argmax kept, otherwise inference on this path cannot refuse the fallback
    rows. The complementary half -- that models without a nested read do NOT
    get the column -- is asserted in `test_simulate.py`.
    """
    assert "nested_policy_fallback" in simulated.columns
    # Rows of the other regime are NaN-padded, as for any regime-specific column,
    # so only the nested-read regime's own rows carry the flag.
    alive = simulated[simulated["regime_name"] == "alive"]
    flags = alive["nested_policy_fallback"]
    assert flags.notna().all()
    assert set(np.unique(np.asarray(flags))) <= {False, True}


def test_mixed_regime_placeholders_do_not_abort_nested_replay() -> None:
    """Only the subjects currently alive govern nested-policy replay safety."""
    mixed = _simulate(
        seed=42,
        initial_conditions=_MIXED_INITIAL,
        terminal_active_from_start=True,
    )
    period0 = mixed[mixed["period"] == 0].sort_values("subject_id")

    assert list(zip(period0["subject_id"], period0["regime_name"], strict=True)) == [
        (0, "alive"),
        (1, "dead"),
    ]
    alive = period0.iloc[0]
    resources = (
        float(alive["wealth"]) + toy.LABOUR_INCOME - float(alive["illiquid_investment"])
    )
    assert 0.0 < float(alive["consumption"]) <= resources
    assert pd.notna(alive["nested_policy_fallback"])
    assert pd.isna(period0.iloc[1]["nested_policy_fallback"])


def test_simulation_is_deterministic(simulated: pd.DataFrame) -> None:
    again = _simulate(seed=42)
    for column in ("wealth", "illiquid", "consumption", "illiquid_investment"):
        np.testing.assert_array_equal(
            np.asarray(simulated[column], dtype=float),
            np.asarray(again[column], dtype=float),
        )


def _terminal_strictly_prefers_the_lower_durable(
    wealth: ContinuousState, illiquid: ContinuousState
) -> FloatND:
    """Keep a positive liquid marginal while making stock zero the strict optimum."""
    return jnp.log1p(wealth) - 100.0 * jnp.square(illiquid)


def test_a_legal_narrow_adjuster_mesh_preserves_the_exact_keeper() -> None:
    """End-to-end replay keeps stock zero outside the adjuster mesh `[2, 18]`.

    The adaptive mesh governs adjustment targets only. The keeper is solved as a
    separate exact branch over the outer state's declared `[0, 20]` domain, so a
    second mesh-domain check must not turn keeping into an adjustment to stock two.
    """
    mesh = AdaptiveOuterMesh(
        initial_grid=LinSpacedGrid(start=2.0, stop=18.0, n_points=3),
        max_nodes=9,
        max_refinement_rounds=2,
        value_atol=1e-4,
        value_rtol=1e-4,
        golden_iterations=12,
        fail_closed=False,
    )
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=mesh,
        terminal_utility_function=_terminal_strictly_prefers_the_lower_durable,
    )
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray([10.0]),
            "illiquid": jnp.asarray([0.0]),
            "age": jnp.asarray([20.0]),
            "regime_id": jnp.asarray([toy.RegimeId.alive], dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=42,
    ).to_dataframe()
    row = result.query("period == 0 and regime_name == 'alive'").iloc[0]

    assert not bool(row["nested_policy_fallback"])
    assert float(row["illiquid_investment"]) == pytest.approx(0.0, abs=0.0)
    assert float(row["illiquid"] + row["illiquid_investment"]) == pytest.approx(
        0.0, abs=0.0
    )


def _resources_ignore_the_outer_move(wealth: ContinuousState) -> FloatND:
    """Keep the wide-domain round-trip witness inside the liquid state grid."""
    return wealth + toy.LABOUR_INCOME


def _terminal_peaks_at_the_interior_half(
    wealth: ContinuousState, illiquid: ContinuousState
) -> FloatND:
    """Make the irregular grid's interior node the strict outer optimum."""
    return jnp.log1p(wealth) - 100.0 * jnp.square(illiquid - 0.5)


def test_wide_float32_interior_target_falls_back_in_public_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nominal target `0.5` may not be replayed as stock zero.

    At float32, `1e10 + (0.5 - 1e10)` is exactly zero. The old containment-only
    rule treated that as a valid replay because zero remains inside `[0, 1e10]`.
    The target-local rule rejects it and makes the fallback observable. Float64
    reaches `0.5` exactly and therefore keeps the continuous proposal.
    """
    outer_grid = IrregSpacedGrid(points=[0.0, 0.5, 1e10])
    investment_grid = IrregSpacedGrid(points=[-1e10, 0.0, 1e10])
    mesh = AdaptiveOuterMesh(
        initial_grid=outer_grid,
        max_nodes=3,
        max_refinement_rounds=0,
        value_atol=1e-4,
        value_rtol=1e-4,
        golden_iterations=4,
        fail_closed=False,
    )
    monkeypatch.setattr(toy, "resources", _resources_ignore_the_outer_move)
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        illiquid_grid=outer_grid,
        illiquid_investment_grid=investment_grid,
        outer_search=mesh,
        terminal_utility_function=_terminal_peaks_at_the_interior_half,
    )
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray([10.0]),
            "illiquid": jnp.asarray([1e10]),
            "age": jnp.asarray([20.0]),
            "regime_id": jnp.asarray([toy.RegimeId.alive], dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=42,
    ).to_dataframe()
    row = result.query("period == 0 and regime_name == 'alive'").iloc[0]
    is_float32 = jnp.asarray(0.0).dtype == jnp.float32

    assert bool(row["nested_policy_fallback"]) is is_float32
    reached = float(row["illiquid"] + row["illiquid_investment"])
    if is_float32:
        assert reached != pytest.approx(0.5, abs=0.0)
    else:
        assert reached == pytest.approx(0.5, abs=0.0)
