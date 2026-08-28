"""Canonical ranking of the published NNBEGM replay bank."""

import dataclasses
import functools
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.engine import NNBEGMPolicyRead, Regime
from _lcm.simulation.simulate import _replay_nnbegm_candidates
from _lcm.utils.logging import get_logger
from lcm import LinSpacedGrid
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousAction, ContinuousState
from tests.test_models import n_nbegm_toy as toy
from tests.test_models.n_nbegm_toy import RegimeId

_PARAMS = {"discount_factor": 0.95}


def _state_dependent_affine_outer_target(
    illiquid: ContinuousState,
    illiquid_investment: ContinuousAction,
) -> ContinuousState:
    """Map the outer action affinely with a slope that varies by state."""
    return illiquid + (1.0 + 0.1 * illiquid) * illiquid_investment


def _simulate_rows(
    *, wealth: np.ndarray, illiquid: np.ndarray, subject_batch_size: int = 0
):
    n_subjects = len(wealth)
    result = toy.build_model(variant="n_nbegm", n_periods=2).simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray(wealth),
            "illiquid": jnp.asarray(illiquid),
            "age": jnp.full(n_subjects, 20.0),
            "regime_id": jnp.full(n_subjects, RegimeId.alive, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=17,
        subject_batch_size=subject_batch_size,
    )
    return (
        result.to_dataframe()
        .query("regime_name == 'alive' and period == 0")
        .sort_index()
    )


def test_public_replay_ranks_every_reconstructed_candidate_by_canonical_q() -> None:
    """The pinned off-grid witness chooses the candidate its emitted pair attains."""
    row = _simulate_rows(
        wealth=np.array([1.0467]),
        illiquid=np.array([2.04]),
    ).iloc[0]

    target = float(row["illiquid"] + row["illiquid_investment"])
    consumption = float(row["consumption"])
    value = float(row["value"])

    # Keeper plus the declared OUTER_GRID are the complete represented bank.
    # Interpolated stored values choose 2.8571 on the immutable baseline, while
    # canonical Q at the already-reconstructed complete actions strictly chooses
    # 4.2857. The target identity is the structural assertion; the associated
    # interpolated inner action and attained Q receive working-dtype headroom.
    is_fp32 = jnp.asarray(0.0).dtype == jnp.float32
    np.testing.assert_allclose(
        target,
        30.0 / 7.0,
        rtol=0.0,
        atol=3e-5 if is_fp32 else 3e-10,
    )
    np.testing.assert_allclose(
        consumption,
        0.6695,
        rtol=0.0,
        atol=4e-3 if is_fp32 else 8e-4,
    )
    np.testing.assert_allclose(
        value,
        -18.1265,
        rtol=0.0,
        atol=8e-3 if is_fp32 else 1.5e-3,
    )


def test_a_state_dependent_outer_slope_is_refused_where_it_is_declared(
    monkeypatch,
) -> None:
    """A slope that varies by state has no single coefficient, so it is refused.

    N-NB-EGM recovers the outer action by dividing the declared map's
    coefficient out of the retained target. That division is exact only for a
    constant power-of-two coefficient; a coefficient that varies by state
    rounds, and a rounded action reaches a stock away from the node the solve
    ranked. The refusal names the map and how to declare one that inverts,
    rather than publishing a policy whose replay silently disagrees.
    """
    monkeypatch.setattr(toy, "new_illiquid", _state_dependent_affine_outer_target)
    model = toy.build_model(variant="n_nbegm", n_periods=2)

    with pytest.raises(RegimeInitializationError) as refusal:
        model.solve(params=_PARAMS, log_level="debug", return_simulation_policy=True)

    message = str(refusal.value)
    assert "illiquid_investment" in message
    assert "affine" in message


def _constant_surfaces(values: list[float]) -> jnp.ndarray:
    working_dtype = jnp.asarray(0.0).dtype
    return jnp.repeat(
        jnp.asarray(values, dtype=working_dtype)[:, None], repeats=2, axis=1
    )


@functools.cache
def _synthetic_regime_template() -> Regime:
    model = toy.build_model(variant="n_nbegm", n_periods=2)
    return model._regimes["alive"]


def _synthetic_replay(
    *,
    inner: list[float],
    outer: list[float],
    marker: list[float],
    q_and_f,
    state: jnp.ndarray | None = None,
    discrete_codes: list[list[int]] | None = None,
) -> tuple[MappingProxyType, jnp.ndarray]:
    """Run the real replay reduction on constant synthetic candidate surfaces."""
    if state is None:
        state = jnp.array([0.37])
    discrete_names = ("choice",) if discrete_codes is not None else ()
    policy = NNBEGMSimPolicy(
        candidate_inner_action=_constant_surfaces(inner),
        candidate_outer_target=_constant_surfaces(outer),
        candidate_value=_constant_surfaces(marker),
        outer_grid_values=jnp.asarray(outer, dtype=jnp.asarray(0.0).dtype),
        candidate_discrete_actions=(
            None
            if discrete_codes is None
            else jnp.asarray(discrete_codes, dtype=jnp.int32)
        ),
        state_names=("state",),
        inner_action_name="inner",
        outer_action_name="outer",
        n_keeper_candidates=0,
        discrete_action_names=discrete_names,
    )

    def outer_target(*, outer):
        return {"outer_target": outer}

    template = _synthetic_regime_template()
    regime = dataclasses.replace(
        template,
        simulation=dataclasses.replace(
            template.simulation,
            grids=MappingProxyType(
                {
                    "state": LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
                    # Declared wide enough to contain every outer target these
                    # stubs use as a candidate marker: replay drops a candidate
                    # whose recovered stock leaves the outer state's domain, so
                    # a fixture must declare a domain its own targets fit in.
                    "outer_state": LinSpacedGrid(start=0.0, stop=200.0, n_points=2),
                }
            ),
            Q_and_F=MappingProxyType({0: q_and_f}),
            egm_policy_read=NNBEGMPolicyRead(
                outer_target_function_by_period=MappingProxyType({0: outer_target}),
                outer_post_decision="outer_target",
                outer_no_adjustment_target=None,
                outer_state_name="outer_state",
            ),
        ),
    )
    states = {"state": jnp.asarray(state)}
    return _replay_nnbegm_candidates(
        optimal_actions=MappingProxyType({}),
        regime=regime,
        sim_policy=policy,
        states={**states, "outer_state": jnp.zeros_like(jnp.asarray(state))},
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.int32(20),
        canonical_states=states,
        action_names=("inner", "outer", *discrete_names),
        next_regime_to_V_arr=MappingProxyType({}),
        logger=get_logger(log_level="off"),
    )


def test_canonical_masks_exclude_unrepresented_infeasible_and_nonfinite_q() -> None:
    """Neither representation gaps nor invalid canonical scores can win."""

    def q_and_f(*, inner, outer, state, next_regime_to_V_arr, period, age):
        del outer, state, next_regime_to_V_arr, period, age
        value = jnp.select(
            [
                inner == 0,
                inner == 1,
                inner == 2,
                inner == 3,
                inner == 4,
                inner == 6,
            ],
            [1.0, jnp.nan, jnp.inf, -jnp.inf, 100.0, 200.0],
            default=2.0,
        )
        return value, inner != 4

    actions, value = _synthetic_replay(
        inner=[0, 1, 2, 3, 4, 5, 6],
        outer=[100, 101, 102, 103, 104, 105, 106],
        marker=[0, 0, 0, 0, 0, 0, np.nan],
        q_and_f=q_and_f,
    )

    np.testing.assert_array_equal(np.asarray(actions["inner"]), [5.0])
    np.testing.assert_array_equal(np.asarray(actions["outer"]), [105.0])
    np.testing.assert_array_equal(np.asarray(value), [2.0])


def test_strict_winner_is_permutation_invariant_and_exact_ties_are_first() -> None:
    """Strict rankings follow economics; exact ties follow supplied row order."""

    def strict_q(*, inner, outer, state, next_regime_to_V_arr, period, age):
        del inner, state, next_regime_to_V_arr, period, age
        return outer, jnp.ones_like(outer, dtype=bool)

    base_actions, base_value = _synthetic_replay(
        inner=[10, 20, 30],
        outer=[1, 4, 2],
        marker=[99, 98, 97],
        q_and_f=strict_q,
    )
    permuted_actions, permuted_value = _synthetic_replay(
        inner=[30, 10, 20],
        outer=[2, 1, 4],
        marker=[97, 99, 98],
        q_and_f=strict_q,
    )
    np.testing.assert_array_equal(np.asarray(base_actions["inner"]), [20.0])
    np.testing.assert_array_equal(np.asarray(permuted_actions["inner"]), [20.0])
    np.testing.assert_array_equal(np.asarray(base_value), [4.0])
    np.testing.assert_array_equal(np.asarray(permuted_value), [4.0])

    def tied_q(*, inner, outer, state, next_regime_to_V_arr, period, age):
        del inner, state, next_regime_to_V_arr, period, age
        return jnp.zeros_like(outer), jnp.ones_like(outer, dtype=bool)

    tied_actions, tied_value = _synthetic_replay(
        inner=[11, 22],
        outer=[3, 4],
        marker=[-10, 10],
        q_and_f=tied_q,
    )
    np.testing.assert_array_equal(np.asarray(tied_actions["inner"]), [11.0])
    np.testing.assert_array_equal(np.asarray(tied_actions["outer"]), [3.0])
    np.testing.assert_array_equal(np.asarray(tied_value), [0.0])


def test_discrete_times_outer_bank_is_scored_as_complete_exact_tuples() -> None:
    """Exact discrete codes participate in Q before the candidate reduction."""

    def q_and_f(*, inner, outer, choice, state, next_regime_to_V_arr, period, age):
        del inner, state, next_regime_to_V_arr, period, age
        return 10.0 * outer + choice, jnp.ones_like(outer, dtype=bool)

    actions, value = _synthetic_replay(
        inner=[1, 2, 3, 4],
        outer=[0, 0, 1, 1],
        marker=[100, 90, 80, 70],
        discrete_codes=[[0], [1], [0], [1]],
        q_and_f=q_and_f,
    )
    np.testing.assert_array_equal(np.asarray(actions["inner"]), [4.0])
    np.testing.assert_array_equal(np.asarray(actions["outer"]), [1.0])
    np.testing.assert_array_equal(np.asarray(actions["choice"]), [1])
    np.testing.assert_array_equal(np.asarray(value), [11.0])

    def tied_q(*, inner, outer, choice, state, next_regime_to_V_arr, period, age):
        del inner, choice, state, next_regime_to_V_arr, period, age
        return jnp.zeros_like(outer), jnp.ones_like(outer, dtype=bool)

    tied_actions, _ = _synthetic_replay(
        inner=[1, 2, 3, 4],
        outer=[0, 0, 1, 1],
        marker=[-1, 5, 6, 7],
        discrete_codes=[[0], [1], [0], [1]],
        q_and_f=tied_q,
    )
    np.testing.assert_array_equal(np.asarray(tied_actions["inner"]), [1.0])
    np.testing.assert_array_equal(np.asarray(tied_actions["choice"]), [0])


def test_empty_valid_set_and_out_of_support_fail_closed() -> None:
    """No foreign fallback replaces the existing NaN/-1/-inf sentinels."""

    def q_and_f(*, inner, outer, choice, state, next_regime_to_V_arr, period, age):
        del outer, choice, state, next_regime_to_V_arr, period, age
        value = jnp.where(inner == 1, jnp.nan, jnp.inf)
        return value, jnp.ones_like(inner, dtype=bool)

    actions, value = _synthetic_replay(
        inner=[1, 2],
        outer=[3, 4],
        marker=[0, 0],
        discrete_codes=[[0], [1]],
        q_and_f=q_and_f,
        state=jnp.array([0.5, 1.5]),
    )
    assert np.all(np.isnan(np.asarray(actions["inner"])))
    assert np.all(np.isnan(np.asarray(actions["outer"])))
    np.testing.assert_array_equal(np.asarray(actions["choice"]), [-1, -1])
    np.testing.assert_array_equal(np.asarray(value), [-np.inf, -np.inf])


def _literal_bracket(points: np.ndarray, query: float) -> tuple[int, float]:
    """Return the enclosing cell and literal linear weight for an in-grid query."""
    assert points[0] <= query <= points[-1]
    left = int(np.searchsorted(points, query, side="right") - 1)
    left = min(max(left, 0), len(points) - 2)
    weight = (query - points[left]) / (points[left + 1] - points[left])
    return left, float(weight)


def _literal_bilinear(
    surface: np.ndarray,
    *,
    wealth: float | np.ndarray,
    illiquid: float,
    wealth_grid: np.ndarray,
    illiquid_grid: np.ndarray,
) -> float | np.ndarray:
    """Independent scalar/vector bilinear interpolation with no production calls."""
    wealth_arr = np.asarray(wealth, dtype=float)
    flat = wealth_arr.reshape(-1)
    iy, ty = _literal_bracket(illiquid_grid, illiquid)
    assert np.all((wealth_grid[0] <= flat) & (flat <= wealth_grid[-1]))
    ix = np.searchsorted(wealth_grid, flat, side="right") - 1
    ix = np.clip(ix, 0, len(wealth_grid) - 2)
    tx = (flat - wealth_grid[ix]) / (wealth_grid[ix + 1] - wealth_grid[ix])
    out = (
        (1.0 - tx) * (1.0 - ty) * surface[ix, iy]
        + tx * (1.0 - ty) * surface[ix + 1, iy]
        + (1.0 - tx) * ty * surface[ix, iy + 1]
        + tx * ty * surface[ix + 1, iy + 1]
    )
    reshaped = out.reshape(wealth_arr.shape)
    return float(reshaped) if reshaped.ndim == 0 else reshaped


@functools.cache
def _public_bank_data():
    """Solve once and expose immutable arrays as data to the independent oracle."""
    model = toy.build_model(variant="n_nbegm", n_periods=2)
    values, policies = model.solve(
        params=_PARAMS,
        log_level="debug",
        return_simulation_policy=True,
    )
    policy = policies[0]["alive"]
    assert isinstance(policy, NNBEGMSimPolicy)
    assert policy.state_names == ("wealth", "illiquid")
    return (
        np.asarray(policy.candidate_inner_action),
        np.asarray(policy.candidate_outer_target),
        np.asarray(policy.candidate_value),
        np.asarray(values[1]["dead"]),
        np.asarray(toy.WEALTH_GRID.to_jax()),
        np.asarray(toy.ILLIQUID_GRID.to_jax()),
    )


def _scalar_bank_oracle(*, wealth: float, illiquid: float) -> dict[str, float | int]:
    """Literal candidate reconstruction, toy equations, masks, and strict max."""
    inner_bank, target_bank, marker_bank, terminal_v, wealth_grid, illiquid_grid = (
        _public_bank_data()
    )
    best: dict[str, float | int] | None = None
    for index in range(inner_bank.shape[0]):
        consumption = cast(
            "float",
            _literal_bilinear(
                inner_bank[index],
                wealth=wealth,
                illiquid=illiquid,
                wealth_grid=wealth_grid,
                illiquid_grid=illiquid_grid,
            ),
        )
        target = cast(
            "float",
            _literal_bilinear(
                target_bank[index],
                wealth=wealth,
                illiquid=illiquid,
                wealth_grid=wealth_grid,
                illiquid_grid=illiquid_grid,
            ),
        )
        investment = target - illiquid
        marker = cast(
            "float",
            _literal_bilinear(
                marker_bank[index],
                wealth=wealth,
                illiquid=illiquid,
                wealth_grid=wealth_grid,
                illiquid_grid=illiquid_grid,
            ),
        )
        resources = wealth + toy.LABOUR_INCOME - investment
        savings = resources - consumption
        if not (
            np.isfinite(consumption)
            and np.isfinite(investment)
            and np.isfinite(marker)
            and consumption > 0.0
            and savings >= toy.SAVINGS_FLOOR
            and wealth_grid[0]
            <= (next_wealth := (1.0 + toy.LIQUID_RATE) * savings)
            <= wealth_grid[-1]
            and illiquid_grid[0] <= target <= illiquid_grid[-1]
        ):
            continue
        continuation = cast(
            "float",
            _literal_bilinear(
                terminal_v,
                wealth=next_wealth,
                illiquid=target,
                wealth_grid=wealth_grid,
                illiquid_grid=illiquid_grid,
            ),
        )
        q = -1.0 / consumption + _PARAMS["discount_factor"] * continuation
        if not np.isfinite(q):
            continue
        candidate: dict[str, float | int] = {
            "index": index,
            "consumption": consumption,
            "investment": investment,
            "target": target,
            "value": q,
        }
        if best is None or q > float(best["value"]):
            best = candidate
    assert best is not None
    return best


def _dense_conditional_oracle(*, wealth: float, illiquid: float) -> dict[str, float]:
    """Dense feasible-consumption search conditional on each outer target."""
    _, _, _, terminal_v, wealth_grid, illiquid_grid = _public_bank_data()
    best: dict[str, float] | None = None
    targets = [illiquid, *np.asarray(toy.OUTER_GRID.to_jax(), dtype=float)]
    for target in targets:
        investment = target - illiquid
        resources = wealth + toy.LABOUR_INCOME - investment
        if resources <= 0.0:
            continue
        consumptions = np.linspace(1e-6, resources, 20_001)
        next_wealth = (1.0 + toy.LIQUID_RATE) * (resources - consumptions)
        inside = (next_wealth >= wealth_grid[0]) & (next_wealth <= wealth_grid[-1])
        if not np.any(inside):
            continue
        feasible_consumption = consumptions[inside]
        continuation = _literal_bilinear(
            terminal_v,
            wealth=next_wealth[inside],
            illiquid=float(target),
            wealth_grid=wealth_grid,
            illiquid_grid=illiquid_grid,
        )
        q = -1.0 / feasible_consumption + _PARAMS["discount_factor"] * continuation
        position = int(np.argmax(q))
        candidate = {
            "target": float(target),
            "consumption": float(feasible_consumption[position]),
            "value": float(q[position]),
        }
        if best is None or candidate["value"] > best["value"]:
            best = candidate
    assert best is not None
    return best


def test_scalar_bank_oracle_matches_public_replay_on_off_grid_mutations() -> None:
    """Independent scalar candidate reconstruction agrees on identity/actions/Q."""
    wealth = np.array([1.0467, 0.83, 1.37, 2.21, 3.42, 4.73])
    illiquid = np.array([2.04, 1.31, 2.63, 3.79, 5.27, 7.41])
    rows = _simulate_rows(wealth=wealth, illiquid=illiquid)

    for row, current_wealth, current_illiquid in zip(
        rows.itertuples(), wealth, illiquid, strict=True
    ):
        oracle = _scalar_bank_oracle(
            wealth=float(current_wealth), illiquid=float(current_illiquid)
        )
        target = float(row.illiquid + row.illiquid_investment)
        is_fp32 = jnp.asarray(0.0).dtype == jnp.float32
        atol = 5e-5 if is_fp32 else 5e-10
        np.testing.assert_allclose(target, oracle["target"], rtol=0.0, atol=atol)
        np.testing.assert_allclose(
            row.consumption,
            oracle["consumption"],
            rtol=0.0,
            atol=2e-4 if is_fp32 else 2e-9,
        )
        np.testing.assert_allclose(
            row.value,
            oracle["value"],
            rtol=0.0,
            atol=2e-3 if is_fp32 else 2e-8,
        )

    dense = _dense_conditional_oracle(wealth=1.0467, illiquid=2.04)
    np.testing.assert_allclose(dense["target"], 30.0 / 7.0, rtol=0.0, atol=3e-7)
    np.testing.assert_allclose(dense["consumption"], 0.7768, rtol=0.0, atol=2e-3)
    np.testing.assert_allclose(dense["value"], -18.0981, rtol=0.0, atol=2e-3)


def test_candidate_ranking_is_invariant_to_subject_batching() -> None:
    """Chunking subjects cannot change candidate identity, actions, or attained Q."""
    wealth = np.array([1.0467, 0.83, 1.37, 2.21, 3.42, 4.73, 7.11])
    illiquid = np.array([2.04, 1.31, 2.63, 3.79, 5.27, 7.41, 10.13])
    whole = _simulate_rows(wealth=wealth, illiquid=illiquid, subject_batch_size=0)
    chunked = _simulate_rows(wealth=wealth, illiquid=illiquid, subject_batch_size=3)

    whole_target = (
        whole["illiquid"].to_numpy() + whole["illiquid_investment"].to_numpy()
    )
    chunked_target = (
        chunked["illiquid"].to_numpy() + chunked["illiquid_investment"].to_numpy()
    )
    np.testing.assert_array_equal(whole_target, chunked_target)
    for column in ("illiquid_investment", "consumption", "value"):
        np.testing.assert_array_equal(
            whole[column].to_numpy(),
            chunked[column].to_numpy(),
            err_msg=column,
        )


def test_a_bank_with_every_candidate_dropped_emits_no_winner() -> None:
    """When the solve dropped every candidate, replay publishes no action.

    A dropped candidate carries `nan` in the published value surface. With the
    whole bank dropped, every objective is masked to `-inf` and `argmax` over
    them returns index 0 by convention -- so a reduction that trusted the argmax
    would emit candidate zero as the winner, an action the solve never
    represented. The fail-closed sentinels must survive instead.

    This is a different input path from an all-invalid objective: there the
    candidates are represented and score badly, here they were never
    represented at all.
    """

    def q_and_f(*, inner, outer, state, next_regime_to_V_arr, period, age):
        del outer, state, next_regime_to_V_arr, period, age
        return jnp.ones_like(inner), jnp.ones_like(inner, dtype=bool)

    actions, value = _synthetic_replay(
        inner=[1.0, 2.0, 3.0],
        outer=[10.0, 20.0, 30.0],
        marker=[np.nan, np.nan, np.nan],
        q_and_f=q_and_f,
    )

    assert np.all(np.isnan(np.asarray(actions["inner"])))
    assert np.all(np.isnan(np.asarray(actions["outer"])))
    np.testing.assert_array_equal(np.asarray(value), [-np.inf])
