"""Regression: a `SamePeriodRef` reading a NON-FOLDED process-state V (E2 x IID).

A process state (`NormalIIDProcess`, `fold=False`) is classified
`topology="discrete"` for the ordinary Markov-chain solve path
(`_lcm.variables._raw_variable_info`), so its V-array axis is read via
integer fancy-indexing there — correct, since the solve-side continuation
always feeds it an exact on-grid node index (`Q_and_F.py`'s `next_V_interpolator`
machinery, unaffected by this fix).

A `SamePeriodRef` projection is different: it computes a genuine VALUE for
every reference-regime state (`lcm.regime.SamePeriodRef.projection`), which
for a process axis is essentially never an exact node index. Before this fix,
`_build_same_period_ref_reader` (`_lcm.regime_building.Q_and_F`) fed that
float straight into the same integer-lookup path used for an ordinary
discrete axis, raising ``ValueError: Indexer must have integer or boolean
type`` at `_lcm.regime_building.V._get_lookup_function` (line ~191,
`kwargs[array_name][positions]`).

The fix: `get_V_interpolator` (`_lcm.regime_building.V`) gains a
process-aware mode (`interpolate_process_axes=True`), auto-selected by
`_build_same_period_ref_reader` whenever the reference regime carries a
process axis, which interpolates the WHOLE array (including any genuine
discrete axes, exactly, at integer-valued coordinates) through one
`jax.scipy.ndimage`-style `map_coordinates` call, using
`_ContinuousStochasticProcess.get_coordinate` (clamped to the node range) for
the process axis.

This test builds a collective ("married") regime whose value-constraint
reads a singleton reference regime's ("shock_ref") V through a
`SamePeriodRef` projected onto an OFF-GRID (strictly between two
quadrature nodes) shock value, and checks:

(a) the model SOLVES (no `ValueError`), both directly (`solve`) and through
    forward simulation (`simulate`) — the simulate-side value router
    (`_lcm.simulation.simulate.simulate`'s `argmax_and_max_Q_over_a`) shares
    the exact same compiled `same_period_refs` reader the solve-side
    `Q_and_F` kernel uses, so this exercises the identical code path.
(b) the read reference value equals a hand-computed linear interpolation of
    `V_ref` (`shock_ref`'s own, independently solved period-0 V) along the
    shock's node axis — verified indirectly-but-precisely: `married`'s value
    constraint is `|Q_f - V_shock_ref| < 1e-6`, and `Q_f` is engineered to
    equal the hand-computed number exactly, so a wrong interpolation (wrong
    neighbouring nodes, wrong weight, or no interpolation at all) would make
    every action infeasible (`D=True`, `V=-inf`) instead of the asserted
    exact match.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.simulation.simulate import simulate
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    NormalIIDProcess,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.regime import Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, DiscreteAction, FloatND, ScalarInt


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class RegimeId:
    shock_ref: ScalarInt
    shock_ref_terminal: ScalarInt
    married: ScalarInt
    married_terminal: ScalarInt


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


# Nodes: linspace(mu - n_std*sigma, mu + n_std*sigma, n_points) = [-2, -1, 0, 1, 2].
_SHOCK = NormalIIDProcess(n_points=5, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2.0)
_WAGE = LinSpacedGrid(start=0.0, stop=2.0, n_points=2)  # {0.0, 2.0}
_AGES = AgeGrid(start=0, stop=2, step="Y")
_REGIME_NAMES_TO_IDS = MappingProxyType(
    {
        "shock_ref": jnp.int32(0),
        "shock_ref_terminal": jnp.int32(1),
        "married": jnp.int32(2),
        "married_terminal": jnp.int32(3),
    }
)


def _utility_shock_ref(shock: FloatND, work: DiscreteAction) -> FloatND:
    return work * (10.0 + shock)


def _project_shock(wage: FloatND) -> FloatND:
    """Map `married`'s own wage grid onto an OFF-GRID shock coordinate.

    wage=0.0 -> -0.7 (strictly between nodes -1 and 0);
    wage=2.0 ->  0.3 (strictly between nodes  0 and 1).
    """
    return wage / 2.0 - 0.7


def _make_shock_ref_regimes() -> dict[str, Regime]:
    shock_ref = Regime(
        transition={"shock_ref_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"shock": _SHOCK},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_shock_ref},
    )
    shock_ref_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: 0.0},
    )
    return {"shock_ref": shock_ref, "shock_ref_terminal": shock_ref_terminal}


def _solve_shock_ref_only() -> tuple[np.ndarray, np.ndarray]:
    """Solve `shock_ref` in isolation: its V never depends on `married`.

    Returns `(nodes, V_ref)` for period 0 — used to hand-compute the exact
    off-grid interpolated value `married`'s constraint must reproduce.
    """
    regime_names_to_ids = MappingProxyType(
        {"shock_ref": jnp.int32(0), "shock_ref_terminal": jnp.int32(1)}
    )
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_shock_ref_regimes(), derived_categoricals={}
        ),
        ages=_AGES,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=False,
    )
    flat_params = MappingProxyType(
        {
            "shock_ref": MappingProxyType({"H__discount_factor": jnp.asarray(0.95)}),
            "shock_ref_terminal": MappingProxyType({}),
        }
    )
    solution, _sim_policies, _dissolution = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    nodes = np.asarray(_SHOCK.get_gridpoints())
    V_ref = np.asarray(solution[0]["shock_ref"])
    return nodes, V_ref


def _hand_interpolate(nodes: np.ndarray, V_ref: np.ndarray, value: float) -> float:
    """Independent linear-interpolation formula (ascending `nodes`)."""
    idx_upper = int(
        np.clip(np.searchsorted(nodes, value, side="right"), 1, len(nodes) - 1)
    )
    idx_lower = idx_upper - 1
    weight_upper = (value - nodes[idx_lower]) / (nodes[idx_upper] - nodes[idx_lower])
    return float(
        V_ref[idx_lower] * (1.0 - weight_upper) + V_ref[idx_upper] * weight_upper
    )


def _target_values() -> np.ndarray:
    """Hand-computed `V_shock_ref` at each wage cell's projected shock value."""
    nodes, V_ref = _solve_shock_ref_only()
    shock_values = np.asarray([float(_project_shock(w)) for w in [0.0, 2.0]])
    return np.array([_hand_interpolate(nodes, V_ref, s) for s in shock_values])


_TARGET = _target_values()


def _utility_married_f(wage: FloatND, work: DiscreteAction) -> FloatND:
    target = jnp.where(wage < 1.0, _TARGET[0], _TARGET[1])
    return work * target


def _utility_married_m(wage: FloatND, work: DiscreteAction) -> FloatND:
    return 1.0 * work + 0.0 * wage


def _utility_married_terminal(wage: FloatND, work: DiscreteAction) -> FloatND:
    return 0.0 * wage * work


def _vc_f(Q_f: FloatND, V_shock_ref: FloatND) -> BoolND:
    """Feasible only when `Q_f` matches the interpolated reference EXACTLY.

    A wrong interpolation (wrong node pair, wrong weight, or the pre-fix
    integer-lookup crash) would make `Q_f` (engineered off the hand-computed
    value) fail this tight tolerance for every action, emptying the mask
    (`D=True`, `V=-inf`) instead of the exact match asserted below.
    """
    return jnp.abs(Q_f - V_shock_ref) < 1e-6


def _make_regimes() -> dict[str, Regime]:
    regimes = _make_shock_ref_regimes()
    married = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_married_f, "utility_m": _utility_married_m},
        value_constraints={"vc_f": _vc_f},
        same_period_refs={
            "V_shock_ref": SamePeriodRef(
                regime="shock_ref", projection={"shock": _project_shock}
            ),
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _utility_married_terminal,
            "utility_m": _utility_married_terminal,
        },
    )
    regimes["married"] = married
    regimes["married_terminal"] = married_terminal
    return regimes


def _flat_params() -> MappingProxyType:
    return MappingProxyType(
        {
            "shock_ref": MappingProxyType({"H__discount_factor": jnp.asarray(0.95)}),
            "shock_ref_terminal": MappingProxyType({}),
            "married": MappingProxyType({"H__discount_factor": jnp.asarray(0.95)}),
            "married_terminal": MappingProxyType({}),
        }
    )


def _build_and_solve():
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_regimes(), derived_categoricals={}
        ),
        ages=_AGES,
        regime_names_to_ids=_REGIME_NAMES_TO_IDS,
        enable_jit=False,
    )
    flat_params = _flat_params()
    solution, _sim_policies, dissolution_flags = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return regimes, flat_params, solution, dissolution_flags


def test_same_period_ref_solves_with_a_nonfolded_process_reference():
    """(a) The model SOLVES: pre-fix this raised `ValueError` at `V.py:191`."""
    _regimes, _flat_params, solution, dissolution_flags = _build_and_solve()

    V_married = np.asarray(solution[0]["married"])
    D_married = np.asarray(dissolution_flags[0]["married"])

    assert V_married.shape == (2, 2)
    np.testing.assert_array_equal(D_married, [False, False])


def test_same_period_ref_value_matches_hand_computed_interpolation():
    """(b) The married regime reads exactly the hand-computed interpolated V_ref.

    `_TARGET` was computed independently (via `_solve_shock_ref_only` +
    `_hand_interpolate`, a plain numpy linear interpolation over the actual
    solved `shock_ref` V array) BEFORE `married`'s utility function — which
    bakes `_TARGET` in as constants — was even defined. The value constraint
    only keeps an action feasible when `Q_f` matches the (independently,
    solve-time) interpolated `V_shock_ref` to `1e-6`; a wrong interpolation
    would leave every action infeasible.
    """
    _regimes, _flat_params, solution, dissolution_flags = _build_and_solve()

    V_married = np.asarray(solution[0]["married"])
    D_married = np.asarray(dissolution_flags[0]["married"])

    np.testing.assert_array_equal(D_married, [False, False])
    # f-component (stakeholder axis index 0) equals the hand-computed target.
    np.testing.assert_allclose(V_married[:, 0], _TARGET, atol=1e-6)
    # m-component: utility_m(work=1) = 1.0, the unique feasible action.
    np.testing.assert_allclose(V_married[:, 1], [1.0, 1.0], atol=1e-6)

    # Sanity: the off-grid shock projections really do fall strictly between
    # two quadrature nodes, and the two hand-computed targets differ (this
    # is a genuine interpolation, not a degenerate on-node lookup).
    assert -1.0 < float(_project_shock(0.0)) < 0.0
    assert 0.0 < float(_project_shock(2.0)) < 1.0
    assert not np.isclose(_TARGET[0], _TARGET[1])


def test_same_period_ref_process_interpolation_reproduced_at_simulate():
    """Simulate-side: the same reader is shared by `argmax_and_max_Q_over_a`.

    Households placed directly into `married` (bypassing any routing) at
    period 0 recompute the argmax against the stored (masked) V; the
    simulated `V_arr` must reproduce the identical hand-computed
    interpolated values the solve-side assertion above checks.
    """
    regimes, flat_params, solution, dissolution_flags = _build_and_solve()

    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([0.0, 2.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array(
                [_REGIME_NAMES_TO_IDS["married"]] * 2, dtype=jnp.int32
            ),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=_REGIME_NAMES_TO_IDS,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=_AGES,
        simulation_output_dtypes={},
        seed=0,
    )
    married_period0 = result.raw_results["married"][0]
    np.testing.assert_array_equal(np.asarray(married_period0.in_regime), [True, True])
    V_arr = np.asarray(married_period0.V_arr)
    np.testing.assert_allclose(V_arr[:, 0], _TARGET, atol=1e-6)
    np.testing.assert_allclose(V_arr[:, 1], [1.0, 1.0], atol=1e-6)
