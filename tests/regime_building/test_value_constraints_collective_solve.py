"""Integration tests for E2: value-aware feasibility + same-period refs + D (slice 3).

A collective regime may declare `value_constraints` — predicates masking actions
on the per-stakeholder action values `Q^s` and on *same-period* reference values
of OTHER regimes at projected states (design doc
`pylcm-extension-collective-regimes.md` §2 E2; EKL 2019 eq. 11):

    F(x) = { a : Q^j(x, a) >= V^j_single(pi_j(x)) - Delta_j  for j = f, m }.

The singles' period-t value is available because singles solve before the
married regime *within* period t (the within-period topological order induced
by `same_period_refs`) — no transition edges between the singleton and
collective regimes exist (they are separate regime "islands", each reaching its
own terminal). The final mask is ordinary constraints AND all value
constraints; an all-infeasible cell publishes the dissolution flag `D = True`,
distinct from a numeric `-inf` value (which occurs on-path).

Hand computation for the IR model (wage grid {1, 2, 3}, terminal utilities all
zero so every period-0 Q equals the felicity):

Married felicities (leisure L = work 0, work W = work 1):
    u_f = 3 (1 - work) + 2 wage work   ->  L: 3,   W: 2 wage
    u_m = 0.5 (1 - work) + wage work   ->  L: 0.5, W: wage

Singles' period-0 values (own work action, terminal V = 0):
    V_f(w) = 1.5 at w = 1, 3;  5.5 at w = 2      (piecewise felicity * work)
    V_m(w) = 1.0 everywhere

IR thresholds with delta_f = 0.5, delta_m = 0.2 (declared as params):
    t_f(w) = V_f(w) - 0.5 = {1.0, 5.0, 1.0},  t_m = 1.0 - 0.2 = 0.8

w=1: L Q=(3, 0.5): m fails (0.5 < 0.8).  W Q=(2, 1): both pass.
     -> household argmax over the masked set is W, V=(2, 1).  The
        UNCONSTRAINED argmax is L (O_L = 1.75 > O_W = 1.5, V=(3, 0.5)):
        the IR constraint BINDS.  D = False.
w=2: L: f fails (3 < 5).  W Q=(4, 2): f fails (4 < 5).  Mask empty ->
     D = True, V = (-inf, -inf).
w=3: L: m fails.  W Q=(6, 3): both pass -> V=(6, 3); the unconstrained
     argmax is W too (constraint slack).  D = False.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, LinSpacedGrid, Model, categorical, fixed_transition
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class IRRegimeId:
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m: ScalarInt
    single_m_terminal: ScalarInt
    married: ScalarInt
    married_terminal: ScalarInt


_WAGE_GRID = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


# --------------------------------------------------------------------------------------
# Married (collective) regime
# --------------------------------------------------------------------------------------


def _utility_married_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 3.0 * (1.0 - work) + 2.0 * wage * work


def _utility_married_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.5 * (1.0 - work) + wage * work


def _ir_f(Q_f: FloatND, V_single_f_ref: FloatND, delta_f: FloatND) -> BoolND:
    return Q_f >= V_single_f_ref - delta_f


def _ir_m(Q_m: FloatND, V_single_m_ref: FloatND, delta_m: FloatND) -> BoolND:
    return Q_m >= V_single_m_ref - delta_m


def _project_wage(wage: ContinuousState) -> ContinuousState:
    return wage


# --------------------------------------------------------------------------------------
# Singleton (single) regimes
# --------------------------------------------------------------------------------------


def _utility_single_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    # 5.5 at w = 2, 1.5 at w = 1 and w = 3 (choosing work dominates leisure = 0).
    target = jnp.where((wage > 1.5) & (wage < 2.5), 5.5, 1.5)
    return target * work


def _utility_single_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 1.0 * work + 0.0 * wage


def _utility_zero(wage: ContinuousState) -> FloatND:
    return 0.0 * wage


def _utility_zero_collective(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.0 * wage * work


def _make_ir_regimes(
    *,
    married_first: bool = False,
    with_value_constraints: bool = True,
) -> dict[str, Regime]:
    single_f = Regime(
        transition={"single_f_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_single_f},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_GRID},
        functions={"utility": _utility_zero},
    )
    single_m = single_f.replace(
        transition={"single_m_terminal": MarkovTransition(_prob_one)},
        functions={"utility": _utility_single_m},
    )
    single_m_terminal = single_f_terminal.replace()
    married = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_married_f, "utility_m": _utility_married_m},
        value_constraints=(
            {"ir_f": _ir_f, "ir_m": _ir_m} if with_value_constraints else {}
        ),
        same_period_refs=(
            {
                "V_single_f_ref": SamePeriodRef(
                    regime="single_f", projection={"wage": _project_wage}
                ),
                "V_single_m_ref": SamePeriodRef(
                    regime="single_m", projection={"wage": _project_wage}
                ),
            }
            if with_value_constraints
            else {}
        ),
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _utility_zero_collective,
            "utility_m": _utility_zero_collective,
        },
    )
    if married_first:
        return {
            "married": married,
            "married_terminal": married_terminal,
            "single_f": single_f,
            "single_f_terminal": single_f_terminal,
            "single_m": single_m,
            "single_m_terminal": single_m_terminal,
        }
    return {
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m": single_m,
        "single_m_terminal": single_m_terminal,
        "married": married,
        "married_terminal": married_terminal,
    }


_IR_REGIME_IDS = MappingProxyType(
    {
        "single_f": jnp.int32(0),
        "single_f_terminal": jnp.int32(1),
        "single_m": jnp.int32(2),
        "single_m_terminal": jnp.int32(3),
        "married": jnp.int32(4),
        "married_terminal": jnp.int32(5),
    }
)


def _flat_params_for_ir_model() -> MappingProxyType:
    return MappingProxyType(
        {
            "single_f": MappingProxyType({"H__discount_factor": jnp.asarray(0.95)}),
            "single_f_terminal": MappingProxyType({}),
            "single_m": MappingProxyType({"H__discount_factor": jnp.asarray(0.95)}),
            "single_m_terminal": MappingProxyType({}),
            "married": MappingProxyType(
                {
                    "H__discount_factor": jnp.asarray(0.95),
                    "ir_f__delta_f": jnp.asarray(0.5),
                    "ir_m__delta_m": jnp.asarray(0.2),
                }
            ),
            "married_terminal": MappingProxyType({}),
        }
    )


def _solve_ir_model(
    *, married_first: bool = False, with_value_constraints: bool = True
):
    ages = AgeGrid(start=0, stop=2, step="Y")
    flat_params = dict(_flat_params_for_ir_model())
    if not with_value_constraints:
        flat_params["married"] = MappingProxyType(
            {"H__discount_factor": jnp.asarray(0.95)}
        )
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_ir_regimes(
                married_first=married_first,
                with_value_constraints=with_value_constraints,
            ),
            derived_categoricals={},
        ),
        ages=ages,
        regime_names_to_ids=_IR_REGIME_IDS,
        enable_jit=False,
    )
    return solve(
        flat_params=MappingProxyType(flat_params),
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )


_EXPECTED_V_SINGLE_F = np.array([1.5, 5.5, 1.5])
_EXPECTED_V_SINGLE_M = np.array([1.0, 1.0, 1.0])
_EXPECTED_V_MARRIED = np.array([[2.0, 1.0], [-np.inf, -np.inf], [6.0, 3.0]])
_EXPECTED_D_MARRIED = np.array([False, True, False])
_EXPECTED_V_MARRIED_UNCONSTRAINED = np.array([[3.0, 0.5], [4.0, 2.0], [6.0, 3.0]])


def test_ir_constraints_mask_actions_and_publish_dissolution_flag():
    """EKL eq. 11 end-to-end: binding IR, empty-mask dissolution, slack cells."""
    solution, _sim_policies, dissolution_flags = _solve_ir_model()

    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]), _EXPECTED_V_SINGLE_F, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_m"]), _EXPECTED_V_SINGLE_M, rtol=1e-6
    )

    V_married = np.asarray(solution[0]["married"])
    assert V_married.shape == (3, 2)
    np.testing.assert_allclose(V_married, _EXPECTED_V_MARRIED, rtol=1e-6)

    D_married = np.asarray(dissolution_flags[0]["married"])
    assert D_married.dtype == np.bool_
    assert D_married.shape == (3,)
    np.testing.assert_array_equal(D_married, _EXPECTED_D_MARRIED)

    # Singleton regimes publish no dissolution flag.
    assert "single_f" not in dissolution_flags[0]
    assert "single_m" not in dissolution_flags[0]


def test_ir_constraint_binds_relative_to_unconstrained_solve():
    """The masked household argmax differs from the unconstrained one at w=1."""
    constrained, _, _ = _solve_ir_model()
    unconstrained, _, unconstrained_dissolution = _solve_ir_model(
        with_value_constraints=False
    )

    V_c = np.asarray(constrained[0]["married"])
    V_u = np.asarray(unconstrained[0]["married"])
    np.testing.assert_allclose(V_u, _EXPECTED_V_MARRIED_UNCONSTRAINED, rtol=1e-6)
    # Binding at w=1: values change; slack at w=3: identical.
    assert not np.allclose(V_c[0], V_u[0])
    np.testing.assert_allclose(V_c[2], V_u[2], rtol=1e-6)
    # Without value constraints no cell is empty.
    np.testing.assert_array_equal(
        np.asarray(unconstrained_dissolution[0]["married"]), np.array([False] * 3)
    )


def test_within_period_topological_order_overrides_dict_order():
    """Married declared FIRST in the regimes dict still reads the singles' V."""
    solution, _, dissolution_flags = _solve_ir_model(married_first=True)
    np.testing.assert_allclose(
        np.asarray(solution[0]["married"]), _EXPECTED_V_MARRIED, rtol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(dissolution_flags[0]["married"]), _EXPECTED_D_MARRIED
    )


def test_ir_model_via_public_model_api():
    """The same model through public `Model(...)` + `solve()` (V values only)."""
    ages = AgeGrid(start=0, stop=2, step="Y")
    model = Model(
        regimes=_make_ir_regimes(),
        ages=ages,
        regime_id_class=IRRegimeId,
    )
    solution = model.solve(
        params={"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2},
        log_level="off",
    )
    np.testing.assert_allclose(
        np.asarray(solution[0]["married"]), _EXPECTED_V_MARRIED, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]), _EXPECTED_V_SINGLE_F, rtol=1e-6
    )


# --------------------------------------------------------------------------------------
# Genuine projection + interpolation at off-grid projected coordinates
# --------------------------------------------------------------------------------------


@categorical(ordered=False)
class ProjRegimeId:
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    married: ScalarInt
    married_terminal: ScalarInt


def test_projection_maps_states_and_reference_v_is_interpolated_off_grid():
    """A non-identity projection lands between the reference regime's grid points.

    single_f: wage grid {1, 3}, V_f(w) = 2w - 1 -> {1, 5}.
    married:  wage grid {1, 2}, projection pi(w) = 4 - w -> {3, 2}; the read at
    projected wage 2 must LINEARLY interpolate V_f to 3.0 (a nearest-neighbour
    or identity-projection bug lands elsewhere and flips the assertions).

    Married: u_f = 2 wage work, u_m = wage work; ir_f only, delta_f = 0.
    w=1: pi=3 -> t_f = 5. L Q_f=0 < 5, W Q_f=2 < 5 -> D=True.
        (an identity-projection bug gives t_f = V_f(1) = 1 -> W feasible)
    w=2: pi=2 -> t_f = 3 (midpoint of {1, 5}). W Q_f=4 >= 3 -> V=(4, 2).
        (upper-nearest-neighbour gives t_f = 5 -> D=True)
    """

    def _utility_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
        return 2.0 * wage * work

    def _utility_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
        return wage * work

    def _utility_single(wage: ContinuousState, work: DiscreteAction) -> FloatND:
        return (2.0 * wage - 1.0) * work

    def _ir(Q_f: FloatND, V_f_ref: FloatND, delta_f: FloatND) -> BoolND:
        return Q_f >= V_f_ref - delta_f

    def _project(wage: ContinuousState) -> ContinuousState:
        return 4.0 - wage

    single_grid = LinSpacedGrid(start=1.0, stop=3.0, n_points=2)
    married_grid = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)

    single_f = Regime(
        transition={"single_f_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage": single_grid},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_single},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": single_grid},
        functions={"utility": _utility_zero},
    )
    married = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": married_grid},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
        value_constraints={"ir_f": _ir},
        same_period_refs={
            "V_f_ref": SamePeriodRef(regime="single_f", projection={"wage": _project})
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": married_grid},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _utility_zero_collective,
            "utility_m": _utility_zero_collective,
        },
    )
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes={
                "single_f": single_f,
                "single_f_terminal": single_f_terminal,
                "married": married,
                "married_terminal": married_terminal,
            },
            derived_categoricals={},
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {
                "single_f": jnp.int32(0),
                "single_f_terminal": jnp.int32(1),
                "married": jnp.int32(2),
                "married_terminal": jnp.int32(3),
            }
        ),
        enable_jit=False,
    )
    solution, _, dissolution_flags = solve(
        flat_params=MappingProxyType(
            {
                "single_f": MappingProxyType({"H__discount_factor": jnp.asarray(0.95)}),
                "single_f_terminal": MappingProxyType({}),
                "married": MappingProxyType(
                    {
                        "H__discount_factor": jnp.asarray(0.95),
                        "ir_f__delta_f": jnp.asarray(0.0),
                    }
                ),
                "married_terminal": MappingProxyType({}),
            }
        ),
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )

    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]), np.array([1.0, 5.0]), rtol=1e-6
    )
    V_married = np.asarray(solution[0]["married"])
    np.testing.assert_allclose(
        V_married, np.array([[-np.inf, -np.inf], [4.0, 2.0]]), rtol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(dissolution_flags[0]["married"]), np.array([True, False])
    )


# --------------------------------------------------------------------------------------
# On-path -inf vs the dissolution flag (no value constraints involved)
# --------------------------------------------------------------------------------------


@categorical(ordered=False)
class InfRegimeId:
    couple: ScalarInt
    couple_terminal: ScalarInt


def test_on_path_minus_inf_value_is_not_dissolution():
    """-inf utility with a NONEMPTY mask has D=False; an EMPTY mask has D=True.

    Wage grid {1, 2, 3}; the non-terminal couple has an ordinary constraint
    `wage < 2.5` (masks every action at w=3) and felicities of -inf for
    w > 1.5; the terminal couple is finite and unconstrained (a -inf terminal
    V would leak NaN through the linear continuation interpolation of the
    NEIGHBOURING cell — 0 * inf — which is the raw-V read the E3' gated edges
    replace in slice 4):

    w=1: both actions feasible, finite -> V=(2, 1) (work wins), D=False.
    w=2: both actions feasible but u = -inf -> V=(-inf, -inf), D=False.
    w=3: mask empty -> D=True (never inferred from V == -inf).
    """

    def _utility_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
        return jnp.where(wage > 1.5, -jnp.inf, 1.0 + work)

    def _utility_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
        return jnp.where(wage > 1.5, -jnp.inf, 1.0 * work)

    def _wage_ok(wage: ContinuousState) -> BoolND:
        return wage < 2.5

    couple = Regime(
        transition={"couple_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
        constraints={"wage_ok": _wage_ok},
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _utility_zero_collective,
            "utility_m": _utility_zero_collective,
        },
    )
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes={"couple": couple, "couple_terminal": couple_terminal},
            derived_categoricals={},
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {"couple": jnp.int32(0), "couple_terminal": jnp.int32(1)}
        ),
        enable_jit=False,
    )
    solution, _, dissolution_flags = solve(
        flat_params=MappingProxyType(
            {
                "couple": MappingProxyType({"H__discount_factor": jnp.asarray(0.95)}),
                "couple_terminal": MappingProxyType({}),
            }
        ),
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )

    V = np.asarray(solution[0]["couple"])
    D = np.asarray(dissolution_flags[0]["couple"])
    np.testing.assert_allclose(V[0], np.array([2.0, 1.0]), rtol=1e-6)
    assert np.all(np.isneginf(V[1]))
    np.testing.assert_array_equal(D, np.array([False, False, True]))

    # A collective TERMINAL regime publishes D too (here: nothing infeasible).
    D_terminal = np.asarray(dissolution_flags[1]["couple_terminal"])
    assert D_terminal.dtype == np.bool_
    np.testing.assert_array_equal(D_terminal, np.array([False, False, False]))


# --------------------------------------------------------------------------------------
# Scope fences and build-time validation
# --------------------------------------------------------------------------------------


def _minimal_collective_kwargs() -> dict:
    return {
        "transition": {"married_terminal": MarkovTransition(_prob_one)},
        "active": lambda age: age < 1,
        "states": {"wage": _WAGE_GRID},
        "state_transitions": {"wage": fixed_transition("wage")},
        "actions": {"work": DiscreteGrid(Work)},
    }


def test_value_constraints_on_singleton_regime_are_rejected():
    with pytest.raises(RegimeInitializationError, match="value_constraints"):
        Regime(
            **_minimal_collective_kwargs(),
            functions={"utility": _utility_single_f},
            value_constraints={"ir": lambda Q_f: Q_f >= 0.0},
        )


def test_same_period_refs_on_singleton_regime_are_rejected():
    with pytest.raises(RegimeInitializationError, match="same_period_refs"):
        Regime(
            **_minimal_collective_kwargs(),
            functions={"utility": _utility_single_f},
            same_period_refs={
                "V_ref": SamePeriodRef(
                    regime="single_f", projection={"wage": _project_wage}
                )
            },
        )


def test_same_period_refs_without_value_constraints_are_rejected():
    with pytest.raises(RegimeInitializationError, match="value_constraints"):
        Regime(
            **_minimal_collective_kwargs(),
            stakeholders=("f", "m"),
            functions={
                "utility_f": _utility_married_f,
                "utility_m": _utility_married_m,
            },
            same_period_refs={
                "V_ref": SamePeriodRef(
                    regime="single_f", projection={"wage": _project_wage}
                )
            },
        )


def test_value_constraints_on_terminal_collective_regime_are_rejected():
    with pytest.raises(NotImplementedError, match="TERMINAL"):
        Regime(
            transition=None,
            active=lambda age: age >= 1,
            stakeholders=("f", "m"),
            states={"wage": _WAGE_GRID},
            actions={"work": DiscreteGrid(Work)},
            functions={
                "utility_f": _utility_married_f,
                "utility_m": _utility_married_m,
            },
            value_constraints={"ir_f": _ir_f},
        )


def _married_with_refs(refs: dict[str, SamePeriodRef]) -> Regime:
    return Regime(
        **_minimal_collective_kwargs(),
        stakeholders=("f", "m"),
        functions={
            "utility_f": _utility_married_f,
            "utility_m": _utility_married_m,
        },
        value_constraints={"ir_f": _ir_f, "ir_m": _ir_m},
        same_period_refs=refs,
    )


def _process_ir_variant(regimes: dict[str, Regime]) -> None:
    ages = AgeGrid(start=0, stop=2, step="Y")
    process_regimes(
        user_regimes=finalize_regimes(user_regimes=regimes, derived_categoricals={}),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {name: jnp.int32(i) for i, name in enumerate(regimes)}
        ),
        enable_jit=False,
    )


def test_same_period_ref_to_unknown_regime_is_rejected():
    regimes = _make_ir_regimes()
    regimes["married"] = _married_with_refs(
        {
            "V_single_f_ref": SamePeriodRef(
                regime="no_such_regime", projection={"wage": _project_wage}
            ),
            "V_single_m_ref": SamePeriodRef(
                regime="single_m", projection={"wage": _project_wage}
            ),
        }
    )
    with pytest.raises(ModelInitializationError, match="no_such_regime"):
        _process_ir_variant(regimes)


def test_same_period_ref_cycle_is_rejected_at_build():
    """Two collective regimes reading each other's same-period V form a cycle."""

    def _vc_a(Q_f: FloatND, V_b_ref: FloatND) -> BoolND:
        return Q_f >= V_b_ref - 1000.0

    def _vc_b(Q_f: FloatND, V_a_ref: FloatND) -> BoolND:
        return Q_f >= V_a_ref - 1000.0

    def _make_couple(
        *,
        terminal_name: str,
        value_constraints: dict,
        same_period_refs: dict,
    ) -> Regime:
        return Regime(
            transition={terminal_name: MarkovTransition(_prob_one)},
            active=lambda age: age < 1,
            states={"wage": _WAGE_GRID},
            state_transitions={"wage": fixed_transition("wage")},
            actions={"work": DiscreteGrid(Work)},
            stakeholders=("f", "m"),
            functions={
                "utility_f": _utility_married_f,
                "utility_m": _utility_married_m,
            },
            value_constraints=value_constraints,
            same_period_refs=same_period_refs,
        )

    couple_a = _make_couple(
        terminal_name="terminal_a",
        value_constraints={"vc_a": _vc_a},
        same_period_refs={
            "V_b_ref": SamePeriodRef(
                regime="couple_b",
                projection={"wage": _project_wage},
                stakeholder="f",
            )
        },
    )
    couple_b = _make_couple(
        terminal_name="terminal_b",
        value_constraints={"vc_b": _vc_b},
        same_period_refs={
            "V_a_ref": SamePeriodRef(
                regime="couple_a",
                projection={"wage": _project_wage},
                stakeholder="f",
            )
        },
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _utility_zero_collective,
            "utility_m": _utility_zero_collective,
        },
    )
    with pytest.raises(ModelInitializationError, match="form a cycle"):
        _process_ir_variant(
            {
                "couple_a": couple_a,
                "terminal_a": terminal,
                "couple_b": couple_b,
                "terminal_b": terminal.replace(),
            }
        )


def test_same_period_ref_to_collective_regime_requires_stakeholder():
    """Reading a collective reference V without naming a stakeholder is rejected."""
    regimes = _make_ir_regimes()
    # Point the f-reference at the married regime itself is a cycle; use a second
    # collective island instead: reference the married_terminal... terminal refs
    # are not the issue here — the ref must name a stakeholder for ANY collective
    # target. Reuse single_m as the m-ref and misdeclare the f-ref onto a
    # collective regime without a stakeholder.
    couple_b = Regime(
        transition={"married_terminal_b": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _utility_married_f,
            "utility_m": _utility_married_m,
        },
    )
    regimes["couple_b"] = couple_b
    regimes["married_terminal_b"] = regimes["married_terminal"].replace()
    regimes["married"] = _married_with_refs(
        {
            "V_single_f_ref": SamePeriodRef(
                regime="couple_b", projection={"wage": _project_wage}
            ),
            "V_single_m_ref": SamePeriodRef(
                regime="single_m", projection={"wage": _project_wage}
            ),
        }
    )
    with pytest.raises(ModelInitializationError, match="stakeholder"):
        _process_ir_variant(regimes)


def test_same_period_ref_to_singleton_regime_rejects_stakeholder():
    regimes = _make_ir_regimes()
    regimes["married"] = _married_with_refs(
        {
            "V_single_f_ref": SamePeriodRef(
                regime="single_f",
                projection={"wage": _project_wage},
                stakeholder="f",
            ),
            "V_single_m_ref": SamePeriodRef(
                regime="single_m", projection={"wage": _project_wage}
            ),
        }
    )
    with pytest.raises(ModelInitializationError, match="stakeholder"):
        _process_ir_variant(regimes)


def test_same_period_ref_projection_must_cover_reference_states():
    regimes = _make_ir_regimes()
    regimes["married"] = _married_with_refs(
        {
            "V_single_f_ref": SamePeriodRef(regime="single_f", projection={}),
            "V_single_m_ref": SamePeriodRef(
                regime="single_m", projection={"wage": _project_wage}
            ),
        }
    )
    with pytest.raises(ModelInitializationError, match="projection"):
        _process_ir_variant(regimes)


def test_same_period_ref_requires_reference_active_in_same_periods():
    """The reference regime must be solved in every period the reader is active."""
    regimes = _make_ir_regimes()
    # single_f exits after period 0 twice as fast: shrink its active window so the
    # married regime (active in period 0) would read a V that... make married
    # active in periods 0 AND 1 instead, while singles stay period-0 only.
    regimes["married"] = regimes["married"].replace(active=lambda age: age < 2)
    regimes["married_terminal"] = regimes["married_terminal"].replace(
        active=lambda age: age >= 2
    )
    ages = AgeGrid(start=0, stop=3, step="Y")
    with pytest.raises(ModelInitializationError, match="active"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes=regimes, derived_categoricals={}
            ),
            ages=ages,
            regime_names_to_ids=_IR_REGIME_IDS,
            enable_jit=False,
        )
