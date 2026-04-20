"""Test that a custom aggregation function H can be used in a model."""

from collections.abc import Callable

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class LaborSupply:
    work: int
    retire: int


@categorical(ordered=False)
class RegimeId:
    working_life: int
    dead: int


def utility(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    # Multiplicative disutility ensures positive utility for CES aggregator
    work_factor = jnp.where(is_working, 1.0 / (1.0 + disutility_of_work), 1.0)
    return consumption * work_factor


def labor_income(is_working: BoolND) -> FloatND:
    return jnp.where(is_working, 1.5, 0.0)


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborSupply.work


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
) -> ContinuousState:
    return wealth - consumption + labor_income


def next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.working_life)


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def ces_H(
    utility: float,
    E_next_V: float,
    discount_factor: float,
    ies: float,
) -> float:
    rho = 1 - ies
    return ((1 - discount_factor) * utility**rho + discount_factor * E_next_V**rho) ** (
        1 / rho
    )


START_AGE = 0
N_PERIODS = 4
FINAL_AGE_ALIVE = START_AGE + N_PERIODS - 2  # = 2


@categorical(ordered=False)
class PrefType:
    type_0: int
    type_1: int
    type_2: int


def discount_factor_from_type(
    pref_type: DiscreteState,
    discount_factor_by_type: FloatND,
) -> FloatND:
    """Index a per-type discount factor Series by the pref_type state.

    Wiring this as `functions["discount_factor"]` exercises pylcm's
    ability to resolve an H argument from a DAG function output when
    the name is not in `states_actions_params`.
    """
    return discount_factor_by_type[pref_type]


def _make_model(custom_H=None, *, with_pref_type: bool = False):
    """Create a simple model, optionally with a custom H and pref_type state.

    When `with_pref_type=True`, the working-life regime gains a
    `pref_type` discrete state (`batch_size=1`, three categories) and
    wires `discount_factor` as a DAG function that indexes
    `discount_factor_by_type` by the state. This exercises the
    "DAG output feeds H" path in pylcm's Q_and_F — and relies on
    `_validate_all_variables_used` treating H-DAG targets as reachable
    so `pref_type` counts as used without any workaround in `utility`.
    """
    functions: dict[str, Callable] = {
        "utility": utility,
        "labor_income": labor_income,
        "is_working": is_working,
    }
    if custom_H is not None:
        functions["H"] = custom_H
    if with_pref_type:
        functions["discount_factor"] = discount_factor_from_type

    working_life_states: dict = {
        "wealth": LinSpacedGrid(start=0.5, stop=10, n_points=30),
    }
    working_life_state_transitions: dict = {
        "wealth": next_wealth,
    }
    if with_pref_type:
        working_life_states["pref_type"] = DiscreteGrid(PrefType, batch_size=1)
        working_life_state_transitions["pref_type"] = None

    working_life_regime = Regime(
        actions={
            "labor_supply": DiscreteGrid(LaborSupply),
            "consumption": LinSpacedGrid(start=0.5, stop=10, n_points=50),
        },
        states=working_life_states,
        state_transitions=working_life_state_transitions,
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime,
        functions=functions,
        active=lambda age: age <= FINAL_AGE_ALIVE,
    )

    def dead_utility() -> float:
        return 0.0

    dead_regime = Regime(
        transition=None,
        functions={"utility": dead_utility},
        active=lambda age: age > FINAL_AGE_ALIVE,
    )

    return Model(
        regimes={"working_life": working_life_regime, "dead": dead_regime},
        ages=AgeGrid(start=START_AGE, stop=FINAL_AGE_ALIVE + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def test_custom_ces_aggregator_differs_from_default():
    """A CES aggregator with ies != 1 should produce different value functions."""
    model_default = _make_model()
    model_ces = _make_model(custom_H=ces_H)

    params_default = {
        "discount_factor": 0.95,
        "working_life": {
            "utility": {"disutility_of_work": 0.5},
            "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
        },
    }
    params_ces = {
        "working_life": {
            "H": {"discount_factor": 0.95, "ies": 0.5},
            "utility": {"disutility_of_work": 0.5},
            "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
        },
        "dead": {},
    }

    V_default = model_default.solve(params=params_default)
    V_ces = model_ces.solve(params=params_ces)

    # The value functions should differ because the aggregation rule differs
    has_difference = False
    for period in V_default:
        for regime_name in V_default[period]:
            if not jnp.allclose(
                V_default[period][regime_name], V_ces[period][regime_name]
            ):
                has_difference = True
                break

    assert has_difference, (
        "CES and default aggregator should produce different value functions"
    )


def test_default_H_injected_for_non_terminal():
    """The default H function should be in non-terminal Regime.functions."""
    r = Regime(
        functions={"utility": lambda: 0.0},
        transition=lambda: {"a": 1.0},
        active=lambda age: age < 1,
    )
    assert "H" in r.functions


def test_default_H_not_injected_for_terminal():
    """Terminal regimes should not have H injected."""
    r = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
    )
    assert "H" not in r.functions


def test_custom_H_not_overwritten():
    """A user-provided H should not be replaced by the default."""

    def my_H(utility: float, E_next_V: float) -> float:
        return utility + E_next_V

    r = Regime(
        transition=lambda: {"a": 1.0},
        active=lambda age: age < 1,
        functions={"utility": lambda: 0.0, "H": my_H},
    )
    assert r.functions["H"] is my_H


def test_params_template_includes_H():
    """The params template should include H's parameters."""
    model = _make_model()
    template = model._params_template
    # Default H has discount_factor parameter
    assert "H" in template["working_life"]
    assert "discount_factor" in template["working_life"]["H"]


def test_params_template_custom_H():
    """Custom H params should appear in the template."""
    model = _make_model(custom_H=ces_H)
    template = model._params_template
    assert "H" in template["working_life"]
    assert "discount_factor" in template["working_life"]["H"]
    assert "ies" in template["working_life"]["H"]


def test_terminal_regime_value_unchanged_by_H():
    """Terminal regimes don't use H, so different H should give same terminal values."""
    model_default = _make_model()
    model_ces = _make_model(custom_H=ces_H)

    params_default = {
        "discount_factor": 0.95,
        "working_life": {
            "utility": {"disutility_of_work": 0.5},
            "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
        },
    }
    params_ces = {
        "working_life": {
            "H": {"discount_factor": 0.95, "ies": 0.5},
            "utility": {"disutility_of_work": 0.5},
            "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
        },
        "dead": {},
    }

    V_default = model_default.solve(params=params_default)
    V_ces = model_ces.solve(params=params_ces)

    # Last period is terminal — value functions should be identical
    last_period = max(V_default.keys())
    assert_array_equal(
        V_default[last_period]["dead"],
        V_ces[last_period]["dead"],
    )


# ---------------------------------------------------------------------------
# DAG-output feeds H: `discount_factor` computed by a DAG function that
# indexes a per-type Series by the `pref_type` state.
# ---------------------------------------------------------------------------


def test_model_constructs_when_state_reachable_only_via_h_dag():
    """State reached only via H's DAG deps must pass the usage check.

    `pref_type` is used by `discount_factor_from_type`, whose output
    feeds the default H. `utility` / `feasibility` / transitions do
    not reference `pref_type`. Pre-fix, this failed with
    "states defined but never used"; post-fix, the state-usage walk
    treats H-DAG targets as reachable.
    """
    _make_model(with_pref_type=True)


def test_dag_output_feeds_default_h_monotone_in_discount_factor():
    """Higher per-type discount factor ⇒ higher value function.

    The working-life regime uses the default H (which expects a scalar
    `discount_factor`). That scalar is produced by a DAG function that
    indexes `discount_factor_by_type` by the `pref_type` state. This
    only works if pylcm's Q_and_F resolves H arguments from DAG
    function outputs when they are not in `states_actions_params`.
    """
    model = _make_model(with_pref_type=True)

    params = {
        "discount_factor_by_type": jnp.array([0.70, 0.85, 0.99]),
        "working_life": {
            "utility": {"disutility_of_work": 0.5},
            "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
        },
    }
    V = model.solve(params=params)

    # At the last working_life-active period (FINAL_AGE_ALIVE), the deterministic
    # transition goes to `dead` and V_dead = 0, so V = U is pref_type-independent.
    # Check earlier periods where the discount factor actually multiplies E[V_next].
    periods_with_future = [p for p in V if p < FINAL_AGE_ALIVE]
    assert periods_with_future
    for period in periods_with_future:
        # Shape is (..., n_pref_type). Compare averages across the
        # other axes so the comparison is robust to the grid layout.
        v = V[period]["working_life"]
        # pref_type is the innermost batched state ⇒ last axis.
        v_type_0 = jnp.mean(v[..., 0])
        v_type_1 = jnp.mean(v[..., 1])
        v_type_2 = jnp.mean(v[..., 2])
        assert v_type_0 < v_type_1 < v_type_2, (
            f"Expected V monotone in discount factor at period {period}; "
            f"got {v_type_0:.4f} < {v_type_1:.4f} < {v_type_2:.4f}"
        )


# H's permissive kwarg contract: H may name any argument supported by
# regime functions — states, actions, flat params, or DAG-output
# functions. The following tests lock that contract in.


def wealth_H(
    utility: float,
    E_next_V: float,
    discount_factor: float,
    wealth: float,
    wealth_weight: float,
) -> float:
    return utility + discount_factor * E_next_V + wealth_weight * wealth


def test_h_consumes_continuous_state():
    """Solve when H names a continuous state; exact lift at the last period.

    Regression guard against a refactor that narrows `_H_accepted_params`
    to reject state names. At the last period where `working_life` is
    active, `E_next_V = 0` (dead utility is zero), so adding
    `wealth_weight * wealth` to `Q` shifts `V` by exactly that term —
    independent of the argmax.
    """
    model = _make_model(custom_H=wealth_H)
    common = {
        "utility": {"disutility_of_work": 0.5},
        "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
    }
    V_zero = model.solve(
        params={
            "working_life": {
                "H": {"discount_factor": 0.95, "wealth_weight": 0.0},
                **common,
            },
            "dead": {},
        },
    )
    V_pos = model.solve(
        params={
            "working_life": {
                "H": {"discount_factor": 0.95, "wealth_weight": 0.1},
                **common,
            },
            "dead": {},
        },
    )
    lift_at_terminal = (
        V_pos[FINAL_AGE_ALIVE]["working_life"] - V_zero[FINAL_AGE_ALIVE]["working_life"]
    )
    expected = 0.1 * jnp.linspace(0.5, 10.0, 30)
    assert bool(jnp.allclose(lift_at_terminal, expected, atol=1e-5))


def consumption_H(
    utility: float,
    E_next_V: float,
    discount_factor: float,
    consumption: float,
    action_weight: float,
) -> float:
    return utility + discount_factor * E_next_V + action_weight * consumption


def test_h_consumes_continuous_action():
    """H may name a continuous action; non-zero weight shifts V.

    Regression guard: when `H` names `consumption`, the scalar at the
    current action-gridpoint is bound at Q evaluation (before argmax).
    A positive `action_weight` therefore shifts `V` relative to the
    `action_weight=0` baseline.
    """
    model = _make_model(custom_H=consumption_H)
    common = {
        "utility": {"disutility_of_work": 0.5},
        "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
    }
    V_zero = model.solve(
        params={
            "working_life": {
                "H": {"discount_factor": 0.95, "action_weight": 0.0},
                **common,
            },
            "dead": {},
        },
    )
    V_pos = model.solve(
        params={
            "working_life": {
                "H": {"discount_factor": 0.95, "action_weight": 0.1},
                **common,
            },
            "dead": {},
        },
    )
    non_terminal = [p for p in V_zero if p <= FINAL_AGE_ALIVE]
    assert non_terminal
    diffs_exist = any(
        not jnp.allclose(V_zero[p]["working_life"], V_pos[p]["working_life"])
        for p in non_terminal
    )
    assert diffs_exist, "action_weight>0 must shift V at some working-life period"


def labor_supply_H(
    utility: float,
    E_next_V: float,
    discount_factor: float,
    labor_supply: DiscreteAction,
    bonus: float,
) -> FloatND:
    return (
        utility + discount_factor * E_next_V + bonus * labor_supply.astype(jnp.float32)
    )


def test_h_consumes_discrete_action():
    """H may name a discrete action; solve compiles and V shapes match baseline.

    Regression guard: discrete action scalars reach `H` via
    `states_actions_params` the same way continuous ones do.
    """
    model = _make_model(custom_H=labor_supply_H)
    V = model.solve(
        params={
            "working_life": {
                "H": {"discount_factor": 0.95, "bonus": 0.1},
                "utility": {"disutility_of_work": 0.5},
                "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
            },
            "dead": {},
        },
    )
    baseline = _make_model().solve(
        params={
            "discount_factor": 0.95,
            "working_life": {
                "utility": {"disutility_of_work": 0.5},
                "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
            },
        },
    )
    for period in V:
        for regime in V[period]:
            assert V[period][regime].shape == baseline[period][regime].shape


def pref_type_direct_H(
    utility: float,
    E_next_V: float,
    discount_factor: float,
    pref_type: DiscreteState,
) -> FloatND:
    return utility + discount_factor * E_next_V + 0.1 * pref_type.astype(jnp.float32)


def test_h_consumes_discrete_state():
    """H may name a discrete state directly, without a DAG function of that name.

    Regression guard: `pref_type` reaches `H` as a scalar per
    state-action gridpoint — the same path utility uses.
    `discount_factor` here is still a DAG output
    (`discount_factor_from_type`), proving state-direct and
    DAG-output paths can coexist in one `H`.
    """
    model = _make_model(custom_H=pref_type_direct_H, with_pref_type=True)
    V = model.solve(
        params={
            "discount_factor_by_type": jnp.array([0.70, 0.85, 0.99]),
            "working_life": {
                "utility": {"disutility_of_work": 0.5},
                "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
            },
        },
    )
    non_terminal = [p for p in V if p <= FINAL_AGE_ALIVE]
    assert non_terminal
    for period in non_terminal:
        v = V[period]["working_life"]
        assert 3 in v.shape, f"Period {period}: pref_type axis missing ({v.shape})"
        assert bool(jnp.all(jnp.isfinite(v)))


def mixed_H(
    utility: float,
    E_next_V: float,
    discount_factor: float,
    ies: float,
    wealth: float,
    consumption: float,
    pref_type: DiscreteState,
) -> FloatND:
    rho = 1 - ies
    u_eff = utility + 1e-3 * wealth
    v_eff = E_next_V + 1e-3 * consumption
    combined = ((1 - discount_factor) * u_eff**rho + discount_factor * v_eff**rho) ** (
        1 / rho
    )
    return combined + 1e-3 * pref_type.astype(jnp.float32)


def test_h_consumes_flat_param_state_action_and_dag_output():
    """H may simultaneously name a flat param, a state, an action, and a DAG output.

    Regression guard: every kwarg-resolution path fires at once —
    `states_actions_params` supplies wealth/consumption/pref_type,
    flat params supply `ies`, the DAG supplies `discount_factor`.
    """
    model = _make_model(custom_H=mixed_H, with_pref_type=True)
    V = model.solve(
        params={
            "discount_factor_by_type": jnp.array([0.70, 0.85, 0.99]),
            "working_life": {
                "H": {"ies": 0.5},
                "utility": {"disutility_of_work": 0.5},
                "next_regime": {"final_age_alive": FINAL_AGE_ALIVE},
            },
        },
    )
    for period in V:
        if "working_life" in V[period]:
            v = V[period]["working_life"]
            assert bool(jnp.all(jnp.isfinite(v))), (
                f"Non-finite working_life V at period {period}"
            )
            assert 3 in v.shape
