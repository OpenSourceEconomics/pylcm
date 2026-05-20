"""Tests for _lcm.pandas_utils and categorical.to_categorical_dtype."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from _lcm.pandas_utils import (
    _build_discrete_grid_lookup,
    array_from_series,
    convert_series_in_params,
    initial_conditions_from_dataframe,
)
from _lcm.params.processing import broadcast_to_template
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.regime import Regime as UserRegime
from lcm.typing import ScalarInt
from tests.test_models.basic_discrete import (
    Health,
)
from tests.test_models.basic_discrete import (
    RegimeId as BasicRegimeId,
)
from tests.test_models.basic_discrete import (
    get_model as get_basic_model,
)
from tests.test_models.regime_markov import get_model as get_regime_markov_model
from tests.test_models.shock_grids import get_model as get_shock_model
from tests.test_models.stochastic import get_model as get_stochastic_model


@categorical(ordered=False)
class Occupation:
    blue_collar: ScalarInt
    white_collar: ScalarInt


def test_to_categorical_dtype_returns_correct_type():
    result = Health.to_categorical_dtype()  # ty: ignore[unresolved-attribute]
    assert isinstance(result, CategoricalDtype)


def test_to_categorical_dtype_has_correct_categories():
    result = Health.to_categorical_dtype()  # ty: ignore[unresolved-attribute]
    assert list(result.categories) == ["bad", "good"]


def test_to_categorical_dtype_preserves_order():
    result = Occupation.to_categorical_dtype()  # ty: ignore[unresolved-attribute]
    assert list(result.categories) == ["blue_collar", "white_collar"]


def test_to_categorical_dtype_is_not_ordered():
    result = Occupation.to_categorical_dtype()  # ty: ignore[unresolved-attribute]
    assert result.ordered is False


def test_to_categorical_dtype_ordered():
    @categorical(ordered=True)
    class Severity:
        mild: ScalarInt
        moderate: ScalarInt
        severe: ScalarInt

    result = Severity.to_categorical_dtype()  # ty: ignore[unresolved-attribute]
    assert result.ordered is True
    assert list(result.categories) == ["mild", "moderate", "severe"]


def test_build_discrete_grid_lookup_basic():
    regimes = {
        "a": UserRegime(
            transition=None,
            states={"health": DiscreteGrid(Health)},
            functions={"utility": lambda: 0.0},
        ),
    }
    lookup = _build_discrete_grid_lookup(regimes)
    assert "health" in lookup
    assert lookup["health"].categories == ("bad", "good")


def test_build_discrete_grid_lookup_ignores_continuous():
    regimes = {
        "a": UserRegime(
            transition=None,
            states={
                "health": DiscreteGrid(Health),
                "wealth": LinSpacedGrid(start=0, stop=100, n_points=10),
            },
            functions={"utility": lambda: 0.0},
        ),
    }
    lookup = _build_discrete_grid_lookup(regimes)
    assert "wealth" not in lookup
    assert "health" in lookup


def test_build_discrete_grid_lookup_inconsistent_raises():
    @categorical(ordered=False)
    class HealthAlt:
        sick: ScalarInt
        healthy: ScalarInt

    regimes = {
        "a": UserRegime(
            transition=None,
            states={"health": DiscreteGrid(Health)},
            functions={"utility": lambda: 0.0},
        ),
        "b": UserRegime(
            transition=None,
            states={"health": DiscreteGrid(HealthAlt)},
            functions={"utility": lambda: 0.0},
        ),
    }
    with pytest.raises(ValueError, match="Inconsistent DiscreteGrid"):
        _build_discrete_grid_lookup(regimes)


def test_regime_name_column_maps_to_regime_id_codes():
    """`regime_name` column (strings) yields a `regime_id` dict entry of int codes."""
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime_name": ["working_life", "retirement"],
            "health": ["bad", "good"],
            "wealth": [10.0, 50.0],
            "age": [25.0, 25.0],
        }
    )
    conditions = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert jnp.array_equal(
        conditions["regime_id"],
        jnp.array([BasicRegimeId.working_life, BasicRegimeId.retirement]),
    )
    assert "regime_name" not in conditions


def test_continuous_states_and_age():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime_name": ["working_life", "working_life"],
            "health": ["bad", "good"],
            "wealth": [10.0, 50.0],
            "age": [25.0, 35.0],
        }
    )
    conditions = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert jnp.array_equal(
        conditions["regime_id"],
        jnp.array([BasicRegimeId.working_life, BasicRegimeId.working_life]),
    )
    assert jnp.allclose(conditions["wealth"], jnp.array([10.0, 50.0]))
    assert jnp.allclose(conditions["age"], jnp.array([25.0, 35.0]))


def test_categorical_string_labels():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime_name": ["working_life", "retirement"],
            "health": ["bad", "good"],
            "wealth": [10.0, 50.0],
            "age": [25.0, 25.0],
        }
    )
    conditions = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert jnp.array_equal(
        conditions["regime_id"],
        jnp.array([BasicRegimeId.working_life, BasicRegimeId.retirement]),
    )
    assert jnp.array_equal(conditions["health"], jnp.array([Health.bad, Health.good]))


def test_categorical_pd_categorical_column():
    model = get_basic_model()
    health_dtype = Health.to_categorical_dtype()  # ty: ignore[unresolved-attribute]
    df = pd.DataFrame(
        {
            "regime_name": ["working_life", "working_life"],
            "health": pd.Categorical(["good", "bad"], dtype=health_dtype),
            "wealth": [10.0, 50.0],
            "age": [25.0, 25.0],
        }
    )
    conditions = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert jnp.array_equal(conditions["health"], jnp.array([Health.good, Health.bad]))


def test_multi_regime():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime_name": ["working_life", "retirement", "working_life"],
            "health": ["good", "bad", "good"],
            "wealth": [10.0, 50.0, 30.0],
            "age": [25.0, 25.0, 25.0],
        }
    )
    conditions = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert jnp.array_equal(
        conditions["regime_id"],
        jnp.array(
            [
                BasicRegimeId.working_life,
                BasicRegimeId.retirement,
                BasicRegimeId.working_life,
            ]
        ),
    )
    assert len(conditions["wealth"]) == 3


def test_missing_regime_column_raises():
    model = get_basic_model()
    df = pd.DataFrame({"wealth": [10.0]})
    with pytest.raises(ValueError, match="'regime_name' column"):
        initial_conditions_from_dataframe(
            df=df,
            user_regimes=model.user_regimes,
            regime_names_to_ids=model.regime_names_to_ids,
        )


def test_invalid_regime_name_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime_name": ["working_life", "nonexistent"],
            "wealth": [10.0, 50.0],
        }
    )
    with pytest.raises(ValueError, match="Invalid regime names"):
        initial_conditions_from_dataframe(
            df=df,
            user_regimes=model.user_regimes,
            regime_names_to_ids=model.regime_names_to_ids,
        )


def test_invalid_category_label_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime_name": ["working_life"],
            "health": ["excellent"],
            "wealth": [10.0],
            "age": [25.0],
        }
    )
    with pytest.raises(ValueError, match="Invalid labels"):
        initial_conditions_from_dataframe(
            df=df,
            user_regimes=model.user_regimes,
            regime_names_to_ids=model.regime_names_to_ids,
        )


def test_empty_dataframe_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {"regime_name": pd.Series([], dtype=str), "wealth": pd.Series([], dtype=float)}
    )
    with pytest.raises(ValueError, match="empty"):
        initial_conditions_from_dataframe(
            df=df,
            user_regimes=model.user_regimes,
            regime_names_to_ids=model.regime_names_to_ids,
        )


def test_unknown_column_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime_name": ["working_life"],
            "health": ["bad"],
            "wealth": [10.0],
            "age": [25.0],
            "subject_id": [42],
        }
    )
    with pytest.raises(ValueError, match="Unknown columns"):
        initial_conditions_from_dataframe(
            df=df,
            user_regimes=model.user_regimes,
            regime_names_to_ids=model.regime_names_to_ids,
        )


def test_missing_state_column_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime_name": ["working_life"],
            "age": [25.0],
            # missing "health" and "wealth"
        }
    )
    with pytest.raises(ValueError, match="Missing required"):
        initial_conditions_from_dataframe(
            df=df,
            user_regimes=model.user_regimes,
            regime_names_to_ids=model.regime_names_to_ids,
        )


def test_shock_state_columns_accepted():
    """Shock grid columns are accepted as continuous float columns."""
    model = get_shock_model(n_periods=4, distribution_type="uniform")
    df = pd.DataFrame(
        {
            "regime_name": ["alive", "alive"],
            "wealth": [2.0, 4.0],
            "health": ["bad", "good"],
            "income": [0.3, 0.7],
            "age": [0.0, 0.0],
        }
    )
    conditions = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert jnp.allclose(conditions["income"], jnp.array([0.3, 0.7]))
    assert jnp.allclose(conditions["wealth"], jnp.array([2.0, 4.0]))
    assert "regime_id" in conditions


def test_shock_state_columns_required():
    """DataFrame without shock columns raises (shocks are required)."""
    model = get_shock_model(n_periods=4, distribution_type="uniform")
    df = pd.DataFrame(
        {
            "regime_name": ["alive", "alive"],
            "wealth": [2.0, 4.0],
            "health": ["bad", "good"],
            "age": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match=r"Missing required state columns.*income"):
        initial_conditions_from_dataframe(
            df=df,
            user_regimes=model.user_regimes,
            regime_names_to_ids=model.regime_names_to_ids,
        )


def test_round_trip_with_discrete_model():
    """Verify DataFrame-based initial states match raw arrays."""
    from tests.test_models.deterministic.discrete import (  # noqa: PLC0415
        DiscreteWealth,
        RegimeId,
        get_model,
        get_params,
    )

    n_periods = 3
    model = get_model(n_periods)
    params = get_params(n_periods)

    # Raw array approach
    raw_conditions = {
        "wealth": jnp.array([DiscreteWealth.low, DiscreteWealth.high]),
        "age": jnp.array([50.0, 50.0]),
        "regime_id": jnp.array([RegimeId.working_life, RegimeId.working_life]),
    }
    result_raw = model.simulate(
        log_level="debug",
        params=params,
        initial_conditions=raw_conditions,
        period_to_regime_to_V_arr=None,
    )

    # DataFrame approach
    df = pd.DataFrame(
        {
            "regime_name": ["working_life", "working_life"],
            "wealth": ["low", "high"],
            "age": [50.0, 50.0],
        }
    )
    df_conditions = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    result_df = model.simulate(
        log_level="debug",
        params=params,
        initial_conditions=df_conditions,
        period_to_regime_to_V_arr=None,
    )

    df_raw = result_raw.to_dataframe()
    df_from_df = result_df.to_dataframe()
    pd.testing.assert_frame_equal(df_raw, df_from_df)


@categorical(ordered=True)
class HealthWithDisability:
    disabled: ScalarInt
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class _HetRegimeId:
    pre65: ScalarInt
    post65: ScalarInt
    dead: ScalarInt


def _het_next_regime() -> ScalarInt:
    return _HetRegimeId.dead


def _het_utility(wealth: float, health: int, bonus: float) -> float:
    return wealth + health + bonus


def _het_next_wealth(wealth: float) -> float:
    return wealth


def _het_dead_utility() -> float:
    return 0.0


def _get_heterogeneous_health_model() -> Model:
    """Model where 'health' has different categories per regime."""
    pre65 = UserRegime(
        transition=_het_next_regime,
        active=lambda age: age < 65,
        states={
            "health": DiscreteGrid(HealthWithDisability),
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
        },
        state_transitions={"health": None, "wealth": _het_next_wealth},
        functions={"utility": _het_utility},
    )
    post65 = UserRegime(
        transition=_het_next_regime,
        active=lambda age: 65 <= age < 80,
        states={
            "health": DiscreteGrid(Health),
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
        },
        state_transitions={"health": None, "wealth": _het_next_wealth},
        functions={"utility": _het_utility},
    )
    dead = UserRegime(
        transition=None,
        functions={"utility": _het_dead_utility},
    )
    return Model(
        regimes={"pre65": pre65, "post65": post65, "dead": dead},
        ages=AgeGrid(start=50, stop=80, step="10Y"),
        regime_id_class=_HetRegimeId,
    )


def test_initial_conditions_heterogeneous_health_grids() -> None:
    """Handle regimes with different categories for the same state."""
    model = _get_heterogeneous_health_model()
    df = pd.DataFrame(
        {
            "regime_name": ["pre65", "pre65", "post65", "post65"],
            "health": ["disabled", "good", "bad", "good"],
            "wealth": [10.0, 50.0, 30.0, 70.0],
            "age": [50.0, 50.0, 70.0, 70.0],
        }
    )
    result = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )

    # pre65: disabled=0, good=2; post65: bad=0, good=1
    assert jnp.array_equal(result["health"], jnp.array([0, 2, 0, 1]))
    assert jnp.allclose(result["wealth"], jnp.array([10.0, 50.0, 30.0, 70.0]))
    assert jnp.array_equal(
        result["regime_id"],
        jnp.array(
            [
                _HetRegimeId.pre65,
                _HetRegimeId.pre65,
                _HetRegimeId.post65,
                _HetRegimeId.post65,
            ]
        ),
    )


def test_initial_conditions_heterogeneous_state_sets() -> None:
    """Handle regimes where a state only exists in some regimes."""

    @categorical(ordered=False)
    class _Rid:
        with_status: ScalarInt
        without_status: ScalarInt
        dead: ScalarInt

    @categorical(ordered=False)
    class _Status:
        low: ScalarInt
        high: ScalarInt

    def _next_regime() -> ScalarInt:
        return _Rid.dead

    def _utility_with_status(wealth: float, status: int) -> float:
        return wealth + status

    def _utility_without_status(wealth: float) -> float:
        return wealth

    with_status = UserRegime(
        transition=_next_regime,
        states={
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
            "status": DiscreteGrid(_Status),
        },
        state_transitions={"wealth": None, "status": None},
        functions={"utility": _utility_with_status},
    )
    without_status = UserRegime(
        transition=_next_regime,
        states={"wealth": LinSpacedGrid(start=0, stop=100, n_points=5)},
        state_transitions={"wealth": None},
        functions={"utility": _utility_without_status},
    )
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})

    model = Model(
        regimes={
            "with_status": with_status,
            "without_status": without_status,
            "dead": dead,
        },
        ages=AgeGrid(start=50, stop=52, step="Y"),
        regime_id_class=_Rid,
    )

    df = pd.DataFrame(
        {
            "regime_name": ["with_status", "with_status", "without_status"],
            "wealth": [10.0, 20.0, 30.0],
            "status": ["low", "high", pd.NA],
            "age": [50.0, 51.0, 50.0],
        }
    )
    result = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )

    # status: low=0, high=1 for with_status regime
    assert result["status"][0] == 0
    assert result["status"][1] == 1
    # without_status regime: NaN pre-fill → explicit int32 sentinel
    assert result["status"][2] == jnp.iinfo(jnp.int32).min
    assert jnp.allclose(result["wealth"], jnp.array([10.0, 20.0, 30.0]))


def test_initial_conditions_shock_grid_heterogeneous_state_sets() -> None:
    """A shock state (income) only present in one regime is NaN-filled elsewhere."""
    from lcm import UniformIIDProcess  # noqa: PLC0415

    @categorical(ordered=False)
    class _Rid:
        earner: ScalarInt
        retiree: ScalarInt
        dead: ScalarInt

    def _next_regime() -> ScalarInt:
        return _Rid.dead

    def _earner_utility(wealth: float, income: float) -> float:
        return wealth + income

    def _retiree_utility(wealth: float) -> float:
        return wealth

    earner = UserRegime(
        transition=_next_regime,
        states={
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
            "income": UniformIIDProcess(n_points=5),
        },
        state_transitions={"wealth": None},
        functions={"utility": _earner_utility},
    )
    retiree = UserRegime(
        transition=_next_regime,
        states={"wealth": LinSpacedGrid(start=0, stop=100, n_points=5)},
        state_transitions={"wealth": None},
        functions={"utility": _retiree_utility},
    )
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})

    model = Model(
        regimes={"earner": earner, "retiree": retiree, "dead": dead},
        ages=AgeGrid(start=50, stop=52, step="Y"),
        regime_id_class=_Rid,
    )

    df = pd.DataFrame(
        {
            "regime_name": ["earner", "earner", "retiree"],
            "wealth": [10.0, 20.0, 30.0],
            "income": [0.3, 0.7, float("nan")],
            "age": [50.0, 51.0, 50.0],
        }
    )
    result = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )

    # earner subjects retain provided shock values
    assert jnp.isclose(result["income"][0], 0.3)
    assert jnp.isclose(result["income"][1], 0.7)
    # retiree has no shock state; value is not asserted (only earner reads it)
    assert jnp.allclose(result["wealth"], jnp.array([10.0, 20.0, 30.0]))


def test_convert_series_heterogeneous_grids() -> None:
    """convert_series_in_params handles per-regime grid lookup."""
    model = _get_heterogeneous_health_model()
    ages = model.ages.exact_values
    sr = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.Index(ages, name="age"))
    # Should not raise despite heterogeneous health grids
    internal = broadcast_to_template(
        params={"bonus": sr}, template=model._params_template, required=False
    )
    convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )


def test_convert_series_next_function_no_outcome_axis() -> None:
    """Period-indexed Series on a next_* function should not get outcome axis."""

    @categorical(ordered=False)
    class _RId:
        a: ScalarInt
        dead: ScalarInt

    def _next_regime() -> ScalarInt:
        return _RId.dead

    def _next_wealth(wealth: float, rate: float, period: int) -> float:  # noqa: ARG001
        return wealth * rate

    def _utility(wealth: float) -> float:
        return wealth

    def _dead_utility() -> float:
        return 0.0

    a = UserRegime(
        transition=_next_regime,
        states={"wealth": LinSpacedGrid(start=0, stop=100, n_points=5)},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility},
    )
    dead = UserRegime(transition=None, functions={"utility": _dead_utility})
    m = Model(
        regimes={"a": a, "dead": dead},
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=_RId,
    )
    ages = m.ages.exact_values
    sr = pd.Series(range(len(ages)), index=pd.Index(ages, name="age"), dtype=float)
    # Should not raise KeyError on continuous state 'wealth'
    internal = broadcast_to_template(
        params={"rate": sr}, template=m._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=m.user_regimes,
        ages=m.ages,
        regime_names_to_ids=m.regime_names_to_ids,
    )
    assert result is not None


def test_heterogeneous_health_solve_simulate() -> None:
    """Solve and simulate with heterogeneous discrete grids, check DataFrame output."""
    model = _get_heterogeneous_health_model()
    df = pd.DataFrame(
        {
            "regime_name": ["pre65", "pre65", "post65", "post65"],
            "health": ["disabled", "good", "bad", "good"],
            "wealth": [10.0, 50.0, 30.0, 70.0],
            "age": [50.0, 50.0, 70.0, 70.0],
        }
    )
    ic = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    result = model.simulate(
        log_level="debug",
        params={"bonus": 0.0, "discount_factor": 0.95},
        initial_conditions=ic,
        period_to_regime_to_V_arr=None,
    )
    out = result.to_dataframe()

    # health column should be Categorical with merged ordered categories
    assert isinstance(out["health"].dtype, CategoricalDtype)
    assert list(out["health"].cat.categories) == ["disabled", "bad", "good"]
    assert out["health"].cat.ordered

    # Period 0: pre65 subjects have correct health labels
    period_0 = out.query("period == 0").sort_values("subject_id")
    assert list(period_0["health"]) == ["disabled", "good"]

    # Period 2: post65 subjects have correct health labels
    period_2 = out.query("period == 2 and regime_name == 'post65'").sort_values(
        "subject_id"
    )
    assert list(period_2["health"]) == ["bad", "good"]


def test_heterogeneous_health_simulate_use_labels_false() -> None:
    """With use_labels=False, health column contains raw integer codes."""
    model = _get_heterogeneous_health_model()
    df = pd.DataFrame(
        {
            "regime_name": ["pre65", "post65"],
            "health": ["disabled", "good"],
            "wealth": [10.0, 70.0],
            "age": [50.0, 70.0],
        }
    )
    ic = initial_conditions_from_dataframe(
        df=df,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    result = model.simulate(
        log_level="debug",
        params={"bonus": 0.0, "discount_factor": 0.95},
        initial_conditions=ic,
        period_to_regime_to_V_arr=None,
    )
    out = result.to_dataframe(use_labels=False)

    # health column should contain integer codes, not Categorical
    assert not isinstance(out["health"].dtype, CategoricalDtype)


def _make_partner_probs_array():
    """Build a (n_periods=3, n_work=2, n_partner=2, n_next_partner=2) array."""
    return jnp.array(
        [
            [[[0.3, 0.7], [0.6, 0.4]], [[0.1, 0.9], [0.5, 0.5]]],
            [[[0.4, 0.6], [0.8, 0.2]], [[0.2, 0.8], [0.7, 0.3]]],
            [[[0.5, 0.5], [0.9, 0.1]], [[0.3, 0.7], [0.6, 0.4]]],
        ]
    )


def _array_to_series(arr, model):
    """Convert a probs array to a labeled Series with named MultiIndex using ages."""
    partner_labels = ("single", "partnered")
    work_labels = ("work", "retire")
    ages = model.ages.values  # noqa: PD011

    records = []
    for p in range(model.n_periods):
        for w_idx, w_label in enumerate(work_labels):
            for s_idx, s_label in enumerate(partner_labels):
                for ns_idx, ns_label in enumerate(partner_labels):
                    records.append(
                        (
                            (float(ages[p]), w_label, s_label, ns_label),
                            float(arr[p, w_idx, s_idx, ns_idx]),
                        )
                    )

    index = pd.MultiIndex.from_tuples(
        [r[0] for r in records],
        names=["age", "labor_supply", "partner", "next_partner"],
    )
    return pd.Series([r[1] for r in records], index=index)


def test_array_from_series_transition_basic_round_trip():
    """4D transition probs via array_from_series."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_partner",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_array_from_series_transition_categorical_labels():
    """Verify specific label-based values in transition probs array."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_partner",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    # age=40, work, single->partnered
    assert float(result[0, 0, 0, 1]) == pytest.approx(0.7)
    # age=50, retire, partnered->single
    assert float(result[1, 1, 1, 0]) == pytest.approx(0.7)


def test_array_from_series_transition_reordered_levels():
    """Reordered MultiIndex levels still produce correct output."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    # Reorder levels: put next_partner first, then partner, work, age
    series = series.reorder_levels(["next_partner", "partner", "labor_supply", "age"])
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_partner",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_array_from_series_transition_wrong_level_names_raises():
    """Wrong MultiIndex level names raise ValueError."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    # Rename outcome level to something wrong
    series.index = series.index.set_names(
        ["age", "labor_supply", "partner", "wrong_name"]
    )
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    with pytest.raises(ValueError, match="level names"):
        array_from_series(
            sr=series,
            func=func,
            param_name="probs_array",
            func_name="next_partner",
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
            regime_name="working_life",
        )


def test_array_from_series_transition_invalid_label_raises():
    """Invalid categorical label in transition probs raises ValueError."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    # Replace one label with an invalid one
    new_index = series.index.set_levels(["single", "INVALID"], level="partner")
    series.index = new_index
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    with pytest.raises(ValueError, match="Invalid labels"):
        array_from_series(
            sr=series,
            func=func,
            param_name="probs_array",
            func_name="next_partner",
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
            regime_name="working_life",
        )


def test_array_from_series_transition_period_level_raises():
    """Using 'period' instead of 'age' should raise a clear error."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(0, "work", "single", "single")],
        names=["period", "labor_supply", "partner", "next_partner"],
    )
    series = pd.Series([1.0], index=index)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    with pytest.raises(ValueError, match="age"):
        array_from_series(
            sr=series,
            func=func,
            param_name="probs_array",
            func_name="next_partner",
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
            regime_name="working_life",
        )


def test_array_from_series_transition_duplicate_level_names_raises():
    """Duplicate MultiIndex level names should raise."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(40.0, "work", "single", "single")],
        names=["age", "labor_supply", "labor_supply", "next_partner"],
    )
    series = pd.Series([1.0], index=index)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    with pytest.raises(ValueError, match="duplicate"):
        array_from_series(
            sr=series,
            func=func,
            param_name="probs_array",
            func_name="next_partner",
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
            regime_name="working_life",
        )


def test_array_from_series_transition_invalid_age_dropped():
    """Age values not on the model's AgeGrid are silently dropped (all NaN)."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    # Replace age level with invalid values
    series.index = series.index.set_codes([0] * len(series), level="age").set_levels(
        [999.0], level="age"
    )
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_partner",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    # All ages are invalid, so all positions should be NaN
    assert jnp.all(jnp.isnan(result))


def test_array_from_series_transition_sparse_input_fills_nan():
    """Unfilled positions should be NaN, not zero."""
    model = get_stochastic_model(3)
    # Provide data for only the first age — other ages should be NaN
    index = pd.MultiIndex.from_tuples(
        [
            (40.0, "work", "single", "single"),
            (40.0, "work", "single", "partnered"),
        ],
        names=["age", "labor_supply", "partner", "next_partner"],
    )
    series = pd.Series([0.3, 0.7], index=index)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_partner",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    # age=40 (period 0), work (0), single (0) → provided
    np.testing.assert_allclose(result[0, 0, 0], jnp.array([0.3, 0.7]), atol=1e-7)
    # age=50 (period 1) → all NaN
    assert jnp.all(jnp.isnan(result[1]))
    # age=60 (period 2) → all NaN
    assert jnp.all(jnp.isnan(result[2]))


def _make_regime_probs_array():
    """Build a (n_periods=3, n_health=2, n_regimes=2) array."""
    return jnp.array(
        [
            [[0.95, 0.05], [0.98, 0.02]],  # age 60: bad=95%, good=98%
            [[0.90, 0.10], [0.96, 0.04]],  # age 61
            [[0.0, 1.0], [0.0, 1.0]],  # age 62: certain death
        ]
    )


def _regime_array_to_series(arr, model):
    """Convert a regime probs array to a labeled Series using ages."""
    health_labels = ("bad", "good")
    regime_labels = ("alive", "dead")
    ages = model.ages.values  # noqa: PD011

    records = []
    for period_idx in range(model.n_periods):
        for h_idx, h_label in enumerate(health_labels):
            for r_idx, r_label in enumerate(regime_labels):
                records.append(
                    (
                        (float(ages[period_idx]), h_label, r_label),
                        float(arr[period_idx, h_idx, r_idx]),
                    )
                )

    index = pd.MultiIndex.from_tuples(
        [r[0] for r in records],
        names=["age", "health", "next_regime"],
    )
    return pd.Series([r[1] for r in records], index=index)


def test_array_from_series_regime_transition_basic_round_trip():
    """Regime transition probs via array_from_series."""
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    func = model.user_regimes["alive"].get_all_functions()["next_regime"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_regime",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="alive",
    )
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_array_from_series_regime_transition_reordered_levels():
    """Reordered MultiIndex levels for regime transitions."""
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    series = series.reorder_levels(["next_regime", "health", "age"])
    func = model.user_regimes["alive"].get_all_functions()["next_regime"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_regime",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="alive",
    )
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_array_from_series_regime_transition_wrong_level_names_raises():
    """Wrong MultiIndex level names for regime transition raise ValueError."""
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    series.index = series.index.set_names(["age", "health", "wrong_name"])
    func = model.user_regimes["alive"].get_all_functions()["next_regime"]
    with pytest.raises(ValueError, match="level names"):
        array_from_series(
            sr=series,
            func=func,
            param_name="probs_array",
            func_name="next_regime",
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
            regime_name="alive",
        )


def test_array_from_series_regime_transition_invalid_label_raises():
    """Invalid regime label in transition probs raises ValueError."""
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    new_index = series.index.set_levels(["alive", "INVALID"], level="next_regime")
    series.index = new_index
    func = model.user_regimes["alive"].get_all_functions()["next_regime"]
    with pytest.raises(ValueError, match="Invalid labels"):
        array_from_series(
            sr=series,
            func=func,
            param_name="probs_array",
            func_name="next_regime",
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
            regime_name="alive",
        )


def _build_partner_probs_series(model: Model) -> pd.Series:
    """Build a 4D Series with age x labor_supply x partner x next_partner MultiIndex."""
    partner_labels = ("single", "partnered")
    work_labels = ("work", "retire")
    ages = model.ages.values  # noqa: PD011

    records = []
    val = 1.0
    for period_idx in range(model.n_periods):
        for w_label in work_labels:
            for p_label in partner_labels:
                for np_label in partner_labels:
                    records.append(
                        (
                            (float(ages[period_idx]), w_label, p_label, np_label),
                            val,
                        )
                    )
                    val += 1.0

    index = pd.MultiIndex.from_tuples(
        [r[0] for r in records],
        names=["age", "labor_supply", "partner", "next_partner"],
    )
    return pd.Series([r[1] for r in records], index=index)


def test_array_from_series_fully_qualified() -> None:
    """Fully qualified func/param/regime produces correct 4D shape."""
    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_partner",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    assert result.shape == (3, 2, 2, 2)
    # First element: age=40, work, single, single
    assert float(result[0, 0, 0, 0]) == pytest.approx(1.0)


def test_array_from_series_scalar_param() -> None:
    """Scalar parameter (no indexing params) returns 1D array from values."""
    model = get_stochastic_model(3)
    # labor_income(is_working, wage) -- is_working is a function output, not
    # a state or action, so wage has no indexing params.
    series = pd.Series([10.0])
    func = model.user_regimes["working_life"].get_all_functions()["labor_income"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="wage",
        func_name="labor_income",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    np.testing.assert_allclose(result, jnp.array([10.0]))


def test_array_from_series_extra_ages_dropped() -> None:
    """Ages outside the model's AgeGrid are silently dropped."""
    model = get_stochastic_model(3)
    # Model ages: 40, 50, 60. Add data for ages 30 and 70.
    partner_labels = ("single", "partnered")
    work_labels = ("work", "retire")
    all_ages = [30.0, 40.0, 50.0, 60.0, 70.0]

    records = []
    val = 1.0
    for age in all_ages:
        for w_label in work_labels:
            for p_label in partner_labels:
                for np_label in partner_labels:
                    records.append(((age, w_label, p_label, np_label), val))
                    val += 1.0

    index = pd.MultiIndex.from_tuples(
        [r[0] for r in records],
        names=["age", "labor_supply", "partner", "next_partner"],
    )
    series = pd.Series([r[1] for r in records], index=index)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_partner",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    assert result.shape == (3, 2, 2, 2)
    # age=30 was idx 0 in input (vals 1-8), age=40 was idx 1 (vals 9-16)
    assert float(result[0, 0, 0, 0]) == pytest.approx(9.0)


def test_array_from_series_missing_ages_filled_with_nan() -> None:
    """Missing grid ages produce NaN instead of raising."""
    model = get_stochastic_model(3)
    # Only provide data for age=40, not 50 or 60
    records = []
    val = 1.0
    for w_label in ("work", "retire"):
        for p_label in ("single", "partnered"):
            for np_label in ("single", "partnered"):
                records.append(((40.0, w_label, p_label, np_label), val))
                val += 1.0

    index = pd.MultiIndex.from_tuples(
        [r[0] for r in records],
        names=["age", "labor_supply", "partner", "next_partner"],
    )
    series = pd.Series([r[1] for r in records], index=index)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_partner",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    assert result.shape == (3, 2, 2, 2)
    # age=40 (period 0) filled
    assert not jnp.any(jnp.isnan(result[0]))
    # age=50, age=60 all NaN
    assert jnp.all(jnp.isnan(result[1]))
    assert jnp.all(jnp.isnan(result[2]))


def test_array_from_series_reordered_levels() -> None:
    """MultiIndex with levels in different order still produces correct output."""
    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    # Reorder: next_partner, partner, labor_supply, age
    series = series.reorder_levels(["next_partner", "partner", "labor_supply", "age"])  # ty: ignore[invalid-argument-type]
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="probs_array",
        func_name="next_partner",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    assert result.shape == (3, 2, 2, 2)
    assert float(result[0, 0, 0, 0]) == pytest.approx(1.0)


def test_array_from_series_invalid_label_raises() -> None:
    """Invalid categorical label raises ValueError."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(40.0, "work", "INVALID", "single")],
        names=["age", "labor_supply", "partner", "next_partner"],
    )
    series = pd.Series([1.0], index=index)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    with pytest.raises(ValueError, match="Invalid labels"):
        array_from_series(
            sr=series,
            func=func,
            param_name="probs_array",
            func_name="next_partner",
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
            regime_name="working_life",
        )


def test_array_from_series_wrong_level_names_raises() -> None:
    """Level names that don't match expected indexing params raise ValueError."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(40.0, "work", "single", "single")],
        names=["age", "labor_supply", "wrong_name", "next_partner"],
    )
    series = pd.Series([1.0], index=index)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    with pytest.raises(ValueError, match="level names"):
        array_from_series(
            sr=series,
            func=func,
            param_name="probs_array",
            func_name="next_partner",
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
            regime_name="working_life",
        )


def test_array_from_series_integer_labels_rejected() -> None:
    """Integer labels on a categorical level raise ValueError."""
    model = get_stochastic_model(3)
    # Use integer labels (0, 1) instead of string category names
    index = pd.MultiIndex.from_tuples(
        [(40.0, 0, "single", "single"), (40.0, 1, "single", "single")],
        names=["age", "labor_supply", "partner", "next_partner"],
    )
    series = pd.Series([0.5, 0.5], index=index)
    func = model.user_regimes["working_life"].get_all_functions()["next_partner"]
    with pytest.raises(ValueError, match="non-string labels"):
        array_from_series(
            sr=series,
            func=func,
            param_name="probs_array",
            func_name="next_partner",
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
            regime_name="working_life",
        )


def test_array_from_series_scalar_param_explicit_lookup() -> None:
    """Scalar parameter with explicit func lookup returns 1D array."""
    model = get_stochastic_model(3)
    # labor_income only exists in working_life. wage has no indexing params.
    series = pd.Series([10.0])
    func = model.user_regimes["working_life"].get_all_functions()["labor_income"]
    result = array_from_series(
        sr=series,
        func=func,
        param_name="wage",
        func_name="labor_income",
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
        regime_name="working_life",
    )
    np.testing.assert_allclose(result, jnp.array([10.0]))


def test_convert_series_function_level_series() -> None:
    """Series at function level is converted to 4D transition prob array."""
    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    params = {
        "working_life": {
            "next_partner": {"probs_array": series},
        },
    }
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    arr = result["working_life"]["next_partner__probs_array"]
    assert arr.shape == (3, 2, 2, 2)  # ty: ignore[unresolved-attribute]
    assert float(arr[0, 0, 0, 0]) == pytest.approx(1.0)  # ty: ignore[not-subscriptable]


def test_convert_series_model_level_scalar_passthrough() -> None:
    """Scalar values at model level pass through unchanged."""
    model = get_stochastic_model(3)
    params = {"discount_factor": 0.95}
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    # Model-level param is broadcast to all regimes/functions that need it
    assert result["working_life"]["H__discount_factor"] == 0.95
    assert result["retirement"]["H__discount_factor"] == 0.95


def test_convert_series_regime_level_series() -> None:
    """Series at regime level is resolved via template and converted."""
    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    params = {
        "working_life": {
            "probs_array": series,
        },
    }
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    arr = result["working_life"]["next_partner__probs_array"]
    assert arr.shape == (3, 2, 2, 2)  # ty: ignore[unresolved-attribute]


def test_convert_series_mixed_dict() -> None:
    """Mix of scalars, arrays, and Series in one params dict."""
    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    params = {
        "discount_factor": 0.95,
        "working_life": {
            "utility": {"disutility_of_work": 0.5},
            "next_partner": {"probs_array": series},
            "next_wealth": {"interest_rate": 0.05},
            "labor_income": {"wage": jnp.array([10.0])},
        },
    }
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert result["working_life"]["H__discount_factor"] == 0.95
    assert result["working_life"]["utility__disutility_of_work"] == 0.5
    assert result["working_life"]["next_partner__probs_array"].shape == (3, 2, 2, 2)  # ty: ignore[unresolved-attribute]
    assert result["working_life"]["next_wealth__interest_rate"] == 0.05
    np.testing.assert_allclose(  # ty: ignore[no-matching-overload]
        result["working_life"]["labor_income__wage"], jnp.array([10.0])
    )


def test_convert_series_mapping_leaf() -> None:
    """Series inside a `UserMappingLeaf` is converted in place."""
    from lcm.params import UserMappingLeaf  # noqa: PLC0415

    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    leaf = UserMappingLeaf({"sub_key": series})
    params = {
        "working_life": {
            "next_partner": {"probs_array": leaf},
        },
    }
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    converted_leaf = result["working_life"]["next_partner__probs_array"]
    assert isinstance(converted_leaf, UserMappingLeaf)
    arr = converted_leaf.data["sub_key"]
    assert arr.shape == (3, 2, 2, 2)  # ty: ignore[unresolved-attribute]


def test_convert_series_nested_mapping_leaf() -> None:
    """Series inside nested `UserMappingLeaf` is recursively converted."""
    from lcm.params import UserMappingLeaf  # noqa: PLC0415

    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    inner = UserMappingLeaf({"sub": series})
    outer = UserMappingLeaf({"inner_leaf": inner})
    params = {
        "working_life": {
            "next_partner": {"probs_array": outer},
        },
    }
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    converted = result["working_life"]["next_partner__probs_array"]
    assert isinstance(converted, UserMappingLeaf)
    inner_converted = converted.data["inner_leaf"]
    assert isinstance(inner_converted, UserMappingLeaf)
    assert not isinstance(inner_converted.data["sub"], pd.Series)
    assert inner_converted.data["sub"].shape == (3, 2, 2, 2)  # ty: ignore[unresolved-attribute]


def test_convert_series_unknown_param_raises() -> None:
    """Unknown param name raises InvalidParamsError."""
    from lcm.exceptions import InvalidParamsError  # noqa: PLC0415

    model = get_stochastic_model(3)
    params = {"nonexistent_param": pd.Series([1.0])}
    with pytest.raises(InvalidParamsError, match="Unknown keys"):
        broadcast_to_template(
            params=params, template=model._params_template, required=False
        )


def test_convert_series_with_derived_categoricals() -> None:
    """Derived variable indexing requires explicit derived_categoricals."""
    # In the stochastic model, next_partner(period, labor_supply, partner, probs_array)
    # uses labor_supply — which is a DiscreteGrid action in working_life.
    # Simulate the case where the converter needs extra categoricals by building
    # a model where a function indexes by a variable not in the model grids.
    # Use the existing stochastic model: next_partner indexes by labor_supply,
    # which IS in the working_life action grids. But for the retirement regime,
    # labor_supply is NOT an action — it's a fixed param. So if we provide
    # probs_array for retirement's next_partner (which also indexes by
    # labor_supply in the source code), we need derived_categoricals to resolve it.
    from tests.test_models.stochastic import LaborSupply  # noqa: PLC0415

    model = get_stochastic_model(3)
    labor_grid = DiscreteGrid(LaborSupply)

    # Build a Series indexed by age x labor_supply x partner x next_partner
    ages = model.ages.values  # noqa: PD011
    records = []
    val = 1.0
    for period_idx in range(model.n_periods):
        for w_label in ("work", "retire"):
            for p_label in ("single", "partnered"):
                for np_label in ("single", "partnered"):
                    records.append(
                        (
                            (float(ages[period_idx]), w_label, p_label, np_label),
                            val,
                        )
                    )
                    val += 1.0

    index = pd.MultiIndex.from_tuples(
        [r[0] for r in records],
        names=["age", "labor_supply", "partner", "next_partner"],
    )
    series = pd.Series([r[1] for r in records], index=index)

    # Without derived_categoricals, this should fail (retirement doesn't have
    # labor_supply as an action)
    params = {"retirement": {"next_partner": {"probs_array": series}}}
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    with pytest.raises(ValueError, match="Unrecognised indexing parameter"):
        convert_series_in_params(
            flat_params=internal,
            user_regimes=model.user_regimes,
            ages=model.ages,
            regime_names_to_ids=model.regime_names_to_ids,
        )

    # With derived_categoricals on the regime, it succeeds
    updated_regimes = {
        name: (
            dataclasses.replace(r, derived_categoricals={"labor_supply": labor_grid})
            if name == "retirement"
            else r
        )
        for name, r in model.user_regimes.items()
    }
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=updated_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    arr = result["retirement"]["next_partner__probs_array"]
    assert arr.shape == (3, 2, 2, 2)  # ty: ignore[unresolved-attribute]


def test_convert_series_per_target_transition() -> None:
    """Per-target state transitions should be convertible."""
    from lcm import AgeGrid, MarkovTransition  # noqa: PLC0415
    from lcm.typing import DiscreteState, FloatND, Period  # noqa: PLC0415

    @categorical(ordered=False)
    class _RId:
        working: ScalarInt
        retired: ScalarInt

    def _health_probs(
        period: Period, health: DiscreteState, probs_array: FloatND
    ) -> FloatND:
        return probs_array[period, health]

    def _utility(health: DiscreteState, wealth: float) -> FloatND:
        return wealth + health

    def _next_wealth(wealth: float) -> float:
        return wealth

    working = UserRegime(
        states={
            "health": DiscreteGrid(Health),
            "wealth": LinSpacedGrid(start=0, stop=10, n_points=5),
        },
        state_transitions={
            "health": {
                "working": MarkovTransition(_health_probs),
                "retired": MarkovTransition(_health_probs),
            },
            "wealth": _next_wealth,
        },
        functions={"utility": _utility},
        transition=lambda age: jnp.where(age >= 1, _RId.retired, _RId.working),
        active=lambda age: age < 2,
    )
    retired = UserRegime(
        transition=None,
        states={
            "health": DiscreteGrid(Health),
            "wealth": LinSpacedGrid(start=0, stop=10, n_points=5),
        },
        functions={"utility": _utility},
    )
    model = Model(
        regimes={"working": working, "retired": retired},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RId,
    )

    index = pd.MultiIndex.from_tuples(
        [
            (0.0, "bad", "bad"),
            (0.0, "bad", "good"),
            (0.0, "good", "bad"),
            (0.0, "good", "good"),
            (1.0, "bad", "bad"),
            (1.0, "bad", "good"),
            (1.0, "good", "bad"),
            (1.0, "good", "good"),
        ],
        names=["age", "health", "next_health"],
    )
    sr = pd.Series([0.9, 0.1, 0.2, 0.8, 0.8, 0.2, 0.3, 0.7], index=index)

    params = {"working": {"to_working_next_health": {"probs_array": sr}}}
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    arr = result["working"]["to_working_next_health__probs_array"]
    assert arr.shape == (3, 2, 2)  # ty: ignore[unresolved-attribute]


def test_build_outcome_mapping_qualified_func_name() -> None:
    """`_build_outcome_mapping` should handle qualified names."""
    from _lcm.pandas_utils import _build_outcome_mapping  # noqa: PLC0415

    model = get_stochastic_model(3)
    from _lcm.pandas_utils import _build_discrete_grid_lookup  # noqa: PLC0415

    grids = _build_discrete_grid_lookup(model.user_regimes)
    result = _build_outcome_mapping(
        func_name="next_health__working",
        grids=grids,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert result.size == 2
    assert result.name == "next_health"


def test_convert_series_structured_derived_categoricals() -> None:
    """Regime-level derived_categoricals should allow different grids per regime."""
    from lcm import AgeGrid  # noqa: PLC0415
    from lcm.typing import FloatND  # noqa: PLC0415

    @categorical(ordered=False)
    class _RId:
        regime_a: ScalarInt
        regime_b: ScalarInt

    @categorical(ordered=False)
    class _ChoiceA:
        x: ScalarInt
        y: ScalarInt

    @categorical(ordered=False)
    class _ChoiceB:
        x: ScalarInt
        y: ScalarInt
        z: ScalarInt

    # "derived" is a DAG function output used as an index in both regimes,
    # but with different cardinalities (2 vs 3 categories). Since it's a
    # function output (not a state/action), it needs external categoricals.
    def func_a(wealth: float, derived: FloatND, rates: FloatND) -> FloatND:  # noqa: ARG001
        return rates[derived]

    def func_b(wealth: float, derived: FloatND, rates: FloatND) -> FloatND:  # noqa: ARG001
        return rates[derived]

    def _derived_a(wealth: float) -> FloatND:
        return jnp.where(wealth > 5, 1, 0)

    def _derived_b(wealth: float) -> FloatND:
        return jnp.where(wealth > 7, 2, jnp.where(wealth > 3, 1, 0))

    def _next_wealth_sc(wealth: float) -> float:
        return wealth

    regime_a = UserRegime(
        transition=lambda age: jnp.where(age >= 1, _RId.regime_b, _RId.regime_a),
        active=lambda age: age < 1,
        states={"wealth": LinSpacedGrid(start=0, stop=10, n_points=5)},
        state_transitions={"wealth": _next_wealth_sc},
        functions={"utility": func_a, "derived": _derived_a},
        derived_categoricals={"derived": DiscreteGrid(_ChoiceA)},
    )
    regime_b = UserRegime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=0, stop=10, n_points=5)},
        functions={"utility": func_b, "derived": _derived_b},
        derived_categoricals={"derived": DiscreteGrid(_ChoiceB)},
    )
    model = Model(
        regimes={"regime_a": regime_a, "regime_b": regime_b},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RId,
    )

    # "derived" has 2 outcomes in regime_a (_ChoiceA: x,y) and 3 in
    # regime_b (_ChoiceB: x,y,z). Each regime declares its own grid.
    sr_a = pd.Series([1.0, 2.0], index=pd.Index(["x", "y"], name="derived"))
    sr_b = pd.Series([1.0, 2.0, 3.0], index=pd.Index(["x", "y", "z"], name="derived"))

    params = {
        "regime_a": {"utility": {"rates": sr_a}},
        "regime_b": {"utility": {"rates": sr_b}},
    }
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result_both = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert result_both["regime_a"]["utility__rates"].shape == (2,)  # ty: ignore[unresolved-attribute]
    assert result_both["regime_b"]["utility__rates"].shape == (3,)  # ty: ignore[unresolved-attribute]


def test_convert_series_runtime_grid_param() -> None:
    """Runtime grid points should be convertible or give a clear error."""
    from lcm import AgeGrid, IrregSpacedGrid  # noqa: PLC0415

    @categorical(ordered=False)
    class _RId:
        alive: ScalarInt
        dead: ScalarInt

    alive = UserRegime(
        transition=lambda age: jnp.where(age >= 1, _RId.dead, _RId.alive),
        active=lambda age: age < 1,
        states={"wealth": IrregSpacedGrid(n_points=4)},
        state_transitions={"wealth": lambda wealth: wealth},
        functions={"utility": lambda wealth: wealth},
    )
    dead = UserRegime(
        transition=None,
        states={"wealth": IrregSpacedGrid(n_points=4)},
        functions={"utility": lambda wealth: wealth},
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RId,
    )

    sr = pd.Series([1.0, 2.0, 5.0, 10.0])
    params = {"alive": {"wealth": {"points": sr}}}
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    np.testing.assert_allclose(result["alive"]["wealth__points"], sr.to_numpy())  # ty: ignore[no-matching-overload]


def test_convert_series_sequence_leaf_traversal() -> None:
    """Series inside a `UserSequenceLeaf` should be converted to JAX arrays."""
    from _lcm.params.sequence_leaf import UserSequenceLeaf  # noqa: PLC0415

    model = get_stochastic_model(3)
    sr = pd.Series([10.0])
    leaf = UserSequenceLeaf((sr, 42))
    params = {"working_life": {"labor_income": {"wage": leaf}}}
    internal = broadcast_to_template(
        params=params, template=model._params_template, required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    converted = result["working_life"]["labor_income__wage"]
    assert isinstance(converted, UserSequenceLeaf)
    assert not isinstance(converted.data[0], pd.Series)
    np.testing.assert_allclose(converted.data[0], jnp.array([10.0]))  # ty: ignore[no-matching-overload]


def test_resolve_categoricals_conflict_raises() -> None:
    """Derived categoricals that conflict with model grids raise ValueError."""
    from _lcm.pandas_utils import _resolve_categoricals  # noqa: PLC0415

    model = get_stochastic_model(3)

    @categorical(ordered=False)
    class WrongPartner:
        x: ScalarInt
        y: ScalarInt

    conflicting_regime = dataclasses.replace(
        model.user_regimes["working_life"],
        derived_categoricals={"partner": DiscreteGrid(WrongPartner)},
    )
    with pytest.raises(ValueError, match="conflicts with model grid"):
        _resolve_categoricals(
            user_regimes={"working_life": conflicting_regime},
            regime_name="working_life",
        )


def test_convert_series_cross_grid_transition() -> None:
    """Outcome axis must use the TARGET regime's grid, not the source's.

    When a per-target MarkovTransition crosses grid sizes (e.g. 3-state
    source → 2-state target), the converted array's last dimension must
    match the target's grid size (2), not the source's (3).
    """
    from lcm import MarkovTransition  # noqa: PLC0415
    from lcm.typing import DiscreteState, FloatND, Period  # noqa: PLC0415

    @categorical(ordered=True)
    class _HealthPre:
        disabled: ScalarInt
        bad: ScalarInt
        good: ScalarInt

    @categorical(ordered=True)
    class _HealthPost:
        bad: ScalarInt
        good: ScalarInt

    @categorical(ordered=False)
    class _RId:
        pre65: ScalarInt
        post65: ScalarInt

    def _health_probs_same(
        period: Period, health: DiscreteState, health_trans_probs: FloatND
    ) -> FloatND:
        return health_trans_probs[period, health]

    def _health_probs_cross(
        period: Period, health: DiscreteState, health_trans_probs_cross: FloatND
    ) -> FloatND:
        return health_trans_probs_cross[period, health]

    pre65 = UserRegime(
        states={
            "health": DiscreteGrid(_HealthPre),
            "wealth": LinSpacedGrid(start=0, stop=10, n_points=5),
        },
        state_transitions={
            "health": {
                "pre65": MarkovTransition(_health_probs_same),
                "post65": MarkovTransition(_health_probs_cross),
            },
            "wealth": lambda wealth: wealth,
        },
        functions={"utility": lambda health, wealth: wealth + health},
        transition=lambda age: jnp.where(age >= 1, _RId.post65, _RId.pre65),
        active=lambda age: age < 1,
    )
    post65 = UserRegime(
        transition=None,
        states={
            "health": DiscreteGrid(_HealthPost),
            "wealth": LinSpacedGrid(start=0, stop=10, n_points=5),
        },
        functions={"utility": lambda health, wealth: wealth + health},
    )
    model = Model(
        regimes={"pre65": pre65, "post65": post65},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RId,
    )

    # Cross-grid transition probs: 3 source states → 2 target states
    index_cross = pd.MultiIndex.from_tuples(
        [
            (0.0, "disabled", "bad"),
            (0.0, "disabled", "good"),
            (0.0, "bad", "bad"),
            (0.0, "bad", "good"),
            (0.0, "good", "bad"),
            (0.0, "good", "good"),
        ],
        names=["age", "health", "next_health"],
    )
    sr_cross = pd.Series([0.65, 0.35, 0.81, 0.19, 0.06, 0.94], index=index_cross)

    params = {
        "pre65": {
            "to_post65_next_health": {"health_trans_probs_cross": sr_cross},
        },
    }
    internal = broadcast_to_template(
        params=params, template=model.get_params_template(), required=False
    )
    result = convert_series_in_params(
        flat_params=internal,
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )

    arr = result["pre65"]["to_post65_next_health__health_trans_probs_cross"]
    # Shape: (n_ages=2, n_source_health=3, n_target_health=2)
    # n_ages=2 because AgeGrid has ages [0, 1]; missing age 1 is NaN-filled.
    assert arr.shape == (2, 3, 2)  # ty: ignore[unresolved-attribute]


def test_resolve_categoricals_includes_derived_when_no_regime_name() -> None:
    """derived_categoricals are included even when regime_name is None."""
    from _lcm.pandas_utils import _resolve_categoricals  # noqa: PLC0415

    model = get_stochastic_model(3)

    @categorical(ordered=False)
    class ExtraVar:
        a: ScalarInt
        b: ScalarInt

    updated_regimes = {
        name: dataclasses.replace(
            r, derived_categoricals={"extra": DiscreteGrid(ExtraVar)}
        )
        for name, r in model.user_regimes.items()
    }
    grids = _resolve_categoricals(
        user_regimes=updated_regimes,
        regime_name=None,
    )
    assert "extra" in grids
