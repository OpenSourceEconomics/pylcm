"""Tests for lcm.pandas_utils and categorical.to_categorical_dtype."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.error_handling import validate_transition_probs
from lcm.pandas_utils import (
    _build_discrete_grid_lookup,
    array_from_series,
    initial_conditions_from_dataframe,
)
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
from tests.test_models.stochastic import get_model as get_stochastic_model


@categorical(ordered=False)
class Occupation:
    blue_collar: int
    white_collar: int


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
        mild: int
        moderate: int
        severe: int

    result = Severity.to_categorical_dtype()  # ty: ignore[unresolved-attribute]
    assert result.ordered is True
    assert list(result.categories) == ["mild", "moderate", "severe"]


def test_build_discrete_grid_lookup_basic():
    regimes = {
        "a": Regime(
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
        "a": Regime(
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
        sick: int
        healthy: int

    regimes = {
        "a": Regime(
            transition=None,
            states={"health": DiscreteGrid(Health)},
            functions={"utility": lambda: 0.0},
        ),
        "b": Regime(
            transition=None,
            states={"health": DiscreteGrid(HealthAlt)},
            functions={"utility": lambda: 0.0},
        ),
    }
    with pytest.raises(ValueError, match="Inconsistent DiscreteGrid"):
        _build_discrete_grid_lookup(regimes)


def test_continuous_states_and_age():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime": ["working_life", "working_life"],
            "health": ["bad", "good"],
            "wealth": [10.0, 50.0],
            "age": [25.0, 35.0],
        }
    )
    conditions = initial_conditions_from_dataframe(df=df, model=model)
    assert jnp.array_equal(
        conditions["regime"],
        jnp.array([BasicRegimeId.working_life, BasicRegimeId.working_life]),
    )
    assert jnp.allclose(conditions["wealth"], jnp.array([10.0, 50.0]))
    assert jnp.allclose(conditions["age"], jnp.array([25.0, 35.0]))


def test_categorical_string_labels():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime": ["working_life", "retirement"],
            "health": ["bad", "good"],
            "wealth": [10.0, 50.0],
            "age": [25.0, 25.0],
        }
    )
    conditions = initial_conditions_from_dataframe(df=df, model=model)
    assert jnp.array_equal(
        conditions["regime"],
        jnp.array([BasicRegimeId.working_life, BasicRegimeId.retirement]),
    )
    assert jnp.array_equal(conditions["health"], jnp.array([Health.bad, Health.good]))


def test_categorical_pd_categorical_column():
    model = get_basic_model()
    health_dtype = Health.to_categorical_dtype()  # ty: ignore[unresolved-attribute]
    df = pd.DataFrame(
        {
            "regime": ["working_life", "working_life"],
            "health": pd.Categorical(["good", "bad"], dtype=health_dtype),
            "wealth": [10.0, 50.0],
            "age": [25.0, 25.0],
        }
    )
    conditions = initial_conditions_from_dataframe(df=df, model=model)
    assert jnp.array_equal(conditions["health"], jnp.array([Health.good, Health.bad]))


def test_multi_regime():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime": ["working_life", "retirement", "working_life"],
            "health": ["good", "bad", "good"],
            "wealth": [10.0, 50.0, 30.0],
            "age": [25.0, 25.0, 25.0],
        }
    )
    conditions = initial_conditions_from_dataframe(df=df, model=model)
    assert jnp.array_equal(
        conditions["regime"],
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
    with pytest.raises(ValueError, match="'regime' column"):
        initial_conditions_from_dataframe(df=df, model=model)


def test_invalid_regime_name_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime": ["working_life", "nonexistent"],
            "wealth": [10.0, 50.0],
        }
    )
    with pytest.raises(ValueError, match="Invalid regime names"):
        initial_conditions_from_dataframe(df=df, model=model)


def test_invalid_category_label_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime": ["working_life"],
            "health": ["excellent"],
            "wealth": [10.0],
            "age": [25.0],
        }
    )
    with pytest.raises(ValueError, match="Invalid labels"):
        initial_conditions_from_dataframe(df=df, model=model)


def test_empty_dataframe_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {"regime": pd.Series([], dtype=str), "wealth": pd.Series([], dtype=float)}
    )
    with pytest.raises(ValueError, match="empty"):
        initial_conditions_from_dataframe(df=df, model=model)


def test_unknown_column_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime": ["working_life"],
            "health": ["bad"],
            "wealth": [10.0],
            "age": [25.0],
            "subject_id": [42],
        }
    )
    with pytest.raises(ValueError, match="Unknown columns"):
        initial_conditions_from_dataframe(df=df, model=model)


def test_missing_state_column_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime": ["working_life"],
            "age": [25.0],
            # missing "health" and "wealth"
        }
    )
    with pytest.raises(ValueError, match="Missing required"):
        initial_conditions_from_dataframe(df=df, model=model)


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
        "regime": jnp.array([RegimeId.working_life, RegimeId.working_life]),
    }
    result_raw = model.simulate(
        params=params,
        initial_conditions=raw_conditions,
        period_to_regime_to_V_arr=None,
    )

    # DataFrame approach
    df = pd.DataFrame(
        {
            "regime": ["working_life", "working_life"],
            "wealth": ["low", "high"],
            "age": [50.0, 50.0],
        }
    )
    df_conditions = initial_conditions_from_dataframe(df=df, model=model)
    result_df = model.simulate(
        params=params,
        initial_conditions=df_conditions,
        period_to_regime_to_V_arr=None,
    )

    df_raw = result_raw.to_dataframe()
    df_from_df = result_df.to_dataframe()
    pd.testing.assert_frame_equal(df_raw, df_from_df)


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
    """4D transition probs via array_from_series with param_path."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "next_partner", "probs_array"),
    )
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_array_from_series_transition_categorical_labels():
    """Verify specific label-based values in transition probs array."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "next_partner", "probs_array"),
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
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "next_partner", "probs_array"),
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
    with pytest.raises(ValueError, match="level names"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("working_life", "next_partner", "probs_array"),
        )


def test_array_from_series_transition_invalid_label_raises():
    """Invalid categorical label in transition probs raises ValueError."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    # Replace one label with an invalid one
    new_index = series.index.set_levels(["single", "INVALID"], level="partner")
    series.index = new_index
    with pytest.raises(ValueError, match="Invalid labels"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("working_life", "next_partner", "probs_array"),
        )


def test_array_from_series_transition_period_level_raises():
    """Using 'period' instead of 'age' should raise a clear error."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(0, "work", "single", "single")],
        names=["period", "labor_supply", "partner", "next_partner"],
    )
    series = pd.Series([1.0], index=index)
    with pytest.raises(ValueError, match="age"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("working_life", "next_partner", "probs_array"),
        )


def test_array_from_series_transition_duplicate_level_names_raises():
    """Duplicate MultiIndex level names should raise."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(40.0, "work", "single", "single")],
        names=["age", "labor_supply", "labor_supply", "next_partner"],
    )
    series = pd.Series([1.0], index=index)
    with pytest.raises(ValueError, match="duplicate"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("working_life", "next_partner", "probs_array"),
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
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "next_partner", "probs_array"),
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
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "next_partner", "probs_array"),
    )
    # age=40 (period 0), work (0), single (0) → provided
    np.testing.assert_allclose(result[0, 0, 0], jnp.array([0.3, 0.7]), atol=1e-7)
    # age=50 (period 1) → all NaN
    assert jnp.all(jnp.isnan(result[1]))
    # age=60 (period 2) → all NaN
    assert jnp.all(jnp.isnan(result[2]))


def test_validate_transition_probs_valid():
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    validate_transition_probs(
        probs=arr, model=model, regime_name="working_life", state_name="partner"
    )


def test_validate_transition_probs_wrong_shape():
    model = get_stochastic_model(3)
    arr = jnp.ones((2, 2, 2)) / 2  # wrong shape
    with pytest.raises(ValueError, match="shape"):
        validate_transition_probs(
            probs=arr, model=model, regime_name="working_life", state_name="partner"
        )


def test_validate_transition_probs_values_out_of_range():
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    # Set one value negative
    bad_arr = arr.at[0, 0, 0, 0].set(-0.1)  # noqa: PD008
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        validate_transition_probs(
            probs=bad_arr, model=model, regime_name="working_life", state_name="partner"
        )


def test_validate_transition_probs_rows_dont_sum_to_one():
    model = get_stochastic_model(3)
    arr = jnp.ones((3, 2, 2, 2)) * 0.3  # rows sum to 0.6, not 1
    with pytest.raises(ValueError, match="sum to 1"):
        validate_transition_probs(
            probs=arr, model=model, regime_name="working_life", state_name="partner"
        )


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
    """Regime transition probs via array_from_series with param_path."""
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("alive", "next_regime", "probs_array"),
    )
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_array_from_series_regime_transition_reordered_levels():
    """Reordered MultiIndex levels for regime transitions."""
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    series = series.reorder_levels(["next_regime", "health", "age"])
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("alive", "next_regime", "probs_array"),
    )
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_array_from_series_regime_transition_wrong_level_names_raises():
    """Wrong MultiIndex level names for regime transition raise ValueError."""
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    series.index = series.index.set_names(["age", "health", "wrong_name"])
    with pytest.raises(ValueError, match="level names"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("alive", "next_regime", "probs_array"),
        )


def test_array_from_series_regime_transition_invalid_label_raises():
    """Invalid regime label in transition probs raises ValueError."""
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    new_index = series.index.set_levels(["alive", "INVALID"], level="next_regime")
    series.index = new_index
    with pytest.raises(ValueError, match="Invalid labels"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("alive", "next_regime", "probs_array"),
        )


def test_validate_regime_transition_probs_valid():
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    validate_transition_probs(probs=arr, model=model, regime_name="alive")


def test_validate_regime_transition_probs_wrong_shape():
    model = get_regime_markov_model()
    arr = jnp.ones((2, 2)) / 2
    with pytest.raises(ValueError, match="shape"):
        validate_transition_probs(probs=arr, model=model, regime_name="alive")


def test_validate_regime_transition_probs_not_markov_raises():
    model = get_basic_model()
    arr = jnp.ones((3, 2)) / 2
    with pytest.raises(TypeError, match="stochastic regime transition"):
        validate_transition_probs(probs=arr, model=model, regime_name="working_life")


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


def test_array_from_series_3_part_path() -> None:
    """Fully qualified (regime, func, param) path produces correct 4D shape."""
    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "next_partner", "probs_array"),
    )
    assert result.shape == (3, 2, 2, 2)
    # First element: age=40, work, single, single
    assert float(result[0, 0, 0, 0]) == pytest.approx(1.0)


def test_array_from_series_2_part_path_ambiguous_regime() -> None:
    """2-part path with ambiguous regime fails at grid resolution."""
    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    # next_partner exists in both regimes. With no specific regime, action
    # grids (labor_supply) are not discovered -> unrecognised indexing param.
    with pytest.raises(ValueError, match="Unrecognised indexing parameter"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("next_partner", "probs_array"),
        )


def test_array_from_series_scalar_param() -> None:
    """Scalar parameter (no indexing params) returns 1D array from values."""
    model = get_stochastic_model(3)
    # labor_income(is_working, wage) -- is_working is a function output, not
    # a state or action, so wage has no indexing params.
    series = pd.Series([10.0])
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "labor_income", "wage"),
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
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "next_partner", "probs_array"),
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
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "next_partner", "probs_array"),
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
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("working_life", "next_partner", "probs_array"),
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
    with pytest.raises(ValueError, match="Invalid labels"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("working_life", "next_partner", "probs_array"),
        )


def test_array_from_series_wrong_level_names_raises() -> None:
    """Level names that don't match expected indexing params raise ValueError."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(40.0, "work", "single", "single")],
        names=["age", "labor_supply", "wrong_name", "next_partner"],
    )
    series = pd.Series([1.0], index=index)
    with pytest.raises(ValueError, match="level names"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("working_life", "next_partner", "probs_array"),
        )


def test_array_from_series_unknown_param_path_raises() -> None:
    """Nonexistent param_path raises ValueError."""
    model = get_stochastic_model(3)
    series = pd.Series([1.0])
    with pytest.raises(ValueError, match="not found"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("working_life", "nonexistent_func", "some_param"),
        )


def test_array_from_series_unknown_1_part_param_raises() -> None:
    """Nonexistent 1-part param_path raises ValueError."""
    model = get_stochastic_model(3)
    series = pd.Series([1.0])
    with pytest.raises(ValueError, match="No function with parameter"):
        array_from_series(
            sr=series,
            model=model,
            param_path=("totally_fake_param",),
        )


def test_array_from_series_2_part_path_consistent() -> None:
    """Function-level path succeeds when function is unique across regimes."""
    model = get_stochastic_model(3)
    # labor_income only exists in working_life. wage has no indexing params.
    series = pd.Series([10.0])
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("labor_income", "wage"),
    )
    np.testing.assert_allclose(result, jnp.array([10.0]))


def test_array_from_series_1_part_path_consistent() -> None:
    """Model-level path succeeds when param has consistent indexing."""
    model = get_stochastic_model(3)
    # wage only appears in working_life.labor_income — unique, no ambiguity
    series = pd.Series([10.0])
    result = array_from_series(
        sr=series,
        model=model,
        param_path=("wage",),
    )
    np.testing.assert_allclose(result, jnp.array([10.0]))


def test_array_from_series_invalid_path_length_raises() -> None:
    """param_path with 0 or 4+ elements raises ValueError."""
    model = get_stochastic_model(3)
    series = pd.Series([1.0])
    with pytest.raises(ValueError, match="1-3 elements"):
        array_from_series(sr=series, model=model, param_path=())
    with pytest.raises(ValueError, match="1-3 elements"):
        array_from_series(sr=series, model=model, param_path=("a", "b", "c", "d"))


def test_params_from_pandas_function_level_series() -> None:
    """Series at function level is converted to 4D transition prob array."""
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415

    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    params = {
        "working_life": {
            "next_partner": {"probs_array": series},
        },
    }
    result = params_from_pandas(params=params, model=model)
    arr = result["working_life"]["next_partner"]["probs_array"]
    assert arr.shape == (3, 2, 2, 2)
    assert float(arr[0, 0, 0, 0]) == pytest.approx(1.0)


def test_params_from_pandas_model_level_scalar_passthrough() -> None:
    """Scalar values at model level pass through unchanged."""
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415

    model = get_stochastic_model(3)
    params = {"discount_factor": 0.95}
    result = params_from_pandas(params=params, model=model)
    assert result["discount_factor"] == 0.95


def test_params_from_pandas_regime_level_series() -> None:
    """Series at regime level is resolved via template and converted."""
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415

    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    params = {
        "working_life": {
            "probs_array": series,
        },
    }
    result = params_from_pandas(params=params, model=model)
    arr = result["working_life"]["probs_array"]
    assert arr.shape == (3, 2, 2, 2)


def test_params_from_pandas_mixed_dict() -> None:
    """Mix of scalars, arrays, and Series in one params dict."""
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415

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
    result = params_from_pandas(params=params, model=model)
    assert result["discount_factor"] == 0.95
    assert result["working_life"]["utility"]["disutility_of_work"] == 0.5
    assert result["working_life"]["next_partner"]["probs_array"].shape == (3, 2, 2, 2)
    assert result["working_life"]["next_wealth"]["interest_rate"] == 0.05
    np.testing.assert_allclose(
        result["working_life"]["labor_income"]["wage"], jnp.array([10.0])
    )


def test_params_from_pandas_mapping_leaf() -> None:
    """Series inside a MappingLeaf is converted."""
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415
    from lcm.params import MappingLeaf  # noqa: PLC0415

    model = get_stochastic_model(3)
    series = _build_partner_probs_series(model)
    leaf = MappingLeaf({"sub_key": series})
    params = {
        "working_life": {
            "next_partner": {"probs_array": leaf},
        },
    }
    result = params_from_pandas(params=params, model=model)
    converted_leaf = result["working_life"]["next_partner"]["probs_array"]
    assert isinstance(converted_leaf, MappingLeaf)
    arr = converted_leaf.data["sub_key"]
    assert arr.shape == (3, 2, 2, 2)


def test_params_from_pandas_unknown_param_raises() -> None:
    """Unknown param name raises ValueError."""
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415

    model = get_stochastic_model(3)
    params = {"nonexistent_param": pd.Series([1.0])}
    with pytest.raises(ValueError, match="No template match"):
        params_from_pandas(params=params, model=model)


def test_params_from_pandas_with_categoricals() -> None:
    """Derived variable indexing requires explicit categoricals."""
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415

    # In the stochastic model, next_partner(period, labor_supply, partner, probs_array)
    # uses labor_supply — which is a DiscreteGrid action in working_life.
    # Simulate the case where the converter needs extra categoricals by building
    # a model where a function indexes by a variable not in the model grids.
    # Use the existing stochastic model: next_partner indexes by labor_supply,
    # which IS in the working_life action grids. But for the retirement regime,
    # labor_supply is NOT an action — it's a fixed param. So if we provide
    # probs_array for retirement's next_partner (which also indexes by
    # labor_supply in the source code), we need categoricals to resolve it.
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

    # Without categoricals, this should fail (retirement doesn't have
    # labor_supply as an action)
    params = {"retirement": {"next_partner": {"probs_array": series}}}
    with pytest.raises(ValueError, match="Unrecognised indexing parameter"):
        params_from_pandas(params=params, model=model)

    # With categoricals providing the labor_supply grid, it succeeds
    result = params_from_pandas(
        params=params,
        model=model,
        categoricals={"labor_supply": labor_grid},
    )
    arr = result["retirement"]["next_partner"]["probs_array"]
    assert arr.shape == (3, 2, 2, 2)


def test_params_from_pandas_per_target_transition() -> None:
    """Per-target state transitions should be convertible via params_from_pandas."""
    from lcm import AgeGrid, MarkovTransition  # noqa: PLC0415
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415
    from lcm.typing import DiscreteState, FloatND, Period  # noqa: PLC0415

    @categorical(ordered=False)
    class _RId:
        working: int
        retired: int

    def _health_probs(
        period: Period, health: DiscreteState, probs_array: FloatND
    ) -> FloatND:
        return probs_array[period, health]

    def _utility(health: DiscreteState, wealth: float) -> FloatND:
        return wealth + health

    def _next_wealth(wealth: float) -> float:
        return wealth

    working = Regime(
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
    retired = Regime(
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
    result = params_from_pandas(params=params, model=model)
    arr = result["working"]["to_working_next_health"]["probs_array"]
    assert arr.shape == (3, 2, 2)


def test_build_outcome_mapping_qualified_func_name() -> None:
    """`_build_outcome_mapping` should handle qualified names."""
    from lcm.pandas_utils import _build_outcome_mapping  # noqa: PLC0415

    model = get_stochastic_model(3)
    from lcm.pandas_utils import _build_discrete_grid_lookup  # noqa: PLC0415

    grids = _build_discrete_grid_lookup(model.regimes)
    result = _build_outcome_mapping(
        func_name="next_health__working", all_grids=grids, model=model
    )
    assert result.size == 2
    assert result.name == "next_health"


def test_params_from_pandas_structured_categoricals() -> None:
    """Regime-level categoricals should allow different grids per regime."""
    from lcm import AgeGrid  # noqa: PLC0415
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415
    from lcm.typing import FloatND  # noqa: PLC0415

    @categorical(ordered=False)
    class _RId:
        regime_a: int
        regime_b: int

    @categorical(ordered=False)
    class _ChoiceA:
        x: int
        y: int

    @categorical(ordered=False)
    class _ChoiceB:
        x: int
        y: int
        z: int

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

    regime_a = Regime(
        transition=lambda age: jnp.where(age >= 1, _RId.regime_b, _RId.regime_a),
        active=lambda age: age < 1,
        states={"wealth": LinSpacedGrid(start=0, stop=10, n_points=5)},
        state_transitions={"wealth": _next_wealth_sc},
        functions={"utility": func_a, "derived": _derived_a},
    )
    regime_b = Regime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=0, stop=10, n_points=5)},
        functions={"utility": func_b, "derived": _derived_b},
    )
    model = Model(
        regimes={"regime_a": regime_a, "regime_b": regime_b},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RId,
    )

    # "derived" has 2 outcomes in regime_a (_ChoiceA: x,y) and 3 in
    # regime_b (_ChoiceB: x,y,z). Need per-regime categoricals.
    sr_a = pd.Series([1.0, 2.0], index=pd.Index(["x", "y"], name="derived"))
    sr_b = pd.Series([1.0, 2.0, 3.0], index=pd.Index(["x", "y", "z"], name="derived"))

    result_both = params_from_pandas(
        params={
            "regime_a": {"utility": {"rates": sr_a}},
            "regime_b": {"utility": {"rates": sr_b}},
        },
        model=model,
        categoricals={
            "derived": {
                "regime_a": DiscreteGrid(_ChoiceA),
                "regime_b": DiscreteGrid(_ChoiceB),
            },
        },
    )
    assert result_both["regime_a"]["utility"]["rates"].shape == (2,)
    assert result_both["regime_b"]["utility"]["rates"].shape == (3,)


def test_params_from_pandas_runtime_grid_param() -> None:
    """Runtime grid points should be convertible or give a clear error."""
    from lcm import AgeGrid, IrregSpacedGrid  # noqa: PLC0415
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415

    @categorical(ordered=False)
    class _RId:
        alive: int
        dead: int

    alive = Regime(
        transition=lambda age: jnp.where(age >= 1, _RId.dead, _RId.alive),
        active=lambda age: age < 1,
        states={"wealth": IrregSpacedGrid(n_points=4)},
        state_transitions={"wealth": lambda wealth: wealth},
        functions={"utility": lambda wealth: wealth},
    )
    dead = Regime(
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
    result = params_from_pandas(params=params, model=model)
    np.testing.assert_allclose(result["alive"]["wealth"]["points"], sr.to_numpy())


def test_params_from_pandas_sequence_leaf_traversal() -> None:
    """Series inside a SequenceLeaf should be converted to JAX arrays."""
    from lcm.pandas_utils import params_from_pandas  # noqa: PLC0415
    from lcm.params.sequence_leaf import SequenceLeaf  # noqa: PLC0415

    model = get_stochastic_model(3)
    sr = pd.Series([10.0])
    leaf = SequenceLeaf((sr, 42))
    params = {"working_life": {"labor_income": {"wage": leaf}}}
    result = params_from_pandas(params=params, model=model)
    converted = result["working_life"]["labor_income"]["wage"]
    assert isinstance(converted, SequenceLeaf)
    assert not isinstance(converted.data[0], pd.Series)
    np.testing.assert_allclose(converted.data[0], jnp.array([10.0]))


def test_resolve_categoricals_conflict_raises() -> None:
    """Categoricals that conflict with model grids raise ValueError."""
    from lcm.pandas_utils import _resolve_categoricals  # noqa: PLC0415

    model = get_stochastic_model(3)

    @categorical(ordered=False)
    class WrongPartner:
        x: int
        y: int

    conflicting = {"partner": DiscreteGrid(WrongPartner)}
    with pytest.raises(ValueError, match="conflicts with model grid"):
        _resolve_categoricals(
            model=model,
            regime_name="working_life",
            categoricals=conflicting,
        )
