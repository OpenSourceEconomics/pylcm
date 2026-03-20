"""Tests for lcm.pandas_utils and categorical.to_categorical_dtype."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype

from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    Regime,
    categorical,
)
from lcm.error_handling import validate_transition_probs
from lcm.pandas_utils import (
    _build_discrete_grid_lookup,
    initial_conditions_from_dataframe,
    transition_probs_from_series,
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
    conditions = initial_conditions_from_dataframe(df, model=model)
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
    conditions = initial_conditions_from_dataframe(df, model=model)
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
    conditions = initial_conditions_from_dataframe(df, model=model)
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
    conditions = initial_conditions_from_dataframe(df, model=model)
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
        initial_conditions_from_dataframe(df, model=model)


def test_invalid_regime_name_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {
            "regime": ["working_life", "nonexistent"],
            "wealth": [10.0, 50.0],
        }
    )
    with pytest.raises(ValueError, match="Invalid regime names"):
        initial_conditions_from_dataframe(df, model=model)


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
        initial_conditions_from_dataframe(df, model=model)


def test_empty_dataframe_raises():
    model = get_basic_model()
    df = pd.DataFrame(
        {"regime": pd.Series([], dtype=str), "wealth": pd.Series([], dtype=float)}
    )
    with pytest.raises(ValueError, match="empty"):
        initial_conditions_from_dataframe(df, model=model)


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
        initial_conditions_from_dataframe(df, model=model)


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
        initial_conditions_from_dataframe(df, model=model)


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
    df_conditions = initial_conditions_from_dataframe(df, model=model)
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


def test_transition_probs_basic_round_trip():
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    result = transition_probs_from_series(
        series=series, model=model, regime_name="working_life"
    )
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_transition_probs_categorical_labels():
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    result = transition_probs_from_series(
        series=series, model=model, regime_name="working_life"
    )
    # Verify specific values by label
    assert float(result[0, 0, 0, 1]) == pytest.approx(
        0.7
    )  # age=40, work, single->partnered
    assert float(result[1, 1, 1, 0]) == pytest.approx(
        0.7
    )  # age=50, retire, partnered->single


def test_transition_probs_reordered_levels():
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    # Reorder levels: put next_partner first, then partner, work, age
    series = series.reorder_levels(["next_partner", "partner", "labor_supply", "age"])
    result = transition_probs_from_series(
        series=series, model=model, regime_name="working_life"
    )
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_transition_probs_not_markov_raises():
    model = get_basic_model()  # health transition is None (fixed), not MarkovTransition
    index = pd.MultiIndex.from_tuples([(25.0, "bad")], names=["age", "next_health"])
    series = pd.Series([1.0], index=index)
    with pytest.raises(TypeError, match="not a MarkovTransition"):
        transition_probs_from_series(
            series=series, model=model, regime_name="working_life"
        )


def test_transition_probs_wrong_level_names_raises():
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    # Rename a level to something wrong
    series.index = series.index.set_names(
        ["age", "labor_supply", "partner", "wrong_name"]
    )
    with pytest.raises(ValueError, match="No 'next_\\*' level"):
        transition_probs_from_series(
            series=series, model=model, regime_name="working_life"
        )


def test_transition_probs_invalid_label_raises():
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    # Replace one label with an invalid one
    new_index = series.index.set_levels(["single", "INVALID"], level="partner")
    series.index = new_index
    with pytest.raises(ValueError, match="Invalid labels"):
        transition_probs_from_series(
            series=series, model=model, regime_name="working_life"
        )


def test_transition_probs_period_level_raises():
    """Using 'period' instead of 'age' should raise a clear error."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(0, "work", "single", "single")],
        names=["period", "labor_supply", "partner", "next_partner"],
    )
    series = pd.Series([1.0], index=index)
    with pytest.raises(ValueError, match="age"):
        transition_probs_from_series(
            series=series, model=model, regime_name="working_life"
        )


def test_transition_probs_duplicate_level_names_raises():
    """Duplicate MultiIndex level names should raise."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(40.0, "work", "single", "single")],
        names=["age", "labor_supply", "labor_supply", "next_partner"],
    )
    series = pd.Series([1.0], index=index)
    with pytest.raises(ValueError, match="duplicate"):
        transition_probs_from_series(
            series=series, model=model, regime_name="working_life"
        )


def test_transition_probs_invalid_age_raises():
    """Age values not on the model's AgeGrid should raise."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    # Replace age level with invalid values
    series.index = series.index.set_codes([0] * len(series), level="age").set_levels(
        [999.0], level="age"
    )
    with pytest.raises(ValueError, match="Invalid age"):
        transition_probs_from_series(
            series=series, model=model, regime_name="working_life"
        )


def test_transition_probs_no_next_level_raises():
    """Series without a 'next_*' MultiIndex level should raise."""
    model = get_stochastic_model(3)
    index = pd.MultiIndex.from_tuples(
        [(40.0, "work", "single")],
        names=["age", "labor_supply", "partner"],
    )
    series = pd.Series([1.0], index=index)
    with pytest.raises(ValueError, match="No 'next_\\*' level"):
        transition_probs_from_series(
            series=series, model=model, regime_name="working_life"
        )


def test_transition_probs_infers_regime_name():
    """regime_name omitted — should infer 'working_life' from model."""
    model = get_stochastic_model(3)
    arr = _make_partner_probs_array()
    series = _array_to_series(arr, model)
    result = transition_probs_from_series(series=series, model=model)
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_transition_probs_per_target_requires_regime_name():
    """Per-target dict with MarkovTransition requires explicit regime_name."""
    from lcm import AgeGrid, MarkovTransition, Model  # noqa: PLC0415

    @categorical(ordered=False)
    class RegimeId:
        working: int
        retired: int

    from lcm.typing import DiscreteState, FloatND  # noqa: PLC0415

    def _health_probs(health: DiscreteState, probs_array: FloatND) -> FloatND:
        return probs_array[health]

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
            "wealth": lambda wealth: wealth,
        },
        functions={"utility": lambda health, wealth: wealth + health},
        transition=lambda age: jnp.where(age >= 1, RegimeId.retired, RegimeId.working),
        active=lambda age: age < 2,
    )
    retired = Regime(
        transition=None,
        states={
            "health": DiscreteGrid(Health),
            "wealth": LinSpacedGrid(start=0, stop=10, n_points=5),
        },
        functions={"utility": lambda health, wealth: wealth + health},
    )

    model = Model(
        regimes={"working": working, "retired": retired},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )

    index = pd.MultiIndex.from_tuples([(0.0, "bad")], names=["age", "next_health"])
    series = pd.Series([1.0], index=index)
    with pytest.raises(TypeError, match="per-target"):
        transition_probs_from_series(series=series, model=model)


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


def test_regime_transition_probs_basic_round_trip():
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    result = transition_probs_from_series(series=series, model=model)
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_regime_transition_probs_reordered_levels():
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    series = series.reorder_levels(["next_regime", "health", "age"])
    result = transition_probs_from_series(series=series, model=model)
    np.testing.assert_allclose(result, arr, atol=1e-7)


def test_regime_transition_probs_not_markov_raises():
    model = get_basic_model()  # deterministic regime transition
    index = pd.MultiIndex.from_tuples(
        [(25.0, "working_life")], names=["age", "next_regime"]
    )
    series = pd.Series([1.0], index=index)
    with pytest.raises(TypeError, match="stochastic regime transition"):
        transition_probs_from_series(
            series=series, model=model, regime_name="working_life"
        )


def test_regime_transition_probs_wrong_level_names_raises():
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    series.index = series.index.set_names(["age", "health", "wrong_name"])
    with pytest.raises(ValueError, match="No 'next_\\*' level"):
        transition_probs_from_series(series=series, model=model, regime_name="alive")


def test_regime_transition_probs_invalid_label_raises():
    model = get_regime_markov_model()
    arr = _make_regime_probs_array()
    series = _regime_array_to_series(arr, model)
    new_index = series.index.set_levels(["alive", "INVALID"], level="next_regime")
    series.index = new_index
    with pytest.raises(ValueError, match="Invalid labels"):
        transition_probs_from_series(series=series, model=model, regime_name="alive")


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
