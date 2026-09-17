"""Tests for converting a user DataFrame into canonical initial conditions."""

import jax.numpy as jnp
import pandas as pd
import pytest

from _lcm.pandas_utils import initial_conditions_from_dataframe
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import ScalarFloat, ScalarInt
from tests.test_models.basic_discrete import (
    Health,
)
from tests.test_models.basic_discrete import (
    RegimeId as BasicRegimeId,
)
from tests.test_models.basic_discrete import (
    get_model as get_basic_model,
)
from tests.test_models.processes import get_model as get_process_model
from tests.test_pandas_utils import _get_heterogeneous_health_model, _HetRegimeId


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


def test_process_state_columns_accepted():
    """Process grid columns are accepted as continuous float columns."""
    model = get_process_model(n_periods=4, distribution_type="uniform")
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


def test_process_state_columns_required():
    """DataFrame without process columns raises (process states are required)."""
    model = get_process_model(n_periods=4, distribution_type="uniform")
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
    model = get_model(n_periods=n_periods)
    params = get_params(n_periods=n_periods)

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
    )

    df_raw = result_raw.to_dataframe()
    df_from_df = result_df.to_dataframe()
    pd.testing.assert_frame_equal(df_raw, df_from_df)


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
    """Handle regimes where a state only exists in some regimes.

    `with_status` and `without_status` each transition to `dead` only, via a
    per-target dict — not a bare coarse transition. A bare transition
    declares conservative support over every regime active next period, and
    `without_status` neither carries `status` nor defines an entry law for
    it, so a coarse transition between the two would fail strict
    state-handoff validation. The per-target dict is required here, not an
    arbitrary workaround: it narrows each regime's declared targets to
    `dead`, which needs no `status` handoff.
    """

    @categorical(ordered=False)
    class _Rid:
        with_status: ScalarInt
        without_status: ScalarInt
        dead: ScalarInt

    @categorical(ordered=False)
    class _Status:
        low: ScalarInt
        high: ScalarInt

    def _one_probability() -> ScalarFloat:
        return jnp.float32(1)

    def _utility_with_status(*, wealth: float, status: int) -> float:
        return wealth + status

    def _utility_without_status(wealth: float) -> float:
        return wealth

    with_status = UserRegime(
        transition={"dead": MarkovTransition(_one_probability)},
        states={
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
            "status": DiscreteGrid(category_class=_Status),
        },
        state_transitions={
            "wealth": fixed_transition("wealth"),
            "status": fixed_transition("status"),
        },
        functions={"utility": _utility_with_status},
    )
    without_status = UserRegime(
        transition={"dead": MarkovTransition(_one_probability)},
        states={"wealth": LinSpacedGrid(start=0, stop=100, n_points=5)},
        state_transitions={"wealth": fixed_transition("wealth")},
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


def test_initial_conditions_process_grid_heterogeneous_state_sets() -> None:
    """A process state (income) only present in one regime is NaN-filled elsewhere.

    `earner` and `retiree` each transition to `dead` only, via a per-target
    dict — not a bare coarse transition. A bare transition declares
    conservative support over every regime active next period, and `retiree`
    neither carries `income` nor defines an entry law for it, so a coarse
    transition from `earner` would fail strict state-handoff validation. The
    per-target dict is required here, not an arbitrary workaround: it narrows
    `earner`'s declared targets to `dead`, which needs no `income` handoff.
    """
    from lcm import UniformIIDProcess  # noqa: PLC0415

    @categorical(ordered=False)
    class _Rid:
        earner: ScalarInt
        retiree: ScalarInt
        dead: ScalarInt

    def _one_probability() -> ScalarFloat:
        return jnp.float32(1)

    def _earner_utility(*, wealth: float, income: float) -> float:
        return wealth + income

    def _retiree_utility(wealth: float) -> float:
        return wealth

    earner = UserRegime(
        transition={"dead": MarkovTransition(_one_probability)},
        states={
            "wealth": LinSpacedGrid(start=0, stop=100, n_points=5),
            "income": UniformIIDProcess(n_points=5),
        },
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _earner_utility},
    )
    retiree = UserRegime(
        transition={"dead": MarkovTransition(_one_probability)},
        states={"wealth": LinSpacedGrid(start=0, stop=100, n_points=5)},
        state_transitions={"wealth": fixed_transition("wealth")},
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
