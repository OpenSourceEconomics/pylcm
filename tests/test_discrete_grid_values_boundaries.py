"""The user-facing boundaries of a `DiscreteGrid(values=...)` speak its values.

The model has one regime with four discrete states and one discrete action:

- `experience` takes the integer values `0, 4, 10`, whose codes are `0, 1, 2`;
- `level` takes the float values `0.5, 1.0, 2.0` and matches within `1e-6`;
- `rank` takes `range(3)`, so its values and codes are the same numbers and only
  the strict rules can tell them apart;
- `health` is a category class, whose boundaries keep their labels;
- `hours` takes the integer values `0, 20, 40`.

Every state stays where it starts. Utility is `level * hours - hours**2 / 80`
plus terms in the other states, so the chosen hours are `20` at level `0.5` and
`40` at levels `1.0` and `2.0`.
"""

import functools
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.pandas_utils import array_from_series
from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    Model,
    Transition,
    categorical,
)
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidSimulationInputError,
)
from lcm.regime import Regime
from lcm.result import PolicyLookup
from lcm.solver_api import SolutionResult
from lcm.typing import (
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserInitialConditionsInput,
    UserParams,
)
from tests.conftest import DECIMAL_PRECISION

# A Series index label or an expected coordinate: a value, a code or a label.
type _Label = int | float | str

_EXPERIENCE = (0, 4, 10)
_LEVEL = (0.5, 1.0, 2.0)
_HOURS = (0, 20, 40)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _Health:
    bad: ScalarInt
    good: ScalarInt


def _utility(
    *,
    experience: DiscreteState,
    level: DiscreteState,
    rank: DiscreteState,
    health: DiscreteState,
    hours: DiscreteAction,
    bonus: FloatND,
) -> FloatND:
    return (
        level * hours - hours**2 / 80 + 0.01 * experience + 0.1 * rank + bonus[health]
    )


def _stay_experience(*, experience: DiscreteState) -> DiscreteState:
    return experience


def _stay_level(*, level: DiscreteState) -> DiscreteState:
    return level


def _stay_rank(*, rank: DiscreteState) -> DiscreteState:
    return rank


def _stay_health(*, health: DiscreteState) -> DiscreteState:
    return health


def _next_regime(*, age: float) -> ScalarInt:
    return jnp.where(age >= 1, _RegimeId.dead, _RegimeId.alive)


def _model() -> Model:
    alive = Regime(
        states={
            "experience": DiscreteGrid(values=_EXPERIENCE),
            "level": DiscreteGrid(values=_LEVEL, tolerance=1e-6),
            "rank": DiscreteGrid(values=range(3)),
            "health": DiscreteGrid(category_class=_Health),
        },
        actions={"hours": DiscreteGrid(values=_HOURS)},
        state_transitions={
            "experience": _stay_experience,
            "level": _stay_level,
            "rank": _stay_rank,
            "health": _stay_health,
        },
        functions={"utility": _utility},
    )
    dead = Regime(functions={"utility": lambda: 0.0})
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(exact_values=(0, 1, 2)),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        edges={
            "alive": Transition(
                targets={"dead": (0, 1), "alive": (0,)},
                law=DeterministicTransition(func=_next_regime),
            )
        },
    )


_MODEL = _model()

_PARAMS: UserParams = {
    "discount_factor": 0.9,
    "alive": {"utility": {"bonus": jnp.array([0.0, 0.2])}},
}

# Three subjects, one per node of every value grid.
_VALUE_START = {
    "experience": np.array([0, 4, 10]),
    "level": np.array([0.5, 1.0, 2.0]),
    "rank": np.array([0, 1, 2]),
    "health": np.array([0, 1, 1]),
    "age": np.array([0.0, 0.0, 0.0]),
    "regime_id": np.array([_RegimeId.alive] * 3),
}
_CODE_START = {
    **_VALUE_START,
    "experience": np.array([0, 1, 2]),
    "level": np.array([0, 1, 2]),
}


def _simulate(
    *,
    initial_conditions: UserInitialConditionsInput | pd.DataFrame,
    use_categorical_codes: bool = False,
) -> pd.DataFrame:
    result = _MODEL.simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions,
        log_level="off",
        seed=0,
        use_categorical_codes=use_categorical_codes,
    )
    return result.to_dataframe(use_categorical_codes=True)


@functools.cache
def _reference() -> pd.DataFrame:
    """Simulate from the starting nodes given as codes."""
    return _simulate(initial_conditions=_CODE_START, use_categorical_codes=True)


def _first_alive_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["period"] == 0].reset_index(drop=True)


@pytest.fixture(scope="module")
def value_frame() -> pd.DataFrame:
    return _MODEL.simulate(
        params=_PARAMS, initial_conditions=_VALUE_START, log_level="off", seed=0
    ).to_dataframe()


@pytest.mark.parametrize(
    ("column", "categories"),
    [("experience", [0, 4, 10]), ("level", [0.5, 1.0, 2.0]), ("hours", [0, 20, 40])],
)
def test_to_dataframe_value_grid_categories_are_its_values(
    *, value_frame: pd.DataFrame, column: str, categories: list[float]
) -> None:
    """A value-grid column is ordered and lists every value, observed or not."""
    accessor = value_frame[column].cat
    assert (accessor.ordered, accessor.categories.tolist()) == (True, categories)


@pytest.mark.parametrize(
    ("column", "kind"), [("experience", "i"), ("level", "f"), ("hours", "i")]
)
def test_to_dataframe_value_grid_categories_keep_the_numeric_kind(
    *, value_frame: pd.DataFrame, column: str, kind: str
) -> None:
    """Integer grids publish integer categories, float grids float ones."""
    assert value_frame[column].cat.categories.dtype.kind == kind


def test_to_dataframe_value_grid_rows_hold_values() -> None:
    """Each row holds the value of the node its subject occupies."""
    frame = _first_alive_rows(
        _MODEL.simulate(
            params=_PARAMS, initial_conditions=_VALUE_START, log_level="off", seed=0
        ).to_dataframe()
    )
    numbers = frame[["experience", "level", "hours"]].astype(float).to_numpy()
    assert numbers.tolist() == [
        [0.0, 0.5, 20.0],
        [4.0, 1.0, 40.0],
        [10.0, 2.0, 40.0],
    ]


def test_to_dataframe_value_grid_column_casts_to_float_for_arithmetic(
    value_frame: pd.DataFrame,
) -> None:
    """`.astype(float)` turns a value-grid column into numbers for moments."""
    alive = value_frame.loc[value_frame["regime_name"] == "alive"]
    aaae(alive["hours"].astype(float).mean(), (20 + 40 + 40) / 3, DECIMAL_PRECISION)


def test_to_dataframe_label_grid_keeps_its_labels(value_frame: pd.DataFrame) -> None:
    """A category-class column keeps its string labels."""
    assert value_frame["health"].cat.categories.tolist() == ["bad", "good"]


def test_to_dataframe_with_categorical_codes_gives_codes() -> None:
    """`use_categorical_codes=True` gives the codes of value-grid columns."""
    frame = _first_alive_rows(_reference())
    assert frame[["experience", "level", "hours"]].to_numpy().tolist() == [
        [0, 0, 1],
        [1, 1, 2],
        [2, 2, 2],
    ]


def test_initial_values_simulate_as_their_codes() -> None:
    """Initial conditions in values simulate exactly as the same nodes in codes."""
    pd.testing.assert_frame_equal(
        _simulate(initial_conditions=_VALUE_START), _reference()
    )


@pytest.mark.parametrize("container", [pd.Series, jnp.asarray])
def test_initial_values_accept_series_and_jax_arrays(container) -> None:
    """A Series or a JAX array of values is matched like a NumPy array."""
    start = {
        **_VALUE_START,
        "experience": container([0, 4, 10]),
        "level": container([0.5, 1.0, 2.0]),
    }
    pd.testing.assert_frame_equal(_simulate(initial_conditions=start), _reference())


def test_initial_integral_floats_match_an_integer_grid_exactly() -> None:
    """Floats equal to integer values, as a column with gaps holds, match them."""
    start = {**_VALUE_START, "experience": np.array([0.0, 4.0, 10.0])}
    pd.testing.assert_frame_equal(_simulate(initial_conditions=start), _reference())


def test_initial_float_values_match_within_the_grid_tolerance() -> None:
    """A float within the grid's tolerance of a node starts at that node."""
    start = {**_VALUE_START, "level": np.array([0.5 + 4e-7, 1.0 - 4e-7, 2.0])}
    pd.testing.assert_frame_equal(_simulate(initial_conditions=start), _reference())


@pytest.mark.parametrize(
    ("name", "values", "offending"),
    [
        ("experience", np.array([0, 1, 10]), "1"),
        ("experience", np.array([0.0, 4.5, 10.0]), "4.5"),
        ("level", np.array([0.5, 1.1, 2.0]), "1.1"),
        ("level", np.array([0.5, np.nan, 2.0]), "nan"),
        ("rank", np.array([0, 1, 3]), "3"),
    ],
)
def test_initial_values_off_the_grid_raise_naming_the_value(
    *, name: str, values: np.ndarray, offending: str
) -> None:
    """A value on no node raises, naming the state and the value."""
    with pytest.raises(InvalidInitialConditionsError, match=rf"{name}.*{offending}"):
        _simulate(initial_conditions={**_VALUE_START, name: values})


@pytest.mark.parametrize(
    "values", [np.array(["0", "4", "10"]), np.array([True, False, True])]
)
def test_initial_values_must_be_numbers(values: np.ndarray) -> None:
    """Strings and booleans are refused for a value-grid state."""
    with pytest.raises(InvalidInitialConditionsError, match="experience"):
        _simulate(initial_conditions={**_VALUE_START, "experience": values})


def test_initial_values_with_categorical_codes_take_codes() -> None:
    """`use_categorical_codes=True` reads code arrays for value-grid states."""
    pd.testing.assert_frame_equal(
        _simulate(initial_conditions=_CODE_START, use_categorical_codes=True),
        _reference(),
    )


def test_initial_codes_off_the_axis_raise_with_categorical_codes() -> None:
    """With codes, a value-grid value that is no code is refused."""
    with pytest.raises(InvalidInitialConditionsError, match="experience"):
        _simulate(
            initial_conditions={**_CODE_START, "experience": np.array([0, 4, 10])},
            use_categorical_codes=True,
        )


def test_initial_numeric_categorical_round_trips_from_to_dataframe(
    value_frame: pd.DataFrame,
) -> None:
    """The first rows of `to_dataframe` restart the simulation they came from."""
    first = _first_alive_rows(value_frame)
    start = {
        "experience": first["experience"],
        "level": first["level"],
        "rank": first["rank"],
        "health": first["health"].cat.codes.to_numpy(),
        "age": first["age"].to_numpy(dtype=float),
        "regime_id": np.array([_RegimeId.alive] * 3),
    }
    pd.testing.assert_frame_equal(_simulate(initial_conditions=start), _reference())


def test_initial_dataframe_round_trips_from_to_dataframe(
    value_frame: pd.DataFrame,
) -> None:
    """A frame of `to_dataframe` rows is a valid initial-conditions frame."""
    first = _first_alive_rows(value_frame)
    start = first[["regime_name", "age", "experience", "level", "rank", "health"]]
    pd.testing.assert_frame_equal(_simulate(initial_conditions=start), _reference())


def test_initial_dataframe_with_categorical_codes_takes_codes() -> None:
    """A frame of codes is read with `use_categorical_codes=True`."""
    start = pd.DataFrame(
        {
            "regime_name": ["alive"] * 3,
            "age": [0.0] * 3,
            "experience": [0, 1, 2],
            "level": [0, 1, 2],
            "rank": [0, 1, 2],
            "health": [0, 1, 1],
        }
    )
    pd.testing.assert_frame_equal(
        _simulate(initial_conditions=start, use_categorical_codes=True), _reference()
    )


def test_initial_dataframe_off_the_grid_raises_naming_the_value() -> None:
    """A frame value on no node raises, naming the state and the value."""
    start = pd.DataFrame(
        {
            "regime_name": ["alive"] * 3,
            "age": [0.0] * 3,
            "experience": [0, 1, 10],
            "level": [0.5, 1.0, 2.0],
            "rank": [0, 1, 2],
            "health": ["bad", "good", "good"],
        }
    )
    with pytest.raises(InvalidInitialConditionsError, match=r"experience.*1"):
        _simulate(initial_conditions=start)


@pytest.mark.parametrize(
    ("name", "categories"),
    [
        ("level", [0.5, 1.0]),
        ("level", [0.5, 2.0, 1.0]),
        ("level", [0.5, 1.0, 2.0, 3.0]),
        ("experience", ["0", "4", "10"]),
        ("rank", [1, 2, 0]),
        ("rank", [0.0, 1.0, 2.0]),
    ],
)
def test_initial_categorical_must_carry_the_grid_values_in_order(
    *, name: str, categories: list[float] | list[str]
) -> None:
    """A Categorical whose categories are not exactly the grid's values raises."""
    column = pd.Categorical(
        [categories[0], categories[1], categories[1]],
        categories=categories,
        ordered=True,
    )
    with pytest.raises(InvalidInitialConditionsError, match=name):
        _simulate(initial_conditions={**_VALUE_START, name: column})


def test_label_grid_initial_conditions_keep_codes_and_labels() -> None:
    """A category-class state takes codes in a mapping and labels in a frame."""
    start = pd.DataFrame(
        {
            "regime_name": ["alive"] * 3,
            "age": [0.0] * 3,
            "experience": [0, 4, 10],
            "level": [0.5, 1.0, 2.0],
            "rank": [0, 1, 2],
            "health": ["bad", "good", "good"],
        }
    )
    pd.testing.assert_frame_equal(_simulate(initial_conditions=start), _reference())


def _table_utility(
    *, experience: DiscreteState, level: DiscreteState, table: FloatND
) -> FloatND:
    return table[experience, level]


_TABLE_REGIMES = MappingProxyType(
    {
        "alive": Regime(
            states={
                "experience": DiscreteGrid(values=_EXPERIENCE),
                "level": DiscreteGrid(values=_LEVEL, tolerance=1e-6),
            },
            state_transitions={
                "experience": _stay_experience,
                "level": _stay_level,
            },
            functions={"utility": _table_utility},
        )
    }
)

# The table's entry at codes (i, j) is 10 * i + j.
_TABLE = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])


def _table_params(*, experience: list[_Label], level: list[_Label]) -> pd.Series:
    index = pd.MultiIndex.from_product(
        [experience, level], names=["experience", "level"]
    )
    return pd.Series(_TABLE.ravel(), index=index)


def _converted_table(*, sr: pd.Series, use_categorical_codes: bool) -> np.ndarray:
    """Convert a Series for `table`, indexed by its two value-grid levels."""
    return np.asarray(
        array_from_series(
            sr=sr,
            func=_table_utility,
            param_name="table",
            func_name="utility",
            ages=AgeGrid(exact_values=(0, 1, 2)),
            user_regimes=_TABLE_REGIMES,
            regime_names_to_ids=_MODEL.regime_names_to_ids,
            regime_name="alive",
            use_categorical_codes=use_categorical_codes,
        )
    )


@pytest.mark.parametrize(
    ("experience", "level"),
    [
        ([0, 4, 10], [0.5, 1.0, 2.0]),
        ([0, 4, 10], [0.5 + 4e-7, 1.0, 2.0 - 4e-7]),
    ],
)
def test_params_series_index_values_place_the_value_grid_axes(
    *, experience: list[_Label], level: list[_Label]
) -> None:
    """Index labels are grid values, floats matched within the grid tolerance."""
    got = _converted_table(
        sr=_table_params(experience=experience, level=level),
        use_categorical_codes=False,
    )
    aaae(got, _TABLE, DECIMAL_PRECISION)


def test_params_series_index_codes_with_categorical_codes() -> None:
    """`use_categorical_codes=True` reads the index labels as codes."""
    got = _converted_table(
        sr=_table_params(experience=[0, 1, 2], level=[0, 1, 2]),
        use_categorical_codes=True,
    )
    aaae(got, _TABLE, DECIMAL_PRECISION)


@pytest.mark.parametrize(
    ("experience", "level", "offending"),
    [
        ([0, 1, 2], [0.5, 1.0, 2.0], "experience"),
        ([0, 4, 10], [0.5, 1.1, 2.0], "level"),
        (["0", "4", "10"], [0.5, 1.0, 2.0], "experience"),
        ([0, 4, 10], ["0.5", "1.0", "2.0"], "level"),
    ],
)
def test_params_series_index_off_the_grid_raises(
    *, experience: list[_Label], level: list[_Label], offending: str
) -> None:
    """An index label that is no grid value, a string included, raises."""
    with pytest.raises(ValueError, match=offending):
        _converted_table(
            sr=_table_params(experience=experience, level=level),
            use_categorical_codes=False,
        )


def test_params_series_index_values_raise_with_categorical_codes() -> None:
    """With codes, an index label that is no code raises."""
    with pytest.raises(ValueError, match="experience"):
        _converted_table(
            sr=_table_params(experience=[0, 4, 10], level=[0, 1, 2]),
            use_categorical_codes=True,
        )


def test_params_series_label_grid_keeps_string_labels() -> None:
    """A category-class level keeps its string labels, in any order."""
    got = array_from_series(
        sr=pd.Series([0.2, 0.0], index=pd.Index(["good", "bad"], name="health")),
        func=_utility,
        param_name="bonus",
        func_name="utility",
        ages=AgeGrid(exact_values=(0, 1, 2)),
        user_regimes=_MODEL.user_regimes,
        regime_names_to_ids=_MODEL.regime_names_to_ids,
        regime_name="alive",
    )
    aaae(np.asarray(got), [0.0, 0.2], DECIMAL_PRECISION)


@pytest.fixture(scope="module")
def solution() -> SolutionResult:
    return _MODEL.solve(params=_PARAMS, log_level="off")


def _lookup(
    *,
    solution: SolutionResult,
    states: dict[str, np.ndarray],
    action_grids: dict[str, jax.Array] | None = None,
    use_categorical_codes: bool = False,
) -> PolicyLookup:
    return _MODEL.lookup_policy(
        params=_PARAMS,
        solution=solution,
        period=0,
        regime_name="alive",
        states={name: jnp.asarray(value) for name, value in states.items()},
        action_grids=action_grids,
        use_categorical_codes=use_categorical_codes,
    )


def _hours(lookup: PolicyLookup) -> list[int]:
    """Return the chosen hours of each queried row."""
    assert lookup.actions is not None
    return np.asarray(lookup.actions["hours"]).tolist()


_LOOKUP_STATES = {name: _VALUE_START[name] for name in ("experience", "level", "rank")}


def test_lookup_policy_reads_values_and_returns_action_values(
    solution: SolutionResult,
) -> None:
    """States in values give the chosen hours as values."""
    lookup = _lookup(
        solution=solution, states={**_LOOKUP_STATES, "health": _VALUE_START["health"]}
    )
    assert _hours(lookup) == [20, 40, 40]


def test_lookup_policy_with_categorical_codes_uses_codes(
    solution: SolutionResult,
) -> None:
    """`use_categorical_codes=True` reads codes and returns action codes."""
    states = {name: _CODE_START[name] for name in ("experience", "level", "rank")}
    lookup = _lookup(
        solution=solution,
        states={**states, "health": _CODE_START["health"]},
        use_categorical_codes=True,
    )
    assert _hours(lookup) == [1, 2, 2]


def test_lookup_policy_action_grids_in_values(solution: SolutionResult) -> None:
    """A restricted action grid is given and returned in values."""
    lookup = _lookup(
        solution=solution,
        states={**_LOOKUP_STATES, "health": _VALUE_START["health"]},
        action_grids={"hours": jnp.asarray([0, 20])},
    )
    assert _hours(lookup) == [20, 20, 20]


def test_lookup_policy_action_grids_in_codes(solution: SolutionResult) -> None:
    """With codes, a restricted action grid is given and returned in codes."""
    states = {name: _CODE_START[name] for name in ("experience", "level", "rank")}
    lookup = _lookup(
        solution=solution,
        states={**states, "health": _CODE_START["health"]},
        action_grids={"hours": jnp.asarray([0, 1])},
        use_categorical_codes=True,
    )
    assert _hours(lookup) == [1, 1, 1]


def test_lookup_policy_value_off_the_grid_raises(solution: SolutionResult) -> None:
    """A state value on no node raises, naming the state."""
    with pytest.raises(InvalidSimulationInputError, match="experience"):
        _lookup(
            solution=solution,
            states={
                **_LOOKUP_STATES,
                "experience": np.array([0, 1, 10]),
                "health": _VALUE_START["health"],
            },
        )


def test_lookup_policy_action_grid_off_the_grid_raises(
    solution: SolutionResult,
) -> None:
    """An action-grid value on no node raises, naming the action."""
    with pytest.raises(InvalidSimulationInputError, match="hours"):
        _lookup(
            solution=solution,
            states={**_LOOKUP_STATES, "health": _VALUE_START["health"]},
            action_grids={"hours": jnp.asarray([0, 1])},
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("experience", [0, 4, 10]),
        ("level", [0.5, 1.0, 2.0]),
        ("rank", [0, 1, 2]),
        ("health", ["bad", "good"]),
    ],
)
def test_value_array_coordinates_are_values_or_labels(
    *, solution: SolutionResult, name: str, expected: list[_Label]
) -> None:
    """A value grid's coordinates are its values, a label grid's its labels."""
    array = solution.value_array(period=0, regime="alive")
    assert array[name].to_numpy().tolist() == expected


@pytest.mark.parametrize("name", ["experience", "level", "health"])
def test_value_array_coordinates_with_categorical_codes_are_codes(
    *, solution: SolutionResult, name: str
) -> None:
    """`use_categorical_codes=True` gives every discrete coordinate as codes."""
    array = solution.value_array(period=0, regime="alive", use_categorical_codes=True)
    assert array[name].to_numpy().tolist() == list(range(len(array[name])))


def test_value_array_selects_by_value(solution: SolutionResult) -> None:
    """`.sel` picks a value-grid node by its value."""
    array = solution.value_array(period=0, regime="alive")
    np.testing.assert_array_equal(
        array.sel(experience=4, level=2.0),
        np.asarray(solution.values[0]["alive"])[1, 2],
    )


def test_value_array_matches_simulated_values(solution: SolutionResult) -> None:
    """The value at each starting node equals the simulated value there."""
    array = solution.value_array(period=0, regime="alive")
    frame = _first_alive_rows(_reference())
    got = [
        float(array.sel(experience=experience, level=level, rank=rank, health=health))
        for experience, level, rank, health in zip(
            _VALUE_START["experience"],
            _VALUE_START["level"],
            _VALUE_START["rank"],
            ["bad", "good", "good"],
            strict=True,
        )
    ]
    aaae(got, frame["value"].to_numpy(), DECIMAL_PRECISION)


def test_saved_frame_keeps_value_grid_categories(tmp_path) -> None:
    """The Feather table `save` writes keeps a value grid's numeric categories."""
    result = _MODEL.simulate(
        params=_PARAMS, initial_conditions=_VALUE_START, log_level="off", seed=0
    )
    expected = result.to_dataframe()
    result.save(directory=tmp_path)
    saved = pd.read_feather(tmp_path / "simulated_data.arrow")
    pd.testing.assert_frame_equal(saved, expected)
