"""Labels of a parameter Series name distinct grid cells, or the conversion raises.

`level` takes the float values `0.5, 1.0, 2.0` and matches within `1e-4`, so the
distinct labels `0.49999` and `0.50001` both name the node `0.5`. Two such labels
in one Series would write one cell twice and leave another cell unwritten; the
conversion must reject them however the rows are ordered, with or without a time
axis and whatever other axes the key has. A label repeated on one axis is ordinary
when the full keys differ.
"""

import itertools
from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.pandas_utils import array_from_series
from lcm import (
    AgeGrid,
    DiscreteGrid,
    InitialNodes,
    Model,
    Periods,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidParamsError
from lcm.regime import Regime
from lcm.typing import DiscreteState, FloatND, ScalarInt

_LEVEL = (0.5, 1.0, 2.0)
_EXPERIENCE = (0, 4, 10)
_ALIASES = (0.5 - 1e-5, 0.5 + 1e-5, 2.0)
_ONE_TO_ONE = (0.5 - 1e-5, 1.0 + 1e-5, 2.0 - 1e-5)
_ENTRIES = (10.0, 20.0, 30.0)
_PERMUTATIONS = list(itertools.permutations(range(3)))


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _dead_utility() -> FloatND:
    return jnp.asarray(0.0)


def _timed_utility(
    *,
    period: ScalarInt,
    level: DiscreteState,
    level_code: DiscreteState,
    table: FloatND,
) -> FloatND:
    return table[period, level_code] + 0.0 * level


def _untimed_utility(
    *, level: DiscreteState, level_code: DiscreteState, table: FloatND
) -> FloatND:
    return table[level_code] + 0.0 * level


def _two_axis_utility(
    *,
    experience: DiscreteState,
    experience_code: DiscreteState,
    level: DiscreteState,
    level_code: DiscreteState,
    table: FloatND,
) -> FloatND:
    return table[experience_code, level_code] + 0.0 * (experience + level)


def _regime(*, utility: Callable[..., FloatND], with_experience: bool = True) -> Regime:
    states = {"level": DiscreteGrid(values=np.array(_LEVEL), tolerance=1e-4)}
    if with_experience:
        states["experience"] = DiscreteGrid(values=np.array(_EXPERIENCE))
    return Regime(
        states=states,
        state_transitions={name: fixed_transition(name) for name in states},
        functions={"utility": utility},
    )


def _model(*, enable_jit: bool) -> Model:
    return Model(
        n_periods=2,
        regimes={
            "alive": _regime(utility=_timed_utility, with_experience=False),
            "dead": Regime(functions={"utility": _dead_utility}),
        },
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={"alive": {"dead": Periods(values=(0,))}},
        enable_jit=enable_jit,
    )


_REGIME_NAMES_TO_IDS = _model(enable_jit=False).regime_names_to_ids


def _convert(
    *,
    utility: Callable[..., FloatND],
    sr: pd.Series,
    use_categorical_codes: bool = False,
) -> np.ndarray:
    return np.asarray(
        array_from_series(
            sr=sr,
            func=utility,
            param_name="table",
            func_name="utility",
            ages=AgeGrid(exact_values=(0, 1)),
            user_regimes=MappingProxyType({"alive": _regime(utility=utility)}),
            regime_names_to_ids=_REGIME_NAMES_TO_IDS,
            regime_name="alive",
            use_categorical_codes=use_categorical_codes,
        )
    )


def _permuted(
    *, values: tuple[float, ...], permutation: tuple[int, ...]
) -> list[float]:
    return [values[i] for i in permutation]


def _timed_series(
    *,
    labels: list[float],
    entries: list[float],
    periods: list[int],
    time_name: str = "age",
) -> pd.Series:
    """One row per (time, label) pair, the labels repeated at every time."""
    index = pd.MultiIndex.from_arrays(
        [
            [period for period in periods for _ in labels],
            [label for _ in periods for label in labels],
        ],
        names=[time_name, "level"],
    )
    return pd.Series([entry for _ in periods for entry in entries], index=index)


def _solve(
    *, labels: list[float], entries: list[float], enable_jit: bool
) -> np.ndarray:
    table = _timed_series(
        labels=labels, entries=entries, periods=[0], time_name="period"
    )
    solution = _model(enable_jit=enable_jit).solve(
        params={"discount_factor": 1.0, "alive": {"utility": {"table": table}}},
        log_level="off",
    )
    return np.asarray(solution.values[0]["alive"])


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("permutation", _PERMUTATIONS)
def test_solve_rejects_two_labels_naming_one_cell(
    *, permutation: tuple[int, ...], enable_jit: bool
) -> None:
    with pytest.raises(InvalidParamsError, match="same grid cell"):
        _solve(
            labels=_permuted(values=_ALIASES, permutation=permutation),
            entries=_permuted(values=_ENTRIES, permutation=permutation),
            enable_jit=enable_jit,
        )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("permutation", _PERMUTATIONS)
def test_solve_places_one_to_one_tolerant_labels_in_node_order(
    *, permutation: tuple[int, ...], enable_jit: bool
) -> None:
    got = _solve(
        labels=_permuted(values=_ONE_TO_ONE, permutation=permutation),
        entries=_permuted(values=_ENTRIES, permutation=permutation),
        enable_jit=enable_jit,
    )
    np.testing.assert_array_equal(got, _ENTRIES)


@pytest.mark.parametrize("permutation", _PERMUTATIONS)
def test_timed_series_at_every_period_rejects_two_labels_naming_one_cell(
    permutation: tuple[int, ...],
) -> None:
    sr = _timed_series(
        labels=_permuted(values=_ALIASES, permutation=permutation),
        entries=_permuted(values=_ENTRIES, permutation=permutation),
        periods=[0, 1],
    )
    with pytest.raises(InvalidParamsError, match="same grid cell"):
        _convert(utility=_timed_utility, sr=sr)


def test_timed_series_ignores_two_labels_naming_one_cell_at_a_surplus_period() -> None:
    """Rows at a time the model does not have are dropped before keys are checked."""
    valid = _timed_series(labels=list(_LEVEL), entries=list(_ENTRIES), periods=[0, 1])
    surplus = _timed_series(labels=list(_ALIASES), entries=[0.0] * 3, periods=[7])
    got = _convert(utility=_timed_utility, sr=pd.concat([valid, surplus]))
    np.testing.assert_array_equal(got, [_ENTRIES, _ENTRIES])


@pytest.mark.parametrize("permutation", _PERMUTATIONS)
def test_untimed_series_rejects_two_labels_naming_one_cell(
    permutation: tuple[int, ...],
) -> None:
    sr = pd.Series(
        _permuted(values=_ENTRIES, permutation=permutation),
        index=pd.Index(
            _permuted(values=_ALIASES, permutation=permutation), name="level"
        ),
    )
    with pytest.raises(InvalidParamsError, match="same grid cell"):
        _convert(utility=_untimed_utility, sr=sr)


@pytest.mark.parametrize("permutation", _PERMUTATIONS)
def test_untimed_series_places_one_to_one_tolerant_labels_in_node_order(
    permutation: tuple[int, ...],
) -> None:
    sr = pd.Series(
        _permuted(values=_ENTRIES, permutation=permutation),
        index=pd.Index(
            _permuted(values=_ONE_TO_ONE, permutation=permutation), name="level"
        ),
    )
    np.testing.assert_array_equal(_convert(utility=_untimed_utility, sr=sr), _ENTRIES)


def _two_axis_series(levels: list[float]) -> pd.Series:
    index = pd.MultiIndex.from_product(
        [list(_EXPERIENCE), levels], names=["experience", "level"]
    )
    return pd.Series(np.arange(len(index), dtype=float), index=index)


def test_two_axis_series_rejects_two_full_keys_naming_one_cell() -> None:
    """Two rows differing only in aliased `level` labels name one full cell."""
    with pytest.raises(InvalidParamsError, match="same grid cell"):
        _convert(utility=_two_axis_utility, sr=_two_axis_series(list(_ALIASES)))


def test_two_axis_series_accepts_labels_repeated_on_one_axis() -> None:
    """Each `level` label recurs once per `experience` node; the full keys differ."""
    got = _convert(utility=_two_axis_utility, sr=_two_axis_series(list(_ONE_TO_ONE)))
    np.testing.assert_array_equal(got, np.arange(9.0).reshape(3, 3))


def test_untimed_series_rejects_a_repeated_code_with_categorical_codes() -> None:
    sr = pd.Series(list(_ENTRIES), index=pd.Index([0, 0, 2], name="level"))
    with pytest.raises(InvalidParamsError, match="same grid cell"):
        _convert(utility=_untimed_utility, sr=sr, use_categorical_codes=True)
