"""A single regime built through the household kernels as the household of `"self"`.

The private switch `_SINGLE_ON_HOUSEHOLD_PATH` in
`_lcm.regime_building.processing` builds every regime declared with a plain
utility function as a household whose one member `"self"` carries Pareto weight
1.0. Its value arrays then carry a trailing member axis of length one; its
simulated frame keeps the short names of a single.
"""

from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.regime_building import processing
from _lcm.regime_building.collective import NO_ROLE
from lcm import Model
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from lcm.typing import FloatND, IntND, RegimeName, UserParamsNode
from lcm_examples import (
    collective_household,
    precautionary_savings_health,
    stochastic_volatility,
)
from tests.conftest import assert_agrees_to_ulp
from tests.solution.test_fingerprint_pin import _MODELS as PINNED_MODELS
from tests.solution.test_fingerprint_pin import _pinned
from tests.test_models import taste_shocks_toy

type InitialConditions = dict[str, FloatND | IntND]
type Case = tuple[Model, dict[str, UserParamsNode], InitialConditions]

_N_SUBJECTS = 8
_SEED = 7


def _precautionary_savings_health() -> Case:
    model = precautionary_savings_health.get_model(retirement_age=20)
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 18.0),
        "wealth": jnp.linspace(5.0, 40.0, _N_SUBJECTS),
        "health": jnp.linspace(0.2, 0.8, _N_SUBJECTS),
        "regime_id": jnp.full(
            _N_SUBJECTS, precautionary_savings_health.RegimeId.working_life
        ),
    }
    return (
        model,
        precautionary_savings_health.get_params(retirement_age=20),
        initial_conditions,
    )


def _stochastic_volatility() -> Case:
    # `get_model` is cached; the wrapped builder makes a model under the switch.
    model = stochastic_volatility.get_model.__wrapped__(
        n_periods=3, wealth_n_points=8, consumption_n_points=8, income_n_points=3
    )
    low_and_high = jnp.arange(_N_SUBJECTS) % 2
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 20.0),
        "wealth": jnp.linspace(2.0, 30.0, _N_SUBJECTS),
        "uncertainty": low_and_high,
        "lagged_uncertainty": low_and_high,
        "regime_id": jnp.full(_N_SUBJECTS, stochastic_volatility.RegimeId.alive),
    }
    return model, stochastic_volatility.get_params(), initial_conditions


def _taste_shocks_toy() -> Case:
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 40.0),
        "wealth": jnp.linspace(1.0, 10.0, _N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, taste_shocks_toy.ToyRegimeId.alive),
    }
    return (
        taste_shocks_toy.get_model(),
        taste_shocks_toy.get_params(scale=0.2),
        initial_conditions,
    )


def _collective_household() -> Case:
    # Half the cohort starts single and half married, the poorest couples
    # included, so the simulation carries both a consented formation and a
    # unilateral dissolution.
    model = collective_household.get_model(
        n_periods=3, wealth_n_points=6, consumption_n_points=6
    )
    n_subjects = 64
    n_single = n_subjects // 2
    is_single = jnp.arange(n_subjects) < n_single
    is_woman = jnp.arange(n_subjects) % 2 == 0
    regime_ids = model.regime_names_to_ids
    member_ids = model.member_names_to_ids
    initial_conditions: InitialConditions = {
        "wealth": jnp.concatenate(
            [jnp.linspace(1.0, 20.0, n_single), jnp.linspace(2.0, 40.0, n_single)]
        ),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.where(
            is_single,
            jnp.where(is_woman, regime_ids["single_f"], regime_ids["single_m"]),
            regime_ids["couple"],
        ).astype(jnp.int32),
        "own_member": jnp.where(
            is_single, NO_ROLE, jnp.where(is_woman, member_ids["f"], member_ids["m"])
        ).astype(jnp.int32),
    }
    return model, collective_household.get_params(), initial_conditions


_CASES: dict[str, Callable[[], Case]] = {
    "collective_household": _collective_household,
    "precautionary_savings_health": _precautionary_savings_health,
    "stochastic_volatility": _stochastic_volatility,
    "taste_shocks_toy": _taste_shocks_toy,
}


def _solve_and_simulate(
    *, name: str, on_household_path: bool
) -> tuple[SolutionResult, SimulationResult]:
    """Build the case afresh with the switch set, then solve and simulate it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(processing, "_SINGLE_ON_HOUSEHOLD_PATH", on_household_path)
        model, params, initial_conditions = _CASES[name]()
        solution = model.solve(params=params, log_level="off")
        simulation = model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            seed=_SEED,
        )
    return solution, simulation


class _Runs(dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]]):
    """Each case solved and simulated on first use, so one failing case fails alone."""

    def __missing__(
        self, key: tuple[str, bool]
    ) -> tuple[SolutionResult, SimulationResult]:
        name, on_household_path = key
        self[key] = _solve_and_simulate(name=name, on_household_path=on_household_path)
        return self[key]


@pytest.fixture(scope="module")
def runs() -> dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]]:
    """Each case solved and simulated once with the switch off and once on."""
    return _Runs()


def _value_shapes(
    solution: SolutionResult,
) -> MappingProxyType[int, MappingProxyType[str, tuple[int, ...]]]:
    return MappingProxyType(
        {
            period: MappingProxyType(
                {regime: tuple(np.shape(v)) for regime, v in values.items()}
            )
            for period, values in solution.values.items()
        }
    )


def _is_declared_household(
    *, solution: SolutionResult, period: int, regime: RegimeName
) -> bool:
    """Whether the switch-off value array already carries the member axis."""
    return "member" in solution.value_array(period=period, regime=regime).dims


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_values_carry_a_member_axis_of_length_one(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """Every single's value array gains one trailing axis of length one.

    A declared household's array keeps its shape.
    """
    solution = runs[name, False][0]
    single = _value_shapes(solution)
    expected = {
        period: {
            regime: shape
            if _is_declared_household(solution=solution, period=period, regime=regime)
            else (*shape, 1)
            for regime, shape in shapes.items()
        }
        for period, shapes in single.items()
    }

    got = _value_shapes(runs[name, True][0])

    assert got == expected


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_names_its_member_self(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """The trailing value axis is the dimension `member` with coordinate `self`."""
    single = runs[name, False][0]
    solution = runs[name, True][0]
    period = min(solution.values)
    regime = min(
        regime
        for regime in single.values[period]
        if not _is_declared_household(solution=single, period=period, regime=regime)
    )

    array = solution.value_array(period=period, regime=regime)

    assert array.coords["member"].values.tolist() == ["self"]


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_simulates_with_the_short_value_column(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """The frame has the single's columns: `value`, no `value_self`, no `own_member`."""
    single = runs[name, False][1].to_dataframe()

    frame = runs[name, True][1].to_dataframe()

    assert list(frame.columns) == list(single.columns)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_simulates_the_same_discrete_choices(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """With one seed, the discrete actions, states and regimes are the single's."""
    single = runs[name, False][1].to_dataframe()
    frame = runs[name, True][1].to_dataframe()
    discrete = [
        column
        for column in single.columns
        if isinstance(single[column].dtype, pd.CategoricalDtype)
    ]

    pd.testing.assert_frame_equal(frame[discrete], single[discrete])


def test_collective_household_cohort_forms_and_dissolves_couples(
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """The compared frames contain a consented formation and a dissolution."""
    frame = (
        runs["collective_household", False][1]
        .to_dataframe()
        .sort_values(["subject_id", "period"])
    )
    regime = frame["regime_name"].astype(str)
    previous = regime.groupby(frame["subject_id"]).shift()
    singles = ["single_f", "single_m"]

    assert (previous.isin(singles) & (regime == "couple")).any()
    assert ((previous == "couple") & regime.isin(singles)).any()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_values_agree_with_the_single_to_16_ulp(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """Every value is within 16 ULP, at the model's value scale, of the single's."""
    solution = runs[name, False][0]
    single = solution.values
    household = runs[name, True][0].values
    magnitude = max(
        float(np.nanmax(np.abs(np.where(np.isfinite(v), v, np.nan))))
        for values in single.values()
        for v in values.values()
        if np.isfinite(v).any()
    )

    for period, values in single.items():
        for regime, expected in values.items():
            got = np.asarray(household[period][regime])
            if not _is_declared_household(
                solution=solution, period=period, regime=regime
            ):
                got = got[..., 0]
            assert_agrees_to_ulp(
                got=got,
                expected=np.asarray(expected),
                n_ulp=16,
                operand_magnitude=magnitude,
                err_msg=f"period {period}, regime {regime!r}",
            )


def test_switch_off_keeps_the_pinned_fingerprint() -> None:
    """With the switch off, a single-agent model keeps its checked-in fingerprint."""
    fingerprint = PINNED_MODELS["multi_regime_normal"]()._model_structure_fingerprint

    assert fingerprint == _pinned()["multi_regime_normal"]


def test_switch_on_gives_the_engine_regime_the_member_self() -> None:
    """The engine regime of a single carries the member axis `("self",)`."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(processing, "_SINGLE_ON_HOUSEHOLD_PATH", True)
        model = taste_shocks_toy.get_model()

    assert model._regimes["alive"].member_axis == ("self",)
