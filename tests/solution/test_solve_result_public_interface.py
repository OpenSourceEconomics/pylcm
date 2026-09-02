"""The public solve interface returns one labelled result for every model."""

from types import MappingProxyType

from lcm.solver_api import DISSOLUTION_FLAG, SolutionResult
from tests.regime_building.test_collective_regime_simulate import (
    _DISSOLUTION_PARAMS,
    _make_dissolution_model,
)
from tests.test_models.deterministic.dcegm_variants import (
    get_retirement_only_model,
    get_retirement_only_params,
)

_N_PERIODS = 2


def _singleton_model():
    """Return a two-regime model with no collective regime."""
    return get_retirement_only_model(solver="dcegm", n_periods=_N_PERIODS)


def test_singleton_solve_returns_one_labelled_result() -> None:
    model = _singleton_model()
    params = get_retirement_only_params(n_periods=_N_PERIODS)

    result = model.solve(params=params, log_level="off")

    assert isinstance(result, SolutionResult)
    assert isinstance(result.values, MappingProxyType)
    assert not result.replay_artifacts.project(DISSOLUTION_FLAG)


def test_collective_solve_addresses_dissolution_flags_in_the_result() -> None:
    model = _make_dissolution_model()

    result = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
    dissolution_flags = result.replay_artifacts.project(DISSOLUTION_FLAG)

    assert isinstance(result, SolutionResult)
    assert isinstance(dissolution_flags, MappingProxyType)
    assert any(dissolution_flags.values())
