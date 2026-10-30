"""The public solve interface returns one labelled result for every model."""

import lcm.solver_api
import lcm.solvers
from lcm.solver_api import SolutionResult, ValueStore
from tests.regime_building.test_collective_regime_simulate import (
    _DISSOLUTION_PARAMS,
    _make_dissolution_model,
)
from tests.test_models.deterministic.dcegm_variants import (
    get_retirement_only_model,
    get_retirement_only_params,
)

_N_PERIODS = 2

# The type id of the removed solve-time dissolution flag.
_REMOVED_DISSOLUTION_TYPE_ID = "pylcm.collective.dissolution_flag"


def _singleton_model():
    """Return a two-regime model with no collective regime."""
    return get_retirement_only_model(solver="dcegm", n_periods=_N_PERIODS)


def test_singleton_solve_returns_one_labelled_result() -> None:
    model = _singleton_model()
    params = get_retirement_only_params(n_periods=_N_PERIODS)

    result = model.solve(params=params, log_level="off")

    assert isinstance(result, SolutionResult)
    assert isinstance(result.values, ValueStore)


def test_collective_solve_publishes_no_dissolution_flag() -> None:
    """Infeasibility routing is realized in simulation, so solve emits no flag."""
    model = _make_dissolution_model()

    result = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")

    assert isinstance(result, SolutionResult)
    assert all(
        ref.key.type_id != _REMOVED_DISSOLUTION_TYPE_ID
        for store in (
            result.replay_artifacts,
            result.retained_continuations,
            result.auxiliary_artifacts,
        )
        for ref in store
    )


def test_no_dissolution_flag_key_is_exported() -> None:
    assert not hasattr(lcm.solver_api, "DISSOLUTION_FLAG")
    assert not hasattr(lcm.solvers, "DISSOLUTION_FLAG")
