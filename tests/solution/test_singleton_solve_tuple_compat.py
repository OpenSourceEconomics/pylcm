"""F6 regression: singleton-only `Model.solve()` keeps its legacy return shape.

The collective-regimes extension added `return_simulation_policy` and
`return_dissolution_flags` to the public `Model.solve()` and threaded a third
internal element (the per-period, per-COLLECTIVE-regime dissolution-flag
mapping) through `_solve_compiled`. That third element must never leak into
the return of a model with NO collective regimes ("singleton-only") unless a
caller explicitly opts in via `return_dissolution_flags=True` — existing
callers that call `solve(...)` bare, or that unpack `V, policy =
solve(..., return_simulation_policy=True)`, must see byte-identical shapes to
the pre-extension API.

This is tested directly against a singleton-only (no `stakeholders` regime)
model here. `test_collective_regime_simulate.py::
test_public_model_solve_default_return_shape_is_byte_identical` already pins
the analogous invariant for a model that DOES declare collective regimes
(default path stays bare even then); together the two tests cover both ends
of the "singleton-only" detection this finding is about.
"""

from types import MappingProxyType

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
    """A two-regime (retirement/dead) model with no collective regimes."""
    return get_retirement_only_model("dcegm", _N_PERIODS)


def test_singleton_solve_bare_call_returns_legacy_mapping_not_a_tuple():
    """No flags passed: `solve()` returns the bare value-function mapping.

    This is the shape every pre-extension caller relies on (hundreds of call
    sites across the test suite assign `model.solve(...)` to a single name).
    It must NOT be a tuple, regardless of whether the model has collective
    regimes.
    """
    model = _singleton_model()
    params = get_retirement_only_params(_N_PERIODS)
    got = model.solve(params=params, log_level="off")

    assert not isinstance(got, tuple)
    assert isinstance(got, MappingProxyType)


def test_singleton_solve_with_simulation_policy_returns_legacy_two_tuple():
    """`return_simulation_policy=True` alone: still exactly a 2-tuple.

    Existing callers that unpack `V, policy = model.solve(...,
    return_simulation_policy=True)` must keep working unchanged for a
    singleton-only model — no dissolution-flags element appended.
    """
    model = _singleton_model()
    params = get_retirement_only_params(_N_PERIODS)
    result = model.solve(params=params, log_level="off", return_simulation_policy=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    value_functions, sim_policy = result
    assert isinstance(value_functions, MappingProxyType)
    assert isinstance(sim_policy, MappingProxyType)


def test_singleton_solve_dissolution_flags_opt_in_returns_empty_mapping():
    """Explicitly opting in on a singleton-only model yields empty flags.

    `return_dissolution_flags=True` is honored (the caller asked for it), but
    since there are no collective regimes, every per-period mapping is empty
    rather than containing a per-regime flag array.
    """
    model = _singleton_model()
    params = get_retirement_only_params(_N_PERIODS)
    value_functions, dissolution_flags = model.solve(
        params=params, log_level="off", return_dissolution_flags=True
    )

    assert isinstance(value_functions, MappingProxyType)
    assert isinstance(dissolution_flags, MappingProxyType)
    assert all(len(regime_map) == 0 for regime_map in dissolution_flags.values())


def test_collective_solve_default_call_still_returns_bare_mapping():
    """A model WITH collective regimes keeps the same default (bare) shape.

    Complements the singleton-only checks above: the gate is the explicit
    `return_dissolution_flags` flag, not model structure, so the default path
    is bare for every model.
    """
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")

    assert not isinstance(solution, tuple)
    assert isinstance(solution, MappingProxyType)


def test_collective_solve_dissolution_flags_opt_in_returns_dissolution_aware_shape():
    """A collective model with `return_dissolution_flags=True` is unchanged.

    The collective path must not regress: opting in still surfaces a
    non-empty dissolution-flag mapping.
    """
    model = _make_dissolution_model()
    solution, dissolution_flags = model.solve(
        params=_DISSOLUTION_PARAMS, log_level="off", return_dissolution_flags=True
    )

    assert isinstance(solution, MappingProxyType)
    assert isinstance(dissolution_flags, MappingProxyType)
    assert any(len(regime_map) > 0 for regime_map in dissolution_flags.values()), (
        "expected at least one period's dissolution-flag mapping to be non-empty"
    )
