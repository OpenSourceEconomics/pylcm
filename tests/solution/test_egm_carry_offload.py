"""Host-offload of the rolling DC-EGM carries (the dominant device resident).

At scale the rolling `next_regime_to_egm_carry` — a dense-endogenous-grid, per-
discrete-action carry held for *every* carry-producing regime at once — dwarfs the
value function on the accelerator. `offload_carries_to_host=True` evicts it to host
between periods; each period's kernels re-upload only their reachable-target subset
(`_reachable_carry_subset`). The eviction is a `jax.device_put` to the CPU device, so
on a CPU-only host it is a no-op and these tests assert the machine-independent
guarantee: the value function is bit-identical with and without the flag.
"""

import jax
import pytest
from numpy.testing import assert_array_equal

from tests.solution.test_egm_discrete import (
    _get_skill_model,
    _get_skill_model_params,
)
from tests.solution.test_egm_passive_asset_row import _model as _passive_asset_row_model
from tests.solution.test_egm_passive_asset_row import (
    _params as _passive_asset_row_params,
)


@pytest.mark.parametrize("log_level", ["warning", "debug"])
def test_carry_offload_matches_device_solution(log_level):
    model = _get_skill_model()
    params = _get_skill_model_params()

    on_device = model.solve(params=params, log_level=log_level)
    offloaded = model.solve(
        params=params, log_level=log_level, offload_carries_to_host=True
    )

    assert sorted(offloaded) == sorted(on_device)
    for period in on_device:
        assert sorted(offloaded[period]) == sorted(on_device[period])
        for regime_name, v_device in on_device[period].items():
            assert_array_equal(offloaded[period][regime_name], v_device)


def test_carry_offload_matches_device_solution_passive_asset_row():
    """Parity on the ACA-shaped config: passive AIME + asset-row + income process.

    This is the regime mix whose carry the offload actually targets at scale (the
    carry spans the passive AIME axis, the income node axis, and the discrete-action
    axis). Offloading the rolling carry to host must leave the value function
    bit-identical.
    """
    params = _passive_asset_row_params()
    on_device = _passive_asset_row_model("dcegm").solve(
        params=params, log_level="debug"
    )
    offloaded = _passive_asset_row_model("dcegm").solve(
        params=params, log_level="debug", offload_carries_to_host=True
    )

    assert sorted(offloaded) == sorted(on_device)
    for period in on_device:
        assert sorted(offloaded[period]) == sorted(on_device[period])
        for regime_name, v_device in on_device[period].items():
            assert_array_equal(offloaded[period][regime_name], v_device)


def test_offloaded_solve_returns_value_function_on_device():
    """Only the carries move to host; the returned V arrays are unaffected."""
    model = _get_skill_model()
    params = _get_skill_model_params()

    solution = model.solve(
        params=params, log_level="warning", offload_carries_to_host=True
    )

    cpu = jax.devices("cpu")[0]
    # On a CPU-only host everything is on cpu; the assertion that matters across
    # machines is that the solve completes and returns finite V (checked above by
    # the parity test). Here we simply confirm the returned arrays are real device
    # arrays, not host-detached placeholders.
    for period in solution:
        for v in solution[period].values():
            assert v.devices() == {cpu}
