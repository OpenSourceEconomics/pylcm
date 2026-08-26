"""A schedule's declared threshold order must be the ascending one.

The step dispatch builds its jump and floor masks from the *declared* breakpoint
order while the interval partition is built from the sorted thresholds, so the
two agree only while the declaration ascends. Thresholds are free parameters an
estimator moves, so a draw can reorder them at runtime and mark the kink as the
jump — the unified step would then bridge the real cliff and split a continuous
point. A mixed-kind schedule whose thresholds arrive out of declaration order is
refused rather than solved.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution.nbegm import _partition_jumps, _ride_along_jump_config
from _lcm.utils.logging import LogLevel
from lcm.exceptions import InvalidValueFunctionError
from tests.test_models import nbegm_mixed_toy as toy


def _solve(*, cliff: float, exemption: float, log_level: LogLevel):
    model = toy.build_model(
        variant="nbegm", n_liquid=30, n_savings=40, savings_max=24.0
    )
    return model.solve(
        params=toy.build_params(cliff=cliff, exemption=exemption),
        log_level=log_level,
    )


def test_thresholds_in_declared_ascending_order_solve():
    """The declared order `(cliff, exemption)` ascends, so the solve runs."""
    solution = _solve(cliff=6.0, exemption=16.0, log_level="debug")
    assert not np.isnan(np.asarray(solution[0]["alive"])).any()


def test_thresholds_arriving_out_of_declared_order_are_refused():
    """A draw putting the kink below the jump is refused, not solved.

    Swapping the two values leaves the jump mask pointing at the exemption and
    the kink mask at the cliff, so the unified step would build the wrong cases.
    """
    with pytest.raises(InvalidValueFunctionError):
        _solve(cliff=16.0, exemption=6.0, log_level="debug")


def test_a_single_variable_mixed_schedule_recovers_its_jump_positions():
    """A jump index refers to the *sorted* partition, not the declared order.

    The threshold-to-asset preimage divides by a slope of either sign, so one
    schedule on a decreasing derived variable maps ascending thresholds to
    descending asset preimages. Sorting them reverses the declared order, and
    the jump index has to follow.
    """
    jump_flags, n_jumps, _has_jump, static_positions, dynamic_jumps = (
        _ride_along_jump_config(("jump", "continuous_kink"))
    )
    assert static_positions == (0,)
    # The jump's preimage sits above the kink's on the asset axis.
    sorted_preimages, jump_positions = _partition_jumps(
        jnp.asarray([8.0, 3.0]),
        dynamic_jumps=dynamic_jumps,
        jump_flags=jump_flags,
        n_jumps=n_jumps,
        static_jump_positions=static_positions,
    )
    np.testing.assert_allclose(np.asarray(sorted_preimages), [3.0, 8.0])
    assert int(jump_positions[0]) == 1
