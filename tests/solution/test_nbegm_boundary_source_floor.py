"""NB boundary continuation reads use the source's effective savings floor."""

import math
from dataclasses import dataclass
from typing import TypedDict

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_step import nbegm_multi_interval_step
from _lcm.egm.preferences import Preferences
from _lcm.solution.feasibility import BoundInputStatus
from _lcm.solution.nbegm import _NBEGMFloor
from lcm.typing import BoolND, Float1D, FloatND, ScalarFloat
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution._crra_preferences import crra_preferences


class _BoundaryStepArguments(TypedDict):
    next_value: Float1D
    next_marginal: Float1D
    liquid_grid: Float1D
    next_liquid_grid: Float1D
    savings_grid: Float1D
    discount_factor: ScalarFloat
    preferences: Preferences
    next_liquid: Float1D
    marginal_return: Float1D
    coh_slopes: Float1D
    coh_intercepts: Float1D
    breakpoints: Float1D
    boundary_savings_targets: Float1D
    boundary_next_liquid: Float1D


def _source_floor(*, floor__level: ScalarFloat) -> ScalarFloat:
    return floor__level


def _floor_and_nodes() -> tuple[ScalarFloat, Float1D, BoundInputStatus]:
    """Bind the source's qualified floor to the existing shared offsets."""
    floor = _NBEGMFloor(
        lower=_source_floor,
        lower_inputs=("floor__level",),
        offsets=jnp.asarray([0.0, 0.5, 1.0]),
    )
    actual_floor, nodes, status = floor.evaluate_floor(
        pool={"floor__level": jnp.asarray(0.5)}
    )
    np.testing.assert_array_equal(actual_floor, 0.5)
    np.testing.assert_array_equal(nodes, [0.5, 1.0, 1.5])
    np.testing.assert_array_equal(jnp.stack(status), [False, False])
    return actual_floor, nodes, status


@dataclass(frozen=True, kw_only=True)
class _TerminalLogRead:
    """The terminal value log(3a+4) and its savings derivative."""

    source_floor: ScalarFloat
    calls: list[tuple[FloatND, BoolND]]

    # keyword-only-exempt: library-callback=NBEGMContinuationRead
    def __call__(self, savings: FloatND, active: BoolND) -> tuple[FloatND, FloatND]:
        self.calls.append((savings, active))
        safe = jnp.where(active, savings, self.source_floor)
        resources = 3.0 * safe + 4.0
        return jnp.log(resources), 3.0 / resources


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_boundary_corner_reads_the_effective_source_floor() -> None:
    """Saving 0.5 funds c=0.25 and log(11/8) at wealth 0.75."""
    floor, savings, _status = _floor_and_nodes()
    liquid = jnp.asarray([0.75, 1.5, 3.0])
    landings = 3.0 * savings + 4.0
    targets = jnp.asarray([0.5, jnp.nan])
    arguments: _BoundaryStepArguments = {
        "next_value": jnp.log(landings),
        "next_marginal": 1.0 / landings,
        "liquid_grid": liquid,
        "next_liquid_grid": landings,
        "savings_grid": savings,
        "discount_factor": jnp.asarray(1.0),
        "preferences": crra_preferences(crra=1.0),
        "next_liquid": landings,
        "marginal_return": jnp.full_like(savings, 3.0),
        "coh_slopes": jnp.asarray([1.0]),
        "coh_intercepts": jnp.asarray([0.0]),
        "breakpoints": jnp.asarray([], dtype=liquid.dtype),
        "boundary_savings_targets": targets,
        "boundary_next_liquid": 3.0 * targets + 4.0,
    }
    baseline = nbegm_multi_interval_step(**arguments)
    np.testing.assert_array_almost_equal(
        jnp.stack(tuple(channel[:2] for channel in baseline)),
        [[math.log(11.0 / 8.0), math.log(5.5)], [4.0, 1.0], [0.25, 1.0]],
        decimal=DECIMAL_PRECISION,
    )
    read = _TerminalLogRead(source_floor=floor, calls=[])
    actual = nbegm_multi_interval_step(**arguments, continuation_read=read)
    assert len(read.calls) == 2
    np.testing.assert_array_equal(
        jnp.stack(tuple(call[0] for call in read.calls)), targets
    )
    np.testing.assert_array_equal(
        jnp.stack(tuple(call[1] for call in read.calls)), [True, False]
    )
    for channel, reference in zip(actual, baseline, strict=True):
        np.testing.assert_array_almost_equal(
            channel, reference, decimal=DECIMAL_PRECISION
        )
    np.testing.assert_array_almost_equal(
        jnp.stack(tuple(channel[:2] for channel in actual)),
        [[math.log(11.0 / 8.0), math.log(5.5)], [4.0, 1.0], [0.25, 1.0]],
        decimal=DECIMAL_PRECISION,
    )
