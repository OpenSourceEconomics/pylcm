"""`NBEGM.envelope_arithmetic` reaches the envelope every solve path ends in.

The merged upper envelope decides which candidate owns each liquid query point.
Certified ownership delegates the stored operands to the exact-affine kernel's
deterministic total order; ordinary ownership reads them in the working format.
The solver-level setting selects between them, so an ordinary solve must avoid the
native call and reproduce the certified value where crossings are well separated.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_step import nbegm_per_interval_continuation_step_savings
from _lcm.egm.upper_envelope.query import ComparisonArithmetic
from _lcm.typing import PeriodToRegimeToVArr
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution._crra_preferences import crra_preferences
from tests.test_models import nbegm_ride_along_toy as toy

_CRRA = 2.0
_N_SAVINGS = 40
_N_LIQUID = 30


def _solve(**overrides: object) -> PeriodToRegimeToVArr:
    """Solve the ride-along tax toy with the given NBEGM overrides."""
    model = toy.build_model(variant="nbegm", nbegm_overrides=overrides)
    return model.solve(params=toy.build_params(), log_level="debug")


def _utility_of_action(consumption):
    return consumption ** (1.0 - _CRRA) / (1.0 - _CRRA)


def _inverse_marginal_utility(marginal_continuation):
    return marginal_continuation ** (-1.0 / _CRRA)


def _per_interval_inputs(n_intervals: int) -> dict:
    """Build one ride cell's per-interval continuation step inputs."""
    shift = jnp.linspace(0.0, 1.0, n_intervals)[:, None]
    return {
        "cont_value": -1.0 / jnp.linspace(0.5, 5.0, _N_SAVINGS)[None, :] + shift,
        "cont_marginal": jnp.linspace(2.0, 0.05, _N_SAVINGS)[None, :] + 0.1 * shift,
        "liquid_grid": jnp.linspace(0.1, 30.0, _N_LIQUID),
        "savings_grid": jnp.linspace(0.0, 28.0, _N_SAVINGS),
        "discount_factor": jnp.asarray(0.96),
        "preferences": crra_preferences(_CRRA),
        "coh_slopes": jnp.linspace(1.0, 1.3, n_intervals),
        "coh_intercepts": jnp.linspace(0.5, 2.0, n_intervals),
        "breakpoints": jnp.linspace(2.0, 27.0, n_intervals - 1),
    }


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_the_ordinary_envelope_reproduces_the_certified_value_function():
    """On a well-separated model both arithmetics publish the same values.

    One of the two arms is the certified comparison, so the agreement is only
    observable where the native library is built.
    """
    certified = _solve(envelope_arithmetic="certified")
    ordinary = _solve(envelope_arithmetic="ordinary")
    for period, regime_to_value in certified.items():
        for regime, value in regime_to_value.items():
            np.testing.assert_allclose(
                np.asarray(ordinary[period][regime]),
                np.asarray(value),
                rtol=10.0**-DECIMAL_PRECISION,
                atol=10.0**-DECIMAL_PRECISION,
                equal_nan=True,
                err_msg=f"period={period}, regime={regime}",
            )


def _step_jaxpr(
    *, arithmetic: ComparisonArithmetic, envelope_segment_block_size: int = 0
) -> str:
    """The program one per-interval step stages out, as text."""
    inputs = _per_interval_inputs(n_intervals=6)
    return str(
        jax.make_jaxpr(
            lambda: nbegm_per_interval_continuation_step_savings(
                **inputs,
                envelope_segment_block_size=envelope_segment_block_size,
                arithmetic=arithmetic,
            )
        )()
    )


def test_an_ordinary_step_stages_out_no_exact_kernel_call():
    """Asking a step for the ordinary read keeps the exact kernel out of it.

    The certified comparison is one call into the native exact-affine kernel.
    A step whose declared arithmetic never reached the merge would emit that
    call regardless, so its absence is what says the choice was honoured.
    """
    assert "ffi_call" not in _step_jaxpr(arithmetic="ordinary")


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_a_certified_step_stages_out_the_exact_kernel_call():
    """The same step asked for the certified read does emit that call.

    Without this half the check above would pass on a program that staged out
    nothing at all, and could not tell a honoured choice from a broken trace.
    """
    assert "ffi_call" in _step_jaxpr(arithmetic="certified")


def test_blocking_the_ordinary_read_publishes_what_the_unblocked_read_does():
    """How the candidates are partitioned does not move the ordinary answer.

    The ordinary read takes a maximum over candidates, which no partition of
    them can change: each block offers its own best and only a strictly better
    one replaces the standing winner, so a tie stays with the earlier candidate
    however the blocks fall.
    """
    inputs = _per_interval_inputs(n_intervals=6)
    unblocked = nbegm_per_interval_continuation_step_savings(
        **inputs, envelope_segment_block_size=0, arithmetic="ordinary"
    )
    blocked = nbegm_per_interval_continuation_step_savings(
        **inputs, envelope_segment_block_size=7, arithmetic="ordinary"
    )

    for one, other in zip(unblocked, blocked, strict=True):
        np.testing.assert_array_equal(np.asarray(one), np.asarray(other))
