"""`NBEGM.envelope_arithmetic` reaches the envelope every solve path ends in.

The merged upper envelope decides which candidate owns each liquid query point.
The certified (double-double) comparison is exact and can abstain; the ordinary
comparison reads each candidate in the working format and always decides, at a
small fraction of the arithmetic. The solver-level setting selects between them,
so a solve configured for the ordinary read must actually get it — and must
reproduce the certified value function wherever the candidates are well
separated, which is the regime the setting is meant for.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_step import nbegm_per_interval_continuation_step_savings
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


def test_a_step_carries_its_arithmetic_into_the_envelope_it_merges_with():
    """A step asked for the ordinary read over a blocked envelope is refused.

    The blocked scan carries the certified arithmetic only, and serving it for
    an ordinary request would report the certified cost under the ordinary
    label. The refusal therefore comes from the envelope itself, so a step that
    raises it is one whose arithmetic reached the merge.
    """
    with pytest.raises(ValueError, match="dense reduction only"):
        nbegm_per_interval_continuation_step_savings(
            **_per_interval_inputs(n_intervals=6),
            envelope_segment_block_size=7,
            arithmetic="ordinary",
        )
