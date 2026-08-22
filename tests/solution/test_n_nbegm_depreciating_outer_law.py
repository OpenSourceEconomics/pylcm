"""An N-NB-EGM durable's declared law of motion governs what the next period carries.

The household chooses a durable stock `s'` this period, but only `alpha * s'`
survives into the next period. That factor lives where any law of motion lives —
in `state_transitions` — so the solver has to apply it: the chosen stock is what
the outer margin searches over, `alpha * s'` is what the continuation is read at.

The oracle is the grid-search twin, which searches the same durable choice set
with a dense consumption grid and evaluates the same declared law by ordinary
backward induction. Substituting the raw search node for the law would carry a
stock `1 / alpha` times too large into every continuation, so the two would stop
solving the same model and their gap would widen well beyond what the shared
grid quantization explains.
"""

from itertools import pairwise

import numpy as np

from lcm.typing import ContinuousState
from tests.test_models import n_nbegm_toy as toy

_PARAMS = {"discount_factor": 0.95}

# Share of the chosen durable stock that survives into the next period.
DEPRECIATION = 0.7

# A stock that grows instead — the law is not restricted to `alpha < 1`.
APPRECIATION = 1.2


def _scaled_law(alpha: float):
    """The durable law `Z' = alpha * s'`, in the vocabulary every variant reads."""

    def durable_transition(new_illiquid: ContinuousState) -> ContinuousState:
        return alpha * new_illiquid

    return durable_transition


def _solve_period0_alive(*, variant: str, alpha: float) -> np.ndarray:
    model = toy.build_model(variant=variant, durable_law=_scaled_law(alpha))
    solution = model.solve(params=_PARAMS, log_level="debug")
    return np.asarray(solution[0]["alive"])


def test_n_nbegm_tracks_the_oracle_no_worse_when_the_stock_depreciates():
    """Depreciation does not widen the gap to the grid-search twin.

    The two methods never agree exactly — the nested solver's consumption margin
    is off-grid while the oracle's is quantized — so the question is not how
    large the gap is but whether declaring a law widens it. `alpha = 1` is the
    control: there the declared law is the identity, so a solver that carried
    the raw chosen stock forward would be carrying the right value anyway and
    the gap measures only the shared quantization. Against that baseline the
    depreciating model must not drift, which it does by a factor of several if
    the law is ignored and the two twins stop solving the same model.
    """
    gap_control = np.abs(
        _solve_period0_alive(variant="n_nbegm", alpha=1.0)
        - _solve_period0_alive(variant="brute", alpha=1.0)
    ).mean()
    gap_depreciating = np.abs(
        _solve_period0_alive(variant="n_nbegm", alpha=DEPRECIATION)
        - _solve_period0_alive(variant="brute", alpha=DEPRECIATION)
    ).mean()
    assert float(gap_depreciating) < 2.5 * float(gap_control)


def test_the_value_rises_with_the_share_of_the_stock_that_survives():
    """More of the chosen stock surviving is worth weakly more, everywhere.

    Without this, the agreement test above would still pass if both twins
    ignored `alpha` in the same way — the two would agree on the wrong model.
    The comparison is a strict ordering rather than a tolerance: carrying more
    durable into the next period only ever adds to what the household holds,
    whatever it chose. This is also where `alpha > 1` is covered, since the
    ordering needs no oracle and so no shared choice set.
    """
    values = [
        _solve_period0_alive(variant="n_nbegm", alpha=alpha)
        for alpha in (DEPRECIATION, 1.0, APPRECIATION)
    ]
    for lower, higher in pairwise(values):
        assert np.all(higher >= lower)
        assert np.any(higher > lower)
