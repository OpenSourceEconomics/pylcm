"""NBEGM's branch axis streams in blocks without changing the solution.

`branch_batch_size` bounds how many discrete-action branches the two ride-along
cores hold in flight at once: `0` runs the whole branch axis in one vectorized
pass, `1` runs branch-by-branch (memory-minimal). The solved value function is
identical either way — the knob trades peak memory against sequential execution.
"""

from collections.abc import Mapping

import numpy as np

from tests.test_models import nbegm_ride_discrete_toy as toy


def _solve(*, branch_batch_size: int) -> Mapping[int, Mapping]:
    model = toy.build_model(
        variant="nbegm",
        n_liquid=40,
        liquid_max=30.0,
        n_savings=60,
        savings_max=28.0,
        n_consumption=40,
        action_in_costate=True,
        action_in_utility=True,
        action_in_regime_transition=True,
        branch_batch_size=branch_batch_size,
    )
    return model.solve(params=toy.build_params(), log_level="off")


def test_branch_batch_size_one_matches_whole_axis() -> None:
    """Streaming the branch axis one branch at a time yields the same `V` as the
    whole-axis pass, with the action feeding co-state, utility, and transition."""
    whole = _solve(branch_batch_size=0)
    streamed = _solve(branch_batch_size=1)
    assert whole.keys() == streamed.keys()
    for period in whole:
        for regime in whole[period]:
            np.testing.assert_allclose(
                np.asarray(streamed[period][regime]),
                np.asarray(whole[period][regime]),
                rtol=1e-12,
                atol=1e-12,
            )
