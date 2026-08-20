"""A grid-search oracle that enumerates N-NB-EGM's own outer candidate set.

Reaching that stock through an investment action would let the grid search hit
durable levels the nested solver never considers and miss most of the ones it
does, so a nested-vs-brute gap measured that way is dominated by the
candidate-set mismatch. The `brute` variant chooses the post-decision durable
stock itself, on the solver's outer grid, so both arms rank the same candidates.
"""

import numpy as np

from tests.test_models import n_nbegm_toy as toy

_PARAMS = {"discount_factor": 0.95}


def test_the_direct_variant_chooses_the_post_decision_stock_itself() -> None:
    """`brute` acts on `new_illiquid` over the nested solver's outer grid."""
    model = toy.build_model(variant="brute", n_periods=2, illiquid_grid=toy.OUTER_GRID)
    alive = model.user_regimes["alive"]
    assert set(alive.actions) == {"consumption", "new_illiquid"}


def test_the_direct_variant_reaches_exactly_the_outer_grid() -> None:
    """Its durable candidates are the outer grid — no extras, none missing."""
    model = toy.build_model(variant="brute", n_periods=2, illiquid_grid=toy.OUTER_GRID)
    durable_action = model.user_regimes["alive"].actions["new_illiquid"]
    assert durable_action is not None
    np.testing.assert_allclose(
        np.asarray(durable_action.to_jax()), np.asarray(toy.OUTER_GRID.to_jax())
    )


def test_the_keeper_candidate_costs_the_direct_variant_nothing_extra() -> None:
    """Holding the stock is free in both arms, so the keeper is a shared candidate.

    N-NB-EGM's keeper evaluates `s' = Z`; the direct oracle reaches the same
    point as an ordinary grid node. They are the same candidate only if the
    credited cost of standing still is zero.
    """
    stock = toy.OUTER_GRID.to_jax()
    np.testing.assert_allclose(
        np.asarray(toy.credited(illiquid=stock, new_illiquid=stock)),
        np.zeros(stock.shape),
        atol=0.0,
    )
