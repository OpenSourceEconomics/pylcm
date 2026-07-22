"""NBEGM's one-sided mode targets each child value cliff between savings nodes.

A child value jump creates a legitimate one-sided optimum — save exactly to
the cliff's owning side — that generically falls strictly between savings
nodes. The one-sided solver carries explicit save-to-cliff candidates: per
ride cell it maps each child cliff preimage back through the savings-form
liquid law and evaluates both one-sided targets alongside the savings nodes,
so a coarse savings grid does not forfeit the cliff payoff a dense grid's
near-cliff nodes capture.
"""

import jax.numpy as jnp
import numpy as np

import _lcm.solution.nbegm as solvers_mod
from tests.test_models import nbegm_jump_ride_along_toy as toy


def _solve_alive_p0(n_savings: int) -> np.ndarray:
    model = toy.build_model(
        variant="nbegm",
        n_liquid=24,
        liquid_max=30.0,
        n_savings=n_savings,
        savings_max=28.0,
        n_consumption=8,
    )
    solution = model.solve(params=toy.build_params(), log_level="off")
    return np.asarray(solution[0]["alive"])


def test_cliff_candidates_recover_the_save_to_cliff_payoff(monkeypatch):
    """A 12-node solve recovers the cliff payoff its nodes alone forfeit.

    With savings-node candidates only, the coarse solve sits far below the
    720-node solve at the states whose optimum is save-to-cliff; the explicit
    cliff candidates close most of that gap (the remainder is the coarse
    grid's smooth-EGM resolution), and they never lower the value anywhere —
    the envelope only gains candidates.
    """
    dense = _solve_alive_p0(n_savings=720)
    with_candidates = _solve_alive_p0(n_savings=12)

    original = solvers_mod._cliff_savings_targets

    def all_dead_targets(**kwargs):
        return jnp.full_like(original(**kwargs), jnp.nan)

    monkeypatch.setattr(solvers_mod, "_cliff_savings_targets", all_dead_targets)
    nodes_only = _solve_alive_p0(n_savings=12)

    improvement = with_candidates - nodes_only
    assert improvement.min() >= -1e-9
    assert improvement.max() > 0.4

    hugger = np.unravel_index(improvement.argmax(), improvement.shape)
    assert abs(nodes_only[hugger] - dense[hugger]) > 0.7
    assert abs(with_candidates[hugger] - dense[hugger]) < 0.35
