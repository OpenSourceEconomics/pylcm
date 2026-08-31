"""The solve stops on a candidate it retained but cannot reconstruct.

Every candidate the solve builds sits at a declared grid node, so an inverse
that lands outside the outer state's domain means the declaration and the grids
disagree — a defect in the model, not an awkward realized state. The solve is
therefore the loud phase, and stops rather than publishing a bank simulation
would have to drop from.

Reading the count back is a host transfer, so it is paid only at
`log_level="debug"`. At every other level the candidate is still dropped, and
simulation reports it when it meets it.
"""

from dataclasses import replace

import jax.numpy as jnp
import pytest

from _lcm.solution.nnbegm import (
    _fail_if_the_solve_grid_cannot_reconstruct_a_candidate as _gate,
)
from _lcm.utils.logging import get_logger
from lcm import AgeSpecializedGrid
from lcm.exceptions import UnrepresentableOuterCandidateError
from tests.test_models import n_nbegm_toy


def _masks():
    """Four retained candidates over two states, one of them unreconstructable."""
    live = jnp.ones((2, 2), dtype=bool)
    dropped = jnp.asarray([[False, True], [False, False]])
    return live, dropped


def test_debug_stops_the_solve_and_names_the_counts() -> None:
    """At `log_level="debug"` an unreconstructable candidate stops the solve."""
    live, dropped = _masks()

    with pytest.raises(UnrepresentableOuterCandidateError) as failure:
        _gate(
            logger=get_logger(log_level="debug"),
            dropped=dropped,
            n_live=live,
            regime_name="alive",
            period=1,
        )

    message = str(failure.value)
    assert "alive" in message
    assert "period 1" in message
    assert "4 outer candidates" in message
    assert "reconstruct 1 of them" in message


@pytest.mark.parametrize("log_level", ["warning", "progress", "off"])
def test_every_other_level_stops_the_solve_too(log_level) -> None:
    """An unreconstructable declared node stops the solve at every log level.

    The log level governs diagnostics, and this is not one. The failure is known
    before anything is published, so letting it pass below raise mode would make
    the published policy bank depend on the diagnostic setting -- the same model
    would publish one policy at `log_level="off"` and a different one at
    `"debug"`. A candidate whose target is a declared node is refused wherever
    it is found.
    """
    live, dropped = _masks()

    with pytest.raises(
        UnrepresentableOuterCandidateError, match="reconstruct 1 of them"
    ):
        _gate(
            logger=get_logger(log_level=log_level),
            dropped=dropped,
            n_live=live,
            regime_name="alive",
            period=1,
        )


def test_a_solve_reconstructing_every_candidate_does_not_stop() -> None:
    """Raise mode is quiet when nothing was dropped."""
    live = jnp.ones((2, 2), dtype=bool)

    _gate(
        logger=get_logger(log_level="debug"),
        dropped=jnp.zeros((2, 2), dtype=bool),
        n_live=live,
        regime_name="alive",
        period=1,
    )


def test_a_custom_no_adjustment_target_is_excluded_from_the_loud_phase() -> None:
    """A computed keeper target drops like a realized state, it does not stop the solve.

    The solve raises on a candidate whose target is a DECLARED node, because only
    such a candidate can show that the declared law and the declared grids
    disagree. A regime declaring its own no-adjustment target computes that
    target from the DAG at each solve cell, so it can land off-node for reasons
    that are economically ordinary — a depreciating durable, say. The caller
    excludes those candidates from the mask, and this pins that the gate honours
    the exclusion rather than stopping on them.
    """
    live = jnp.ones((2, 2), dtype=bool)
    dropped = jnp.asarray([[False, True], [False, False]])
    # The caller's mask: the dropped candidate is not at a declared node.
    at_declared_node = jnp.asarray([[False, False], [True, True]])

    _gate(
        logger=get_logger(log_level="debug"),
        dropped=dropped & at_declared_node,
        n_live=live & at_declared_node,
        regime_name="alive",
        period=1,
    )


def test_the_admission_domain_follows_each_period_s_own_outer_grid() -> None:
    """A stock a later age's outer grid holds is admitted at that age.

    The outer state's declared domain is read per period. An age-varying outer
    grid that reaches higher at later ages therefore keeps the candidates whose
    post-decision stock sits in the part only those ages hold, even though the
    first age's grid stops below it. The outer search stays inside every age's
    domain, so the admission domain is the only thing that can reject them.
    """
    base = n_nbegm_toy.ILLIQUID_GRID
    reaches_higher_later = AgeSpecializedGrid(
        build=lambda age: base if age == 20 else replace(base, stop=base.stop + 6.0),
        signature=lambda age: 0.0 if age == 20 else 6.0,
    )

    period_to_regime_to_V_arr = n_nbegm_toy.build_model(
        variant="n_nbegm", n_periods=4, illiquid_grid=reaches_higher_later
    ).solve(params={"discount_factor": 0.95}, log_level="debug")

    assert bool(jnp.isfinite(period_to_regime_to_V_arr[2]["alive"]).all())
