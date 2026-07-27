"""Normalization of an EGM candidate chain into x-monotone runs.

An exact upper envelope needs to know which candidates lie on a common branch.
The EGM step hands the envelope a single connected chain per cell — the
constrained run followed by the interior Euler run, in savings order — and the
chain's own order is the topology: consecutive candidates belong to one branch
exactly while resources strictly increase. Where the Euler-inverted grid turns
around, the chain folds and a new branch starts.

Reading the split off the resource order is what makes it exact. A rule that also
splits on a value decrease over-splits a connected fold, and one that infers
connectivity from value alone cannot see a fold at all. Neither the number of
runs nor their extent is bounded by the number of discrete actions: one action
can fold arbitrarily often, so any static capacity for the runs must be validated
against the realized count rather than assumed from the action space.

Dead (NaN-padded) candidates carry no branch: they are labelled `-1` and break
the runs on either side, so no link is ever inferred across a hole.
"""

import jax.numpy as jnp

from lcm.typing import BoolND, Float1D, Int1D, ScalarInt

# Label carried by a dead candidate, which belongs to no run.
NO_RUN: int = -1


def monotone_run_ids(*, endog_grid: Float1D, dead: BoolND) -> Int1D:
    """Label each candidate with the maximal x-monotone run it belongs to.

    Args:
        endog_grid: Candidate endogenous grid points in producer (savings) order.
        dead: Per-candidate dead mask; a dead candidate joins no run.

    Returns:
        Per-candidate run label, consecutive from `0` over the live candidates in
        chain order, with `NO_RUN` where the candidate is dead. A live candidate
        that links to neither neighbour is a run of its own; it spans no interval,
        so `count_linked_runs` does not count it.

    """
    continues = _link_continues_run(endog_grid=endog_grid, dead=dead)
    # A live candidate opens a run unless the link from its predecessor continues
    # one. Dead candidates open nothing, so labels stay gap-free over live runs.
    continues_from_predecessor = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.bool_), continues]
    )
    starts_run = (~dead) & (~continues_from_predecessor)
    run_id = jnp.cumsum(starts_run.astype(jnp.int32)) - 1
    return jnp.where(dead, NO_RUN, run_id).astype(jnp.int32)


def count_linked_runs(*, endog_grid: Float1D, dead: BoolND) -> ScalarInt:
    """Count the runs that carry at least one link.

    A run of a single candidate spans no interval and contributes no value
    segment, so it is not a branch the envelope has to cross against. This count
    is the realized branch count a static capacity must be validated against.

    Args:
        endog_grid: Candidate endogenous grid points in producer (savings) order.
        dead: Per-candidate dead mask; a dead candidate joins no run.

    Returns:
        Number of runs holding at least one live, strictly increasing link.

    """
    continues = _link_continues_run(endog_grid=endog_grid, dead=dead)
    # A linked run is counted at its first link: one that no live link precedes.
    preceded = jnp.concatenate([jnp.zeros((1,), dtype=jnp.bool_), continues[:-1]])
    return jnp.sum(continues & ~preceded, dtype=jnp.int32)


def _link_continues_run(*, endog_grid: Float1D, dead: BoolND) -> BoolND:
    """Return, per consecutive pair, whether it continues the same run.

    A pair continues a run when both endpoints are live and resources strictly
    increase across it. Equal abscissae give a zero-width link, which carries no
    affine value line, so they break the run rather than joining it.
    """
    return (~dead[:-1]) & (~dead[1:]) & (endog_grid[1:] > endog_grid[:-1])
