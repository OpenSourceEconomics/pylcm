"""Normalization of an EGM candidate chain into x-monotone runs.

The upper envelope is exact only once the candidate cloud is split into maximal,
strictly resource-increasing runs of live candidates: those runs are the branches
whose pairwise crossings the envelope has to find. The split is read off the
candidate order the producer emits, which encodes the savings chain, so a fold in
the Euler-inverted grid starts a new run while an ordinary value decrease along a
rising grid does not.
"""

import jax
import jax.numpy as jnp

from _lcm.egm.upper_envelope.topology import count_linked_runs, monotone_run_ids


def _run_ids(grid: list[float], dead: list[bool] | None = None) -> list[int]:
    grid_arr = jnp.asarray(grid)
    dead_arr = (
        jnp.isnan(grid_arr) if dead is None else jnp.asarray(dead, dtype=jnp.bool_)
    )
    return monotone_run_ids(endog_grid=grid_arr, dead=dead_arr).tolist()


def _n_runs(grid: list[float], dead: list[bool] | None = None) -> int:
    grid_arr = jnp.asarray(grid)
    dead_arr = (
        jnp.isnan(grid_arr) if dead is None else jnp.asarray(dead, dtype=jnp.bool_)
    )
    return int(count_linked_runs(endog_grid=grid_arr, dead=dead_arr))


def test_a_strictly_increasing_chain_is_one_run():
    """An unfolded candidate chain forms a single branch."""
    assert _run_ids([1.0, 2.0, 3.0, 4.0]) == [0, 0, 0, 0]
    assert _n_runs([1.0, 2.0, 3.0, 4.0]) == 1


def test_one_action_chain_folds_into_four_runs():
    """A single savings chain can fold repeatedly; each fold starts a new run."""
    resources = [10.0, 14.0, 11.0, 17.0, 12.0, 20.0, 13.0, 23.0]
    assert _run_ids(resources) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert _n_runs(resources) == 4


def test_a_value_decrease_along_a_rising_grid_does_not_split_a_run():
    """Run boundaries follow the resource order, not the value order."""
    # Values fall while resources rise; the candidates stay one connected run.
    assert _run_ids([1.0, 2.0, 3.0]) == [0, 0, 0]


def test_a_repeated_abscissa_breaks_the_run():
    """A zero-width link is not a strictly increasing step, so it starts a run."""
    assert _run_ids([1.0, 2.0, 2.0, 3.0]) == [0, 0, 1, 1]
    assert _n_runs([1.0, 2.0, 2.0, 3.0]) == 2


def test_dead_candidates_are_unlabelled_and_split_their_neighbours():
    """A dead candidate belongs to no run and cannot bridge across itself."""
    grid = [1.0, 2.0, float("nan"), 3.0, 4.0]
    assert _run_ids(grid) == [0, 0, -1, 1, 1]
    assert _n_runs(grid) == 2


def test_an_isolated_live_candidate_contributes_no_linked_run():
    """A singleton carries no link, so it is not a branch for crossing purposes."""
    # 5.0 sits below its predecessor and above nothing: it links to neither side.
    grid = [1.0, 2.0, 3.0, 0.5]
    assert _n_runs(grid) == 1


def test_all_dead_chain_has_no_runs():
    """A fully dead row has no branches at all."""
    grid = [float("nan")] * 4
    assert _n_runs(grid) == 0


def test_run_labelling_is_jit_and_vmap_compatible():
    """Labelling runs under jit and vmap keeps static shapes."""
    grids = jnp.array([[10.0, 14.0, 11.0, 17.0], [1.0, 2.0, 3.0, 4.0]])
    labelled = jax.jit(
        jax.vmap(lambda g: monotone_run_ids(endog_grid=g, dead=jnp.isnan(g)))
    )(grids)
    assert labelled.tolist() == [[0, 0, 1, 1], [0, 0, 0, 0]]
