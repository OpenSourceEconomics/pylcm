"""The exact envelope's workspace is bounded by its chunk, not by the row.

Resolving a row's envelope visits every node cell and each cell can be split by
up to `max_runs` links. Holding all of those splits at once makes the working
set the product of the row count, the cell count and the capacity, which is what
exhausts memory on a large solve long before any value is wrong.

The row is therefore compacted as it is built: cells are visited in ascending
order and their owned sub-cells appended to the finished row, so nothing of size
`n_cells * max_runs` is ever materialized. `envelope_cell_batch_size` selects how
many cells are resolved together, and because cells are independent it changes
the working set without changing anything published.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.segment_envelope import refine_envelope_exact


def _folded_row(n_links: int, n_per_link: int, seed: int = 0):
    """A value correspondence folded into `n_links` overlapping monotone runs."""
    rng = np.random.default_rng(seed=seed)
    grids, policies, values = [], [], []
    for link in range(n_links):
        start = 0.35 * link
        grid = np.linspace(start, start + 1.6, n_per_link)
        slope = 1.0 + 0.7 * link
        intercept = 0.4 * link
        grids.append(grid)
        values.append(intercept + slope * grid)
        policies.append(0.2 + 0.3 * link + 0.05 * rng.standard_normal(n_per_link))
    return (
        jnp.asarray(np.concatenate(grids)),
        jnp.asarray(np.concatenate(policies)),
        jnp.asarray(np.concatenate(values)),
    )


def _shape_counts(jaxpr_text: str, n_cells: int, max_runs: int) -> dict[str, int]:
    """Count arrays whose shape is the full cell-by-capacity product."""
    counts: dict[str, int] = {}
    for dtype in ("f32", "f64", "i32", "bool"):
        for shape in (f"[{n_cells},{max_runs}]", f"[{max_runs},{n_cells}]"):
            token = f"{dtype}{shape}"
            found = jaxpr_text.count(token)
            if found:
                counts[token] = found
    return counts


@pytest.mark.parametrize("cell_batch_size", [None, 3])
def test_no_full_cell_by_capacity_array_is_materialized(cell_batch_size):
    """Nothing of size `n_cells * max_runs` survives into the compiled program."""
    max_runs = 4
    grid, policy, value = _folded_row(n_links=max_runs, n_per_link=2)
    n_cells = int(len(np.unique(np.asarray(grid))) - 1)

    jaxpr = jax.make_jaxpr(
        lambda g, p, v: refine_envelope_exact(
            endog_grid=g,
            policy=p,
            value=v,
            n_refined=24,
            max_runs=max_runs,
            cell_batch_size=cell_batch_size,
        )
    )(grid, policy, value)

    offenders = _shape_counts(str(jaxpr), n_cells, max_runs)
    assert offenders == {}, offenders


@pytest.mark.parametrize("cell_batch_size", [None, 1, 2, 3, 5])
def test_the_published_row_does_not_depend_on_the_chunk(cell_batch_size):
    """Every chunk size publishes the identical row and the identical `n_kept`.

    Cells are independent, so the chunk is a partition of the same computation.
    A difference here would be a defect, never rounding.
    """
    max_runs = 4
    grid, policy, value = _folded_row(n_links=max_runs, n_per_link=3, seed=1)

    def refine(batch):
        return jax.jit(
            lambda g, p, v: refine_envelope_exact(
                endog_grid=g,
                policy=p,
                value=v,
                n_refined=32,
                max_runs=max_runs,
                cell_batch_size=batch,
            )
        )(grid, policy, value)

    reference = refine(None)
    got = refine(cell_batch_size)
    for expected_array, got_array in zip(reference, got, strict=True):
        np.testing.assert_array_equal(np.asarray(got_array), np.asarray(expected_array))
