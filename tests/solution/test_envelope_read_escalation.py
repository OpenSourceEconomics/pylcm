"""The envelope spends double-double arithmetic only where it has to.

Ownership is decided between intervals, and the ordinary working-format read
carries a sound bound of its own. Where that bound already separates the leader
from every rival, a sharper read would name the same owner, so the certified read
buys nothing and is not run. Where it does not, the evaluation escalates.

Both halves matter. If it never escalated the decision would rest on rounded
values again; if it always escalated the read would cost an order of magnitude
more than the answer needs.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.upper_envelope.query import _envelope_dense, _SegmentLinks


def _working_dtype() -> np.dtype:
    return np.dtype(jnp.result_type(1.0))


def _links(*, grid, values) -> _SegmentLinks:
    """Consecutive links over one segment, as the envelope builds them."""
    grid_arr = jnp.asarray(grid)
    value_arr = jnp.asarray(values)
    zeros = jnp.zeros(len(grid) - 1)
    return _SegmentLinks(
        left_grid=grid_arr[:-1],
        right_grid=grid_arr[1:],
        left_value=value_arr[:-1],
        right_value=value_arr[1:],
        left_policy=zeros,
        right_policy=zeros,
        left_marginal=zeros,
        right_marginal=zeros,
        live=jnp.ones(len(grid) - 1, dtype=bool),
    )


def _escalates(*, grid, values, query) -> bool:
    """Whether the ordinary read leaves any query's owner undecided."""
    reduction = _envelope_dense(
        links=_links(grid=grid, values=values),
        query=jnp.asarray([query]),
        certified=False,
    )
    return bool(np.asarray(reduction.escalate))


def test_an_ordinary_read_settles_a_well_separated_envelope():
    """Smooth candidates far apart in value do not pay for the certified read."""
    grid = np.linspace(0.5, 4.0, 12)
    assert not _escalates(grid=grid, values=-1.0 / grid, query=2.25)


def test_a_query_on_a_node_settles_without_escalating():
    """A query landing on a node is an exact read, tied with its self-bracket."""
    grid = np.linspace(0.5, 4.0, 12)
    assert not _escalates(grid=grid, values=-1.0 / grid, query=float(grid[5]))


def test_two_branches_within_the_ordinary_read_s_slack_escalate():
    """Rivals closer together than the fast read can resolve force the sharper one.

    Both branches are flat, at a magnitude where the working format's spacing is
    coarse, and they sit a few multiples of that spacing apart — inside the fast
    read's own bound. The fast read cannot say which is higher, so the evaluation
    must not decide on it.
    """
    dtype = _working_dtype()
    magnitude = dtype.type(1.0e6)
    gap = dtype.type(3.0 * np.finfo(dtype).eps * float(magnitude))
    assert gap > 0.0

    zeros = jnp.zeros(2)
    links = _SegmentLinks(
        left_grid=jnp.asarray([0.0, 0.0]),
        right_grid=jnp.asarray([1.0, 1.0]),
        left_value=jnp.asarray([magnitude, magnitude + gap]),
        right_value=jnp.asarray([magnitude, magnitude + gap]),
        left_policy=zeros,
        right_policy=zeros,
        left_marginal=zeros,
        right_marginal=zeros,
        live=jnp.ones(2, dtype=bool),
    )
    reduction = _envelope_dense(links=links, query=jnp.asarray([0.5]), certified=False)
    assert bool(np.asarray(reduction.escalate))
