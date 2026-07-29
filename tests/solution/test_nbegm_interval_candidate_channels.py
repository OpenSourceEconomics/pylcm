"""Interval batching changes no live candidate the envelope sees.

The per-interval continuation step hands `envelope_at_query` five parallel
candidate channels — endogenous grid, value, policy, marginal, segment id.
Batching partitions which intervals solve together; it must not touch that
handover. Every live (non-NaN) candidate, its four companion channels, and its
segment id therefore have to be identical whatever the batch width, and the
segment ids have to stay unique per interval so the branch-aware envelope keeps
fusing only candidates that genuinely share a branch.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import nbegm_step
from _lcm.egm.nbegm_step import nbegm_per_interval_continuation_step_savings

_CRRA = 2.0
_DISCOUNT = 0.96
_N_SAVINGS = 40
_N_LIQUID = 30


def _utility_of_action(consumption):
    return consumption ** (1.0 - _CRRA) / (1.0 - _CRRA)


def _inverse_marginal_utility(marginal_continuation):
    return marginal_continuation ** (-1.0 / _CRRA)


def _build_inputs(n_intervals: int) -> dict:
    liquid_grid = jnp.linspace(0.1, 30.0, _N_LIQUID)
    savings_grid = jnp.linspace(0.0, 28.0, _N_SAVINGS)
    breakpoints = jnp.linspace(2.0, 27.0, n_intervals - 1)
    base_value = -1.0 / jnp.linspace(0.5, 5.0, _N_SAVINGS)
    base_marginal = jnp.linspace(2.0, 0.05, _N_SAVINGS)
    shift = jnp.linspace(0.0, 1.0, n_intervals)[:, None]
    return {
        "cont_value": base_value[None, :] + shift,
        "cont_marginal": base_marginal[None, :] + 0.1 * shift,
        "liquid_grid": liquid_grid,
        "savings_grid": savings_grid,
        "discount_factor": jnp.asarray(_DISCOUNT),
        "utility_of_action": _utility_of_action,
        "inverse_marginal_utility": _inverse_marginal_utility,
        "coh_slopes": jnp.linspace(1.0, 1.3, n_intervals),
        "coh_intercepts": jnp.linspace(0.5, 2.0, n_intervals),
        "breakpoints": breakpoints,
    }


def _capture_envelope_inputs(
    inputs: dict, chunk_size: int, monkeypatch
) -> dict[str, np.ndarray]:
    """Solve, returning the candidate channels handed to `envelope_at_query`."""
    captured: dict[str, np.ndarray] = {}
    original: Callable = nbegm_step.envelope_at_query

    def spy(**kwargs):
        for name in ("endog_grid", "policy", "value", "marginal", "segment_id"):
            captured[name] = np.asarray(kwargs[name])
        return original(**kwargs)

    monkeypatch.setattr(nbegm_step, "_CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(nbegm_step, "envelope_at_query", spy)
    nbegm_per_interval_continuation_step_savings(**inputs)
    return captured


def _live_candidates(captured: dict[str, np.ndarray]) -> np.ndarray:
    """Return the live candidates as rows, sorted so batch order cannot matter.

    The segment column is the *partition* the ids induce, densely renumbered in
    order of first appearance, not the raw id. Ids carry no meaning of their own
    — the branch-aware envelope only ever asks whether two candidates share one
    — and the node and cliff families sit at a block base derived from the
    padded interval count, so their raw labels shift with the batch width while
    the grouping they encode does not.
    """
    channels = np.stack(
        [
            captured["endog_grid"],
            captured["value"],
            captured["policy"],
            captured["marginal"],
        ],
        axis=-1,
    )
    live = np.isfinite(channels[:, 0]) & np.isfinite(channels[:, 1])
    _, partition = np.unique(captured["segment_id"][live], return_inverse=True)
    rows = np.column_stack([channels[live], partition])
    order = np.lexsort((rows[:, 1], rows[:, 0], rows[:, 4]))
    return rows[order]


@pytest.mark.parametrize("n_intervals", [3, 4, 5, 7, 8])
@pytest.mark.parametrize("chunk_size", [1, 4, 6])
def test_live_candidate_channels_do_not_depend_on_the_batch_width(
    n_intervals, chunk_size, monkeypatch
) -> None:
    """Every live candidate reaching the envelope is the same at any batch width."""
    inputs = _build_inputs(n_intervals)
    reference = _live_candidates(_capture_envelope_inputs(inputs, 1, monkeypatch))
    got = _live_candidates(_capture_envelope_inputs(inputs, chunk_size, monkeypatch))
    np.testing.assert_allclose(got, reference, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("chunk_size", [1, 4, 6])
def test_each_interval_keeps_a_segment_id_block_of_its_own(
    chunk_size, monkeypatch
) -> None:
    """Interval candidates never share a segment id across intervals.

    The branch-aware envelope fuses candidates that share a segment id, so two
    intervals landing in one block would be merged as if they were one branch.
    """
    inputs = _build_inputs(n_intervals=5)
    captured = _capture_envelope_inputs(inputs, chunk_size, monkeypatch)
    segment = captured["segment_id"]
    endog = captured["endog_grid"]
    live_segments = segment[np.isfinite(endog)]
    assert len(np.unique(live_segments)) > 1
