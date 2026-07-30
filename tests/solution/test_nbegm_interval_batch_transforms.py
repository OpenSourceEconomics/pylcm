"""Interval batching survives every transform and optional path the step offers.

The batch width partitions which intervals solve together. It is not part of the
problem, so it must not change the answer under any JAX transform, nor interact
with the optional cash-on-hand grid, save-to-cliff candidates, or envelope
blocking. Each case here fixes one of those and varies only the batch width,
including widths that leave a remainder and widths wider than the interval count.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import nbegm_step
from _lcm.egm.nbegm_step import nbegm_per_interval_continuation_step_savings

_CRRA = 2.0
_DISCOUNT = 0.96
_N_SAVINGS = 50
_N_LIQUID = 40


def _utility_of_action(consumption):
    return consumption ** (1.0 - _CRRA) / (1.0 - _CRRA)


def _inverse_marginal_utility(marginal_continuation):
    return marginal_continuation ** (-1.0 / _CRRA)


def _build_inputs(n_intervals: int) -> dict:
    base_value = -1.0 / jnp.linspace(0.5, 5.0, _N_SAVINGS)
    base_marginal = jnp.linspace(2.0, 0.05, _N_SAVINGS)
    shift = jnp.linspace(0.0, 1.0, n_intervals)[:, None]
    return {
        "cont_value": base_value[None, :] + shift,
        "cont_marginal": base_marginal[None, :] + 0.1 * shift,
        "liquid_grid": jnp.linspace(0.1, 30.0, _N_LIQUID),
        "savings_grid": jnp.linspace(0.0, 28.0, _N_SAVINGS),
        "discount_factor": jnp.asarray(_DISCOUNT),
        "utility_of_action": _utility_of_action,
        "inverse_marginal_utility": _inverse_marginal_utility,
        "coh_slopes": jnp.linspace(1.0, 1.3, n_intervals),
        "coh_intercepts": jnp.linspace(0.5, 2.0, n_intervals),
        "breakpoints": jnp.linspace(2.0, 27.0, n_intervals - 1),
    }


def _assert_same(got: tuple, expected: tuple) -> None:
    for candidate, reference in zip(got, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(candidate),
            np.asarray(reference),
            atol=1e-6,
            rtol=1e-6,
            equal_nan=True,
        )


def _solve_at(inputs: dict, chunk_size: int, monkeypatch, **extra) -> tuple:
    monkeypatch.setattr(nbegm_step, "_CHUNK_SIZE", chunk_size)
    return nbegm_per_interval_continuation_step_savings(**inputs, **extra)


# Interval counts that divide four, leave one over, leave three over, and fall
# below a batch width of six.
_COUNTS = [4, 5, 7, 3]


@pytest.mark.parametrize("n_intervals", _COUNTS)
@pytest.mark.parametrize("chunk_size", [1, 4, 6])
def test_eager_solve_is_invariant_to_the_batch_width(
    n_intervals, chunk_size, monkeypatch
) -> None:
    """Outside `jit`, the batch width does not change value, marginal, or policy."""
    inputs = _build_inputs(n_intervals)
    with jax.disable_jit():
        reference = _solve_at(inputs, 1, monkeypatch)
        got = _solve_at(inputs, chunk_size, monkeypatch)
    _assert_same(got, reference)


@pytest.mark.parametrize("n_intervals", _COUNTS)
@pytest.mark.parametrize("chunk_size", [1, 4, 6])
def test_jitted_solve_matches_the_eager_solve(
    n_intervals, chunk_size, monkeypatch
) -> None:
    """Compiling the step changes nothing the batch width could interact with."""
    inputs = _build_inputs(n_intervals)
    with jax.disable_jit():
        reference = _solve_at(inputs, chunk_size, monkeypatch)

    monkeypatch.setattr(nbegm_step, "_CHUNK_SIZE", chunk_size)
    arrays = {
        name: value
        for name, value in inputs.items()
        if name not in ("utility_of_action", "inverse_marginal_utility")
    }

    def solve(**kwargs):
        return nbegm_per_interval_continuation_step_savings(
            **kwargs,
            utility_of_action=_utility_of_action,
            inverse_marginal_utility=_inverse_marginal_utility,
        )

    _assert_same(jax.jit(solve)(**arrays), reference)


@pytest.mark.parametrize("chunk_size", [1, 4, 6])
def test_vmapped_solve_matches_the_per_member_solve(chunk_size, monkeypatch) -> None:
    """Mapping the step over a batch of problems agrees with solving them singly.

    `vmap` adds an axis outside the interval axis the batching partitions, so the
    two must not interact: each member's answer is what that member gets alone.
    """
    inputs = _build_inputs(n_intervals=5)
    monkeypatch.setattr(nbegm_step, "_CHUNK_SIZE", chunk_size)
    offsets = jnp.asarray([0.0, 0.25])

    def solve(cont_value):
        return nbegm_per_interval_continuation_step_savings(
            **{**inputs, "cont_value": cont_value}
        )

    stacked = jnp.stack([inputs["cont_value"] + offset for offset in offsets])
    batched = jax.vmap(solve)(stacked)
    for member, offset in enumerate(offsets):
        expected = solve(inputs["cont_value"] + offset)
        _assert_same(tuple(part[member] for part in batched), expected)


@pytest.mark.parametrize("n_intervals", _COUNTS)
@pytest.mark.parametrize("chunk_size", [1, 6])
def test_an_explicit_cash_on_hand_grid_is_invariant_to_the_batch_width(
    n_intervals, chunk_size, monkeypatch
) -> None:
    """Supplying `coh_grid` does not make the answer depend on the batch width."""
    inputs = _build_inputs(n_intervals)
    # True cash-on-hand at each liquid grid point, so one entry per liquid point.
    coh_grid = 1.15 * inputs["liquid_grid"] + 0.75
    reference = _solve_at(inputs, 1, monkeypatch, coh_grid=coh_grid)
    got = _solve_at(inputs, chunk_size, monkeypatch, coh_grid=coh_grid)
    _assert_same(got, reference)


@pytest.mark.parametrize("n_intervals", _COUNTS)
@pytest.mark.parametrize("chunk_size", [1, 6])
def test_save_to_cliff_candidates_are_invariant_to_the_batch_width(
    n_intervals, chunk_size, monkeypatch
) -> None:
    """The extra save-to-cliff candidates keep their own segment block.

    Their block base is derived from the padded interval count, so the batch
    width moves the base — but the candidates it labels, and the value they
    produce, must not move with it.
    """
    inputs = _build_inputs(n_intervals)
    extra_savings = jnp.linspace(0.5, 26.0, 12)
    extra_cont_value = jnp.broadcast_to(
        -1.0 / jnp.linspace(0.6, 4.0, 12), (n_intervals, 12)
    )
    extra = {"extra_savings": extra_savings, "extra_cont_value": extra_cont_value}
    reference = _solve_at(inputs, 1, monkeypatch, **extra)
    got = _solve_at(inputs, chunk_size, monkeypatch, **extra)
    _assert_same(got, reference)


@pytest.mark.parametrize("chunk_size", [1, 4, 6])
@pytest.mark.parametrize("envelope_segment_block_size", [0, 7])
def test_envelope_blocking_and_batch_width_do_not_interact(
    chunk_size, envelope_segment_block_size, monkeypatch
) -> None:
    """Streaming the envelope over segment blocks is orthogonal to interval batching."""
    inputs = _build_inputs(n_intervals=7)
    reference = _solve_at(inputs, 1, monkeypatch, envelope_segment_block_size=0)
    got = _solve_at(
        inputs,
        chunk_size,
        monkeypatch,
        envelope_segment_block_size=envelope_segment_block_size,
    )
    _assert_same(got, reference)
