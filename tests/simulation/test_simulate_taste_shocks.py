"""Spec for taste-shock-consistent simulation (Gumbel-max discrete choices).

When a regime declares `taste_shocks`, simulation draws the discrete action by
adding `scale * Gumbel(0, 1)` noise to the per-discrete-action `Qc` values and
taking the feasibility-masked argmax — so simulated choice frequencies converge
to the softmax probabilities implied by the solve.
"""

import jax.numpy as jnp
import numpy as np

from tests.test_models import taste_shocks_toy

N_SUBJECTS = 20_000


def _reference_work_probability(
    *, wealth: float, scale: float, discount_factor: float
) -> float:
    """Softmax probability of working at a given wealth, from the numpy reference."""
    terminal_wealth = np.linspace(0.0, 12.0, 25)
    consumption = np.linspace(0.5, 5.0, 8)
    v_done = np.log(terminal_wealth + 1.0)

    qc = np.empty(2)
    for d in (0, 1):
        next_w = wealth - consumption + taste_shocks_toy.WAGE * d
        continuation = np.interp(next_w, terminal_wealth, v_done)
        q = (
            np.log(consumption)
            - taste_shocks_toy.KAPPA * d
            + discount_factor * continuation
        )
        feasible = consumption <= wealth
        qc[d] = np.max(np.where(feasible, q, -np.inf))

    shifted = qc - qc.max()
    probs = np.exp(shifted / scale)
    probs /= probs.sum()
    return float(probs[1])


def test_simulated_work_frequency_converges_to_softmax_probability():
    """The simulated work share matches the logit choice probability.

    All subjects start at the same wealth node, so the period-0 choice
    probability is a single number; with 20k subjects the Monte Carlo standard
    error of the frequency is below 0.005.
    """
    scale = 0.2
    discount_factor = 0.95
    initial_wealth = 4.6  # third node of the 6-point wealth grid

    model = taste_shocks_toy.get_model()
    params = taste_shocks_toy.get_params(scale=scale, discount_factor=discount_factor)
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(N_SUBJECTS, 40.0),
            "wealth": jnp.full(N_SUBJECTS, initial_wealth),
            "regime_id": jnp.full(
                N_SUBJECTS, taste_shocks_toy.ToyRegimeId.alive, dtype=jnp.int32
            ),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=5471,
    )

    df = result.to_dataframe(use_labels=False)
    work_freq = df.query("period == 0")["work"].mean()
    expected = _reference_work_probability(
        wealth=initial_wealth, scale=scale, discount_factor=discount_factor
    )
    np.testing.assert_allclose(work_freq, expected, atol=0.015)


def test_taste_shock_draws_are_invariant_to_subject_chunking():
    """Chunked and unchunked simulations draw identical taste shocks.

    Per-subject Gumbel keys are generated for the full population and sliced
    by global subject index, so `subject_batch_size` never changes any
    subject's simulated choices.
    """
    n_subjects = 64
    model = taste_shocks_toy.get_model()
    params = taste_shocks_toy.get_params(scale=0.2, discount_factor=0.95)
    initial_conditions = {
        "age": jnp.full(n_subjects, 40.0),
        "wealth": jnp.full(n_subjects, 4.6),
        "regime_id": jnp.full(
            n_subjects, taste_shocks_toy.ToyRegimeId.alive, dtype=jnp.int32
        ),
    }

    results = {
        batch_size: model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="debug",
            seed=5471,
            subject_batch_size=batch_size,
        ).to_dataframe(use_labels=False)
        for batch_size in (0, 16)
    }

    np.testing.assert_array_equal(
        results[0]["work"].to_numpy(), results[16]["work"].to_numpy()
    )
