"""Opt-in smoothing diagnostic: per-subject smoothed choice probabilities.

A post-hoc pass over a completed simulation. For every realized subject-period in
a regime that offers `choice_action`, it recomputes `Q_arr`/`F_arr` at the
realized state (from the regime's retained `Q_and_F`) and returns the τ-smoothed
probability of each level of `choice_action`
(`smoothed_choice_probabilities`).

It reads only the finished `SimulationResult` and never touches the forward
simulation, so it changes no production output. Its purpose is the
standard-errors diagnostic: contrasting these smoothed choice-indicator moments
with the hard ones shows whether a flat moment Jacobian column is an artifact of
the hard `argmax` rather than weak identification.
"""

from collections.abc import Mapping
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd

from _lcm.engine import PeriodRegimeSimulationData, Regime
from _lcm.regime_building.max_Q_over_a import get_smoothed_choice_probs_Q_over_a
from _lcm.simulation.transitions import create_regime_state_action_space
from _lcm.typing import (
    ActionName,
    FlatParams,
    FlatRegimeParams,
    QAndFFunction,
    RegimeName,
)
from _lcm.utils.dispatchers import simulation_spacemap
from lcm.ages import AgeGrid
from lcm.typing import FloatND, ScalarFloat, ScalarInt


def compute_inclusive_value_panel(
    *,
    raw_results: Mapping[RegimeName, Mapping[int, PeriodRegimeSimulationData]],
    regimes: Mapping[RegimeName, Regime],
    flat_params: FlatParams,
    period_to_regime_to_V_arr: Mapping[int, Mapping[RegimeName, FloatND]],
    ages: AgeGrid,
    tau: float,
    choice_action: ActionName,
) -> pd.DataFrame:
    """Compute smoothed choice probabilities over `choice_action` per subject-period.

    Args:
        raw_results: Per-regime, per-period raw simulation data (realized states,
            regime membership) from a finished simulation.
        regimes: Canonical regimes; their `simulate_functions.Q_and_F_per_period`
            is recomputed at the realized states.
        flat_params: Model parameters for every regime.
        period_to_regime_to_V_arr: Value-function arrays for all periods/regimes.
        ages: The model's age grid.
        tau: Smoothing temperature; must be strictly positive.
        choice_action: The discrete action whose level probabilities are computed.

    Returns:
        A long DataFrame with columns `subject`, `period`, `regime`, `level`, and
        `probability`, one row per (in-regime subject-period, choice level).

    """
    rows = []
    for regime_name, regime in regimes.items():
        period_to_Q_and_F = regime.simulate_functions.Q_and_F_per_period
        if period_to_Q_and_F is None:
            continue
        for period, period_data in raw_results.get(regime_name, {}).items():
            probabilities = _smoothed_probs_for_period(
                regime=regime,
                Q_and_F=period_to_Q_and_F[period],
                regime_params=flat_params[regime_name],
                period_data=period_data,
                next_regime_to_V_arr=period_to_regime_to_V_arr.get(
                    period + 1, MappingProxyType({})
                ),
                age=ages.values[period],  # noqa: PD011 — AgeGrid array, not pandas
                period=period,
                tau=tau,
                choice_action=choice_action,
            )
            if probabilities is None:
                continue
            rows.append(
                _tidy_period_rows(
                    probabilities=np.asarray(probabilities),
                    in_regime=np.asarray(period_data.in_regime),
                    period=period,
                    regime_name=regime_name,
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=["subject", "period", "regime", "level", "probability"]
        )
    return pd.concat(rows, ignore_index=True)


def _smoothed_probs_for_period(
    *,
    regime: Regime,
    Q_and_F: QAndFFunction,
    regime_params: FlatRegimeParams,
    period_data: PeriodRegimeSimulationData,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    age: ScalarFloat | ScalarInt,
    period: int,
    tau: float,
    choice_action: ActionName,
) -> FloatND | None:
    """Smoothed choice probabilities for one regime-period, or None if N/A."""
    state_action_space = create_regime_state_action_space(
        regime=regime,
        regime_states=period_data.states,
        regime_params=regime_params,
    )
    if choice_action not in state_action_space.discrete_actions:
        return None

    # Build the per-subject smoothed-probability function eagerly (no `jax.jit`).
    # Jitting it would leave a compiled program — including the full GETTSIM
    # tax/transfer system over the joint action grid — resident on the device for
    # the whole run, and that persistent footprint is what starves each
    # backward-induction solve of its large contiguous block. Eager execution
    # holds nothing between calls; subject batching (below) bounds the transient
    # masked-Q-array and GETTSIM intermediates instead.
    inner = get_smoothed_choice_probs_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=state_action_space.action_names,
        state_names=state_action_space.state_names,
        choice_action=choice_action,
        tau=tau,
    )
    spacemapped = simulation_spacemap(
        func=inner,
        action_names=(),
        state_names=tuple(state_action_space.states),
    )
    # Fixed params are partialled into the compiled solve/simulate functions, so
    # `regime_params` (the result's flat params) omits them; the recomputed raw
    # `Q_and_F` still needs them. Merge the regime's resolved fixed params first,
    # exactly as the engine does at solve time, with runtime params overriding.
    all_params = {**regime.resolved_fixed_params, **regime_params}
    # Evaluate in subject batches: the masked Q-array over the joint action grid
    # (and the GETTSIM tax/transfer intermediates feeding it) scale with the batch
    # size, so a small batch caps the smoothed-path peak memory independently of
    # the subject count — leaving each backward-induction solve the large
    # contiguous block it needs.
    states = state_action_space.states
    n_subjects = int(next(iter(states.values())).shape[0])
    shared_inputs = {
        **state_action_space.discrete_actions,
        **state_action_space.continuous_actions,
        "next_regime_to_V_arr": next_regime_to_V_arr,
        **all_params,
        "period": jnp.int32(period),
        "age": age,
    }
    batches = [
        spacemapped(
            **{
                name: array[start : start + _SUBJECT_BATCH_SIZE]
                for name, array in states.items()
            },
            **shared_inputs,
        )
        for start in range(0, n_subjects, _SUBJECT_BATCH_SIZE)
    ]
    return jnp.concatenate(batches, axis=0)


# Subjects per smoothed-recompute batch. Eager execution materializes the masked
# Q-array and the GETTSIM intermediates over `batch x joint-action-grid`, so a
# small batch bounds the transient footprint; the batches free between iterations,
# leaving nothing resident to crowd out the next solve.
_SUBJECT_BATCH_SIZE = 256


def _tidy_period_rows(
    *,
    probabilities: np.ndarray,
    in_regime: np.ndarray,
    period: int,
    regime_name: RegimeName,
) -> pd.DataFrame:
    """Long-format probability rows for one regime-period's in-regime subjects."""
    subject_ids = np.flatnonzero(in_regime)
    n_levels = probabilities.shape[-1]
    return pd.DataFrame(
        {
            "subject": np.repeat(subject_ids, n_levels),
            "period": period,
            "regime": regime_name,
            "level": np.tile(np.arange(n_levels), subject_ids.size),
            "probability": probabilities[subject_ids].reshape(-1),
        }
    )
