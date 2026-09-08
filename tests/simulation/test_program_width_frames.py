"""Execution widths preserve simulated subjects, decisions and numerical values."""

import functools

import numpy as np
import pandas as pd
import pytest

from _lcm.dtypes import canonical_float_dtype
from benchmarks.asv._simulation_witnesses import WITNESSES
from lcm.execution import ExecutionConfig
from tests.conftest import assert_agrees_to_ulp


@functools.cache
def _baseline(*, witness: str, seed: int) -> pd.DataFrame:
    """Simulate the unconfigured model once for each economic witness and seed."""
    model, params, initial = WITNESSES[witness](execution_config=ExecutionConfig())
    return model.simulate(
        params=params, initial_conditions=initial, seed=seed, log_level="off"
    ).to_dataframe(use_labels=False)


@functools.cache
def _frames(
    *,
    witness: str,
    seed: int,
    prewarm: bool,
    action_width: int,
    subject_width: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare widths with fixed model declarations and a globally addressed seed."""
    expected = _baseline(witness=witness, seed=seed)
    _, params, initial = WITNESSES[witness]()
    widths = {} if subject_width is None else {"subject": subject_width}
    if witness == "multi_regime":
        widths["action_product"] = action_width
    model, _, _ = WITNESSES[witness](
        execution_config=ExecutionConfig(axis_widths=widths),
        n_subjects=len(initial["regime_id"]) if prewarm else None,
    )
    got = model.simulate(
        params=params, initial_conditions=initial, seed=seed, log_level="off"
    ).to_dataframe(use_labels=False)
    return got, expected


@pytest.mark.parametrize("witness", sorted(WITNESSES))
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize(
    ("action_width", "subject_width"), [(1, 1), (2, 3), (3, 7), (7, None)]
)
def test_program_widths_preserve_structural_frame_columns(
    *,
    witness: str,
    seed: int,
    prewarm: bool,
    action_width: int,
    subject_width: int | None,
) -> None:
    """Subject identities, membership and selected grid actions are exact."""
    got, expected = _frames(
        witness=witness,
        seed=seed,
        prewarm=prewarm,
        action_width=action_width,
        subject_width=subject_width,
    )
    model, _, _ = WITNESSES[witness]()
    states_and_actions = {
        name
        for regime in model._regimes.values()
        for name in (*regime.simulation.state_names, *regime.simulation.action_names)
    }
    columns = sorted(
        {str(name) for name in got.select_dtypes(exclude=np.floating)}
        | states_and_actions
    )
    pd.testing.assert_frame_equal(got[columns], expected[columns], check_exact=True)


@pytest.mark.parametrize("witness", sorted(WITNESSES))
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("prewarm", [False, True])
@pytest.mark.parametrize(
    ("action_width", "subject_width"), [(1, 1), (2, 3), (3, 7), (7, None)]
)
def test_program_widths_preserve_continuous_frame_columns(
    *,
    witness: str,
    seed: int,
    prewarm: bool,
    action_width: int,
    subject_width: int | None,
) -> None:
    """Solve and simulation widths preserve values within measured format ULP."""
    got, expected = _frames(
        witness=witness,
        seed=seed,
        prewarm=prewarm,
        action_width=action_width,
        subject_width=subject_width,
    )
    value_columns = [
        name
        for name in got.columns
        if name == "value" or str(name).startswith("value_")
    ]
    assert_agrees_to_ulp(
        got=np.asarray(got[value_columns], dtype=canonical_float_dtype()),
        expected=np.asarray(expected[value_columns], dtype=canonical_float_dtype()),
        n_ulp=16 if np.dtype(canonical_float_dtype()) == np.dtype(np.float32) else 8,
    )
