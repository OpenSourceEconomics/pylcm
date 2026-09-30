"""The per-period diagnostics row takes its age from host memory.

The row is appended once per regime-period inside the backward-induction loop,
right after the period's value array was dispatched. Reading the age off the
device array there would be a blocking device-to-host copy queued behind that
dispatch, so the row's age is read from the grid's exact host-side values and
never touches the device array.
"""

import jax.numpy as jnp
import pytest

from _lcm.solution.diagnostics import (
    _fold_period_diagnostics,
    _init_diagnostic_accumulators,
)
from lcm import AgeGrid


def test_diagnostic_row_age_is_read_without_touching_the_device_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row carries the period's age although the device array is unreadable."""
    ages = AgeGrid(start=50, stop=52, step="Y")

    def _refuse(_grid: AgeGrid) -> None:
        msg = "the diagnostics fold read the age off the device array"
        raise AssertionError(msg)

    monkeypatch.setattr(AgeGrid, "values", property(_refuse))
    rows, mins, maxs, means, any_nan, any_inf = _init_diagnostic_accumulators()
    _fold_period_diagnostics(
        V_arr=jnp.zeros((3,)),
        regime_name="alive",
        period=1,
        ages=ages,
        diagnostics_enabled=True,
        stats_enabled=False,
        diagnostic_rows=rows,
        diagnostic_min=mins,
        diagnostic_max=maxs,
        diagnostic_mean=means,
        running_any_nan=any_nan,
        running_any_inf=any_inf,
    )
    assert rows[0].age == pytest.approx(51.0)
