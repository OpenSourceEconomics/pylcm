"""Labeled input conversion must use the admitted numeric entry writer."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from _lcm import pandas_utils
from lcm import ExecutionConfig
from lcm.exceptions import ExecutionPlanningError
from tests.solution.test_solution_result import _small_grid_search_inputs


def _refuse_unadmitted_upload(*args: Any, **kwargs: Any) -> object:
    del args, kwargs
    raise AssertionError("Pandas conversion uploaded outside numeric admission")


@dataclass(frozen=True, kw_only=True)
class _NoDirectPandasUploads:
    """Leave dtype metadata available while exposing direct device allocation."""

    original: Any

    def __getattr__(self, name: str) -> object:
        if name in {"array", "asarray", "full"}:
            return _refuse_unadmitted_upload
        return getattr(self.original, name)


@pytest.mark.parametrize("fits", [True, False])
def test_dataframe_uploads_pass_through_numeric_admission(
    *, fits: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fitting frame runs; an oversized frame refuses before any direct upload."""
    model, params, _ = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28 if fits else 2**14)
    )
    count = 3 if fits else 4096
    frame = pd.DataFrame(
        {
            "regime_name": ["working_life"] * count,
            "wealth": np.full(count, 2.0),
            "age": np.full(count, 18.0),
        }
    )
    monkeypatch.setattr(
        pandas_utils, "jnp", _NoDirectPandasUploads(original=pandas_utils.jnp)
    )
    if fits:
        result = model.simulate(
            params=params, initial_conditions=frame, log_level="off"
        )
        assert result.n_subjects == count
        observed = result.to_dataframe().query("period == 0")
        np.testing.assert_array_equal(observed["wealth"].to_numpy(), frame["wealth"])
    else:
        with pytest.raises(ExecutionPlanningError):
            model.simulate(params=params, initial_conditions=frame, log_level="off")
