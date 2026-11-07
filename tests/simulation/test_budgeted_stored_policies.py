"""Budgeted simulation reads host- and disk-stored policies as it reads device ones.

Under a device-memory budget the chunk planner receives descriptors of the stored
policies rather than the policies themselves. Each panel here is simulated from a
same-instance solve under such a budget and must equal the device-stored panel.
"""

import dataclasses
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import ExecutionConfig, Model
from lcm.ages import AgeGrid
from lcm_examples.iskhakov_et_al_2017 import dead, get_edges
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.test_models import n_nbegm_toy
from tests.test_models.deterministic import base
from tests.test_models.deterministic.dcegm_variants import (
    DCEGM_SOLVER,
    dcegm_retirement,
    dcegm_working_life,
    get_full_params,
)

# A logical ceiling, not an allocation. It keeps a CPU simulation on the budgeted
# route, where chunk admission plans every decision stage.
_BUDGET = 2**32

type ValueStorage = Literal["device", "host"] | Path

_NNBEGM_PARAMS = {"discount_factor": 0.95}
_NNBEGM_INITIAL_CONDITIONS = {
    "wealth": np.array([4.3, 11.7]),
    "illiquid": np.array([1.37, 6.6]),
    "age": np.array([20.0, 20.0]),
    "regime_id": np.zeros(2, dtype=np.int32),
}

_EGM_N_PERIODS = 4
# The last subject starts beyond the published rows, so the grid fallback beside
# the policy-first read is planned and run as well.
_EGM_WEALTH = np.append(np.linspace(20.5, 180.5, 9), 1000.0)


def _execution(value_storage: ValueStorage) -> ExecutionConfig:
    return ExecutionConfig(
        device_memory_bytes=_BUDGET,
        axis_widths={"subject": 1},
        value_storage=value_storage,
    )


def _nnbegm_panel(value_storage: ValueStorage) -> pd.DataFrame:
    model = n_nbegm_toy.build_model(
        variant="n_nbegm", n_periods=2, execution_config=_execution(value_storage)
    )
    solution = model.solve(params=_NNBEGM_PARAMS, log_level="off")
    return model.simulate(
        params=_NNBEGM_PARAMS,
        initial_conditions=dict(_NNBEGM_INITIAL_CONDITIONS),
        solution=solution,
        seed=5,
        log_level="off",
    ).to_dataframe()


def _egm_panel(value_storage: ValueStorage) -> pd.DataFrame:
    solver = dataclasses.replace(DCEGM_SOLVER, replay="candidate")
    ages = AgeGrid(start=40, inclusive_stop=40 + (_EGM_N_PERIODS - 1) * 10, step="10Y")
    model = Model(
        edges=get_edges(ages=ages),
        regimes={
            "working_life": dcegm_working_life.replace(solver=solver),
            "retirement": dcegm_retirement.replace(solver=solver),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
        initial_nodes=((40, "working_life"), (40, "retirement")),
        execution_config=_execution(value_storage),
    )
    params = get_full_params(n_periods=_EGM_N_PERIODS)
    solution = model.solve(params=params, log_level="off")
    return model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.asarray(_EGM_WEALTH),
            "age": jnp.full(_EGM_WEALTH.shape, 40.0),
            "regime_id": jnp.full(_EGM_WEALTH.shape, base.RegimeId.working_life),
        },
        solution=solution,
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize("storage", ["host", "disk"])
def test_budgeted_panel_of_stored_finite_nnbegm_policies_is_the_device_panel(
    *, tmp_path: Path, storage: str
) -> None:
    """A budgeted finite NNBEGM panel read off stored policies is the device panel."""
    if jax.default_backend() != "cpu":
        pytest.skip("Budgeted finite NNBEGM simulation is refused on accelerators.")
    stored = _nnbegm_panel("host" if storage == "host" else tmp_path)

    pd.testing.assert_frame_equal(stored, _nnbegm_panel("device"), check_exact=True)


@pytest.mark.slow
@pytest.mark.parametrize("storage", ["host", "disk"])
def test_budgeted_panel_of_stored_egm_policies_is_the_device_panel(
    *, tmp_path: Path, storage: str
) -> None:
    """A budgeted DC-EGM panel read off stored policies is the device panel."""
    stored = _egm_panel("host" if storage == "host" else tmp_path)

    pd.testing.assert_frame_equal(stored, _egm_panel("device"), check_exact=True)
