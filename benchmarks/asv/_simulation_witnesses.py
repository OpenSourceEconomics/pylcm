"""Forward-simulation witnesses shared by the dispatch tests and benchmarks.

Two CPU models cover the two shapes forward simulation has: a multi-regime
model whose states carry a discretised shock process, and a collective model
whose regime transition runs through gated edges. Each builder returns the
model, its params, and initial conditions for a handful of subjects.

The module imports no test framework, so a benchmark runner whose environment
holds only the project's runtime dependencies can import it.
"""

import pathlib
import sys
from collections.abc import Callable, Mapping

from jax import numpy as jnp

from lcm import Model
from lcm.typing import UserInitialConditions, UserParams

_REPOSITORY_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    # The benchmark runner launches its worker from the benchmark directory, so
    # the shared test models are otherwise not importable.
    sys.path.insert(0, _REPOSITORY_ROOT)

from tests.test_models.processes import (  # noqa: E402
    MultiRegimeId,
    get_multi_regime_model,
    get_multi_regime_params,
)

MULTI_INITIAL_CONDITIONS = {
    "health": jnp.array([0, 1, 0, 1, 0, 1, 0], dtype=jnp.int32),
    "income": jnp.array([0.0, 0.5, -0.3, 0.2, 0.1, -0.1, 0.4]),
    "wealth": jnp.array([1.0, 2.0, 3.0, 1.5, 2.5, 4.0, 1.2]),
    "age": jnp.zeros(7),
    "regime_id": jnp.full(7, MultiRegimeId.work, dtype=jnp.int32),
}


def multi_regime() -> tuple[Model, UserParams, UserInitialConditions]:
    """Build the two-non-terminal-regime shock model and its simulate inputs."""
    return (
        get_multi_regime_model(n_periods=6, distribution_type="normal"),
        get_multi_regime_params("normal"),
        MULTI_INITIAL_CONDITIONS,
    )


def dissolution() -> tuple[Model, UserParams, UserInitialConditions]:
    """Build the collective dissolution model and its simulate inputs."""
    from lcm_examples.collective_regimes import (
        get_dissolution_model,
        get_params,
    )

    model = get_dissolution_model()
    initial_conditions = {
        "wage": jnp.array([1.0, 2.0, 3.0]),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, model.regime_names_to_ids["married"], dtype=jnp.int32),
        "own_stakeholder": jnp.full(
            3, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
        ),
    }
    return model, get_params(), initial_conditions


WITNESSES: Mapping[
    str, Callable[[], tuple[Model, UserParams, UserInitialConditions]]
] = {
    "dissolution": dissolution,
    "multi_regime": multi_regime,
}
