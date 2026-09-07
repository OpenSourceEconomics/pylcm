"""Durable model fingerprints of every in-tree model are pinned to a checked-in table.

The digest covers the arrays a model fixes at build, and their dtype follows the
working float format, so the table records one row per format and each test reads
the row of the format the session runs under.

Running this file as a script rewrites one row of the table:
`python tests/solution/test_fingerprint_pin.py 64`, then the same with `32`. The
float format has to be chosen before the model modules build their grids, which is
why the script sets it above the model imports.
"""

import json
import sys
from pathlib import Path

import jax
import pytest

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", sys.argv[1] == "64")

from lcm_examples.collective_regimes import get_dissolution_model
from tests.test_models import (
    ds_app2_housing,
    n_nbegm_toy,
    nbegm_ride_along_toy,
    nbegm_stochastic_node_toy,
    negm_kinked_toy,
)
from tests.test_models.processes import get_multi_regime_model

_FIXTURE = Path(__file__).parents[1] / "data" / "fingerprints_slice4_base.json"

_MODELS = {
    "multi_regime_normal": lambda: get_multi_regime_model(
        n_periods=6, distribution_type="normal"
    ),
    "dissolution": get_dissolution_model,
    "ds_app2_housing": lambda: ds_app2_housing.build_model(n_grid=8),
    "negm_kinked_toy": negm_kinked_toy.build_model,
    "n_nbegm_toy": lambda: n_nbegm_toy.build_model(variant="n_nbegm"),
    "nbegm_ride_along_toy": nbegm_ride_along_toy.build_model,
    "nbegm_stochastic_node_toy": nbegm_stochastic_node_toy.build_model,
}


def _precision_key() -> str:
    """Return the working float format the current session builds arrays in."""
    return "64" if jax.config.jax_enable_x64 else "32"


def _pinned() -> dict[str, str]:
    return json.loads(_FIXTURE.read_text())[_precision_key()]


@pytest.mark.parametrize("key", sorted(_MODELS))
def test_model_fingerprint_matches_pinned_table(key: str) -> None:
    """A model's durable fingerprint equals the value recorded in the fixture table."""
    model = _MODELS[key]()
    assert model._model_structure_fingerprint == _pinned()[key]


def test_pinned_table_covers_every_listed_model() -> None:
    """The fixture table has one entry per model in the pinned set."""
    assert set(_pinned()) == set(_MODELS)


def test_pinned_table_covers_both_working_float_formats() -> None:
    """The fixture table records a row for each float format the suite runs under."""
    assert set(json.loads(_FIXTURE.read_text())) == {"32", "64"}


if __name__ == "__main__":
    recorded = json.loads(_FIXTURE.read_text()) if _FIXTURE.exists() else {}
    recorded[_precision_key()] = {
        key: _MODELS[key]()._model_structure_fingerprint for key in sorted(_MODELS)
    }
    _FIXTURE.write_text(
        json.dumps({key: recorded[key] for key in sorted(recorded)}, indent=2) + "\n"
    )
