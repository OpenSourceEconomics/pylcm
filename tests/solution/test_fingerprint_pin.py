"""Durable model fingerprints of every in-tree model are pinned to a checked-in table.

The digest covers the arrays a model fixes at build, and their dtype follows the
working float format, so the table records one row per format and each test reads
the row of the format the session runs under. What it never covers is execution
policy: two models differing only in ExecutionConfig widths are the same model.

The immutable table records Solver API 2 identities. A test-only projection of
SolverIdentity.solver_api_version isolates the deliberate API 3 compatibility
break; every other semantic field must still reproduce the historical digest.
Production fingerprints continue to bind the current API version.
"""

import dataclasses
import json
from pathlib import Path

import jax
import pytest

from _lcm.solution.fingerprint import _SemanticHasher, fingerprint_model_structure
from lcm import ExecutionConfig
from lcm.solver_api import SolverIdentity
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


# Model-builder pairs differing only in a solver's declared execution width.
_POLICY_VARIANTS = {
    "nbegm_stochastic_node_width": (
        lambda: nbegm_stochastic_node_toy.build_model(variant="nbegm"),
        lambda: nbegm_stochastic_node_toy.build_model(
            variant="nbegm",
            execution_config=ExecutionConfig(axis_widths={"stochastic_node": 2}),
        ),
    ),
}


@pytest.mark.parametrize("key", sorted(_POLICY_VARIANTS))
def test_fingerprint_is_invariant_to_a_solvers_execution_policy(key: str) -> None:
    """ExecutionConfig widths do not change the model's durable fingerprint."""
    reference, varied = _POLICY_VARIANTS[key]

    assert (
        varied()._model_structure_fingerprint
        == reference()._model_structure_fingerprint
    )


@pytest.mark.parametrize("key", sorted(_MODELS))
def test_model_declaration_matches_api2_pin_after_version_projection(
    *, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the explicit solver API identity differs from the immutable model pins."""
    original = _SemanticHasher._visit_dataclass
    projected = []
    model = _MODELS[key]()

    # keyword-only-exempt: library-callback=_SemanticHasher._visit_dataclass
    def visit_at_api2(self: _SemanticHasher, value: object) -> None:
        if type(value) is SolverIdentity:
            assert value.solver_api_version == 3
            projected.append(value)
            # This intentionally incompatible identity exists only inside the
            # historical hash oracle; no solver or archive consumes it.
            historical = object.__new__(SolverIdentity)
            for declaration in dataclasses.fields(value):
                object.__setattr__(
                    historical,
                    declaration.name,
                    2
                    if declaration.name == "solver_api_version"
                    else getattr(value, declaration.name),
                )
            value = historical
        original(self, value)

    monkeypatch.setattr(_SemanticHasher, "_visit_dataclass", visit_at_api2)
    historical = fingerprint_model_structure(
        ages=model.ages,
        regimes=model._regimes,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    assert projected, "The version projection must actually observe solver identities."
    assert historical == _pinned()[key]


@pytest.mark.parametrize("key", sorted(_MODELS))
def test_production_fingerprint_binds_the_current_solver_api(key: str) -> None:
    """Production fingerprints preserve the deliberate API compatibility break."""
    assert _MODELS[key]()._model_structure_fingerprint != _pinned()[key]


def test_pinned_table_covers_every_listed_model() -> None:
    """The fixture table has one entry per model in the pinned set."""
    assert set(_pinned()) == set(_MODELS)


def test_pinned_table_covers_both_working_float_formats() -> None:
    """The fixture table records a row for each float format the suite runs under."""
    assert set(json.loads(_FIXTURE.read_text())) == {"32", "64"}
