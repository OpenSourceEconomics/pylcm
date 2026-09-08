"""Durable model fingerprints of every in-tree model are pinned to a checked-in table.

The digest covers the arrays a model fixes at build, and their dtype follows the
working float format, so the table records one row per format and each test reads
the row of the format the session runs under. What it never covers is execution
policy: two models differing only in a block-size field are the same model, and
that holds through the role-bound solver subclass a regime actually stores.

Running this file as a script rewrites one row of the table:
`python tests/solution/test_fingerprint_pin.py 64`, then the same with `32`. The
float format has to be chosen before the model modules build their grids, which is
why the script sets it above the model imports.
"""

import dataclasses
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import jax
import pytest

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", sys.argv[1] == "64")

from _lcm.solution.fingerprint import (
    _BUILTIN_EXECUTION_FIELDS_BY_TYPE,
    _exclude_field,
)
from lcm import Model
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


# Deep enough to reach a solver nested inside another solver's config, shallow
# enough that the walk terminates on an ordinary model.
_SCAN_DEPTH = 8


def _registered_owners(*, model: Model) -> list[tuple[object, frozenset[str]]]:
    """Every object a model stores whose type a table row covers.

    A row registers the public solver or config class, and a regime stores the
    role-bound subclass, so the match is by `isinstance` — which is exactly what
    the exclusion predicate has to see through. Walking the user regimes reaches
    the objects the fingerprint visitor walks.
    """
    found: list[tuple[object, frozenset[str]]] = []
    seen: set[int] = set()

    def visit(*, value: object, depth: int = 0) -> None:
        if depth > _SCAN_DEPTH or id(value) in seen:
            return
        seen.add(id(value))
        found.extend(
            (value, names)
            for registered_type, names in _BUILTIN_EXECUTION_FIELDS_BY_TYPE
            if isinstance(value, registered_type)
        )
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for declaration in dataclasses.fields(value):
                visit(value=getattr(value, declaration.name, None), depth=depth + 1)
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(value=item, depth=depth + 1)
        elif isinstance(value, list | tuple):
            for item in value:
                visit(value=item, depth=depth + 1)

    for regime in model.user_regimes.values():
        visit(value=regime)
    return found


@pytest.mark.parametrize("key", sorted(_MODELS))
def test_no_model_hashes_an_execution_policy_field(key: str) -> None:
    """Every registered field of every solver a model stores is excluded.

    Generated from `_BUILTIN_EXECUTION_FIELDS_BY_TYPE` rather than named, so a
    row or a field added there is covered without touching this test, and the
    pinned digests below cannot come to rest on a half-covered mechanism.
    """
    model = _MODELS[key]()
    leaked = sorted(
        (type(owner).__name__, name)
        for owner, names in _registered_owners(model=model)
        for name in names
        if not _exclude_field(owner=owner, field_name=name)
    )

    assert leaked == []


def test_the_policy_field_scan_reaches_the_bound_solvers() -> None:
    """The scan is not vacuous: it finds the role-bound solver a regime stores.

    Without this, a walk that silently stopped finding owners would leave the
    test above asserting nothing at all. `NNBEGM` carries no row of its own —
    its outer search is a registered config object — so the bound forms the
    table can reach are NEGM's and NB-EGM's.
    """
    found = {
        type(owner).__name__
        for key in _MODELS
        for owner, _ in _registered_owners(model=_MODELS[key]())
    }

    assert {"_BoundNEGM", "_BoundNBEGM"} <= found


# One entry per in-tree solver carrying an execution-policy field, as the pair of
# model builders differing in that field and in nothing else.
_POLICY_VARIANTS = {
    "negm_outer_batch_size": (
        lambda: negm_kinked_toy.build_model(outer_batch_size=0),
        lambda: negm_kinked_toy.build_model(outer_batch_size=4),
    ),
    "negm_outer_batch_size_housing": (
        lambda: ds_app2_housing.build_model(n_grid=8, outer_batch_size=0),
        lambda: ds_app2_housing.build_model(n_grid=8, outer_batch_size=2),
    ),
    "nbegm_stochastic_node_batch_size": (
        lambda: nbegm_stochastic_node_toy.build_model(
            variant="nbegm", stochastic_node_batch_size=0
        ),
        lambda: nbegm_stochastic_node_toy.build_model(
            variant="nbegm", stochastic_node_batch_size=2
        ),
    ),
    "nnbegm_outer_batch_size": (
        lambda: n_nbegm_toy.build_model(variant="n_nbegm", outer_batch_size=0),
        lambda: n_nbegm_toy.build_model(variant="n_nbegm", outer_batch_size=2),
    ),
}


@pytest.mark.parametrize("key", sorted(_POLICY_VARIANTS))
def test_fingerprint_is_invariant_to_a_solvers_execution_policy(key: str) -> None:
    """A solver's block-size field is execution policy, so it never enters the digest.

    A regime stores its solver as the role-bound subclass, so the fingerprint
    has to see through that binding: two models differing only in one such
    field are the same model and carry the same durable fingerprint.
    """
    reference, varied = _POLICY_VARIANTS[key]

    assert (
        varied()._model_structure_fingerprint
        == reference()._model_structure_fingerprint
    )


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
