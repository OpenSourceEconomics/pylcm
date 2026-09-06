"""A solve releases a gated continuation after its source consumed it.

`log_level="debug"` logs every release with the artifact key and the closing
dispatch; a solve at any other level publishes the same values.
"""

import logging

import numpy as np

from _lcm.execution.value_transfer import ValueArtifactKind
from lcm import AgeGrid, Model
from lcm.solver_api import SolutionResult
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
)


def _params() -> dict[str, float]:
    """Discount factor and consent-threshold parameters of the mini-EKL model."""
    return {"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2}


class _Records(logging.Handler):
    """Collect the release records the solve emits, in emission order."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if hasattr(record, "artifact_key"):
            self.records.append(record)


def _model() -> Model:
    return Model(
        regimes=_make_full_topology_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=EKLRegimeId,
    )


def _solve_with_release_log() -> tuple[Model, list[logging.LogRecord], SolutionResult]:
    model = _model()
    handler = _Records()
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        solution = model.solve(params=_params(), log_level="debug")
    finally:
        logger.removeHandler(handler)
    return model, handler.records, solution


def test_a_gated_continuation_is_released_by_the_source_that_reads_it() -> None:
    """Every released Wbar names the source's own dispatch as the closer."""
    _, records, _ = _solve_with_release_log()
    released = [
        (r.artifact_key, r.closing_dispatch)  # ty: ignore[unresolved-attribute]
        for r in records
        if r.artifact_key.kind  # ty: ignore[unresolved-attribute]
        is ValueArtifactKind.GATED_CONTINUATION
    ]

    assert released
    assert all(
        dispatch == (artifact.period - 1, artifact.regime)
        for artifact, dispatch in released
    )


def test_no_regime_value_is_ever_released() -> None:
    """Retained values stay on device for the whole solve."""
    _, records, solution = _solve_with_release_log()

    assert not any(
        r.artifact_key.kind  # ty: ignore[unresolved-attribute]
        is ValueArtifactKind.REGIME_VALUE
        for r in records
    )
    assert not any(
        value.is_deleted()
        for period in solution.values.values()
        for value in period.values()
    )


def test_releasing_changes_no_published_value() -> None:
    """A solve with releases publishes the values a silent solve publishes."""
    model, _, debug = _solve_with_release_log()
    silent = model.solve(params=_params(), log_level="off")

    np.testing.assert_array_equal(
        np.asarray(debug.values[1]["married"]), np.asarray(silent.values[1]["married"])
    )
