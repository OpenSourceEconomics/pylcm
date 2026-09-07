"""Durable identity binds an array's original rank and ordered shape.

A model's durable fingerprint hashes the arrays its declarations reference — closure
cells, defaults, and solution-relevant parameters. Two arrays with the same dtype and
bytes but different shapes are different mathematical objects (`jnp.ndim` alone tells
them apart), so the shape frame must record the shape the declaration actually holds.
Memory order is storage, not mathematics: a Fortran-ordered or strided view of the same
values fingerprints identically to its C-contiguous copy.
"""

from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from _lcm.solution import artifacts as private_artifacts
from _lcm.solution import fingerprint as fingerprints
from _lcm.typing import FlatParams
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model
from lcm.exceptions import InvalidSimulationInputError
from lcm.persistence import load_solution
from lcm.typing import FloatND, UserInitialConditions, UserParams
from lcm_examples.mortality import LaborSupply
from tests.test_models.deterministic.regression import (
    START_AGE,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_DTYPES = [np.float32, np.float64]


def _scalar_and_vector(dtype: type[np.generic]) -> list[tuple[object, object]]:
    return [
        (np.array(0.0, dtype=dtype), np.array([0.0], dtype=dtype)),
        (dtype(0.0), np.array([0.0], dtype=dtype)),
        (np.array([0.0], dtype=dtype), np.array([[0.0]], dtype=dtype)),
        (np.array([1.0, 2.0], dtype=dtype), np.array([[1.0, 2.0]], dtype=dtype)),
        (np.array([[1.0, 2.0]], dtype=dtype), np.array([[1.0], [2.0]], dtype=dtype)),
    ]


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float32", "float64"])
@pytest.mark.parametrize("case", range(5))
def test_semantic_fingerprint_distinguishes_arrays_that_differ_only_in_shape(
    *, dtype: type[np.generic], case: int
) -> None:
    """Equal dtype and bytes with different original shapes are different values."""
    lower, higher = _scalar_and_vector(dtype)[case]
    assert np.asarray(lower).tobytes() == np.asarray(higher).tobytes()

    assert fingerprints._semantic_fingerprint(lower) != (
        fingerprints._semantic_fingerprint(higher)
    )


def test_semantic_fingerprint_distinguishes_jax_scalar_and_length_one_arrays() -> None:
    scalar = jnp.asarray(0.0)
    vector = jnp.asarray([0.0])

    assert fingerprints._semantic_fingerprint(scalar) != (
        fingerprints._semantic_fingerprint(vector)
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float32", "float64"])
def test_semantic_fingerprint_ignores_memory_order_of_the_same_array(
    *, dtype: type[np.generic]
) -> None:
    """A Fortran-ordered or strided view of the same values is the same value."""
    base = np.arange(12, dtype=dtype).reshape(3, 4)
    fortran = np.asfortranarray(base)
    strided = np.repeat(base, 2, axis=1)[:, ::2]
    assert not fortran.flags.c_contiguous
    assert not strided.flags.c_contiguous
    np.testing.assert_array_equal(strided, base)

    expected = fingerprints._semantic_fingerprint(base)
    assert fingerprints._semantic_fingerprint(fortran) == expected
    assert fingerprints._semantic_fingerprint(strided) == expected


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float32", "float64"])
def test_semantic_fingerprint_distinguishes_equal_shapes_with_different_contents(
    *, dtype: type[np.generic]
) -> None:
    assert fingerprints._semantic_fingerprint(np.array([0.0], dtype=dtype)) != (
        fingerprints._semantic_fingerprint(np.array([1.0], dtype=dtype))
    )


def _rank_signed_terminal_utility(
    reference: np.ndarray,
) -> Callable[[], FloatND]:
    """Terminal utility `2 * rank(reference) - 1`, from one factory for every rank."""

    def utility() -> FloatND:
        return jnp.asarray(2 * jnp.ndim(reference) - 1, dtype=float)

    return utility


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float32", "float64"])
def test_closures_over_scalar_and_vector_arrays_fingerprint_differently(
    *, dtype: type[np.generic]
) -> None:
    scalar_closure = _rank_signed_terminal_utility(np.array(0.0, dtype=dtype))
    vector_closure = _rank_signed_terminal_utility(np.array([0.0], dtype=dtype))

    assert fingerprints._semantic_fingerprint(scalar_closure) != (
        fingerprints._semantic_fingerprint(vector_closure)
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float32", "float64"])
def test_flat_params_fingerprint_distinguishes_scalar_and_length_one_arrays(
    *, dtype: type[np.generic]
) -> None:
    scalar = private_artifacts.fingerprint_flat_params(
        cast(
            "FlatParams",
            MappingProxyType(
                {"working": MappingProxyType({"weight": np.array(0.0, dtype=dtype)})}
            ),
        )
    )
    vector = private_artifacts.fingerprint_flat_params(
        cast(
            "FlatParams",
            MappingProxyType(
                {"working": MappingProxyType({"weight": np.array([0.0], dtype=dtype)})}
            ),
        )
    )

    assert scalar != vector


def _rank_model(reference: np.ndarray) -> Model:
    """Two-period GridSearch model whose terminal payoff depends on the closure rank."""
    final_age_alive = START_AGE
    grid = LinSpacedGrid(start=1, stop=3, n_points=3)
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": grid},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": grid,
                },
            ),
            "dead": dead.replace(
                functions={"utility": _rank_signed_terminal_utility(reference)}
            ),
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
    )


def _rank_model_inputs() -> tuple[UserParams, UserInitialConditions]:
    params = get_params(n_periods=2)
    initial_conditions: UserInitialConditions = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([18.0]),
        "regime_id": jnp.asarray([RegimeId.working_life], dtype=jnp.int32),
    }
    return params, initial_conditions


def test_restored_solution_is_refused_by_a_model_differing_only_in_closure_rank(
    tmp_path: Path,
) -> None:
    """Saving under the scalar-closure model and loading into the vector-closure
    model fails before replay; an independently rebuilt scalar-closure model still
    replays the archive and reproduces the original simulation."""
    params, initial_conditions = _rank_model_inputs()
    scalar_model = _rank_model(np.array(0.0))
    vector_model = _rank_model(np.array([0.0]))
    rebuilt_scalar_model = _rank_model(np.array(0.0))

    solved = scalar_model.solve(params=params, log_level="off")
    assert solved.metadata.model_fingerprint != (
        vector_model.solve(params=params, log_level="off").metadata.model_fingerprint
    )
    restored = load_solution(path=solved.save(path=tmp_path / "scalar.lcm"))

    with pytest.raises(InvalidSimulationInputError, match="model_fingerprint"):
        vector_model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=restored,
            log_level="off",
        )

    expected = scalar_model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solved,
        log_level="off",
    ).to_dataframe()
    replayed = rebuilt_scalar_model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=restored,
        log_level="off",
    ).to_dataframe()
    assert_frame_equal(replayed, expected)
