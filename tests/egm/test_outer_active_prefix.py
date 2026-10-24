"""Padded outer profiles use only their runtime active node prefix."""

import inspect
from collections.abc import Callable
from dataclasses import fields
from enum import Enum, auto
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_refinement import (
    SafeguardedSearchResult,
    safeguarded_continuous_argmax,
)
from _lcm.typing import HostArray
from lcm.typing import BoolND, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

type NodeCoordinates = tuple[float, ...]
type ReadOutputs = tuple[FloatND | BoolND, ...]

_CAPACITY = 7
_GOLDEN_ITERATIONS = 40
_UNIFORM_TWO = (0.0, 1.0)
_UNIFORM_THREE = (0.0, 0.5, 1.0)
_NONUNIFORM_FOUR = (0.0, 0.125, 0.625, 1.0)


class _Surface(Enum):
    QUADRATIC = auto()
    INCREASING = auto()
    DECREASING = auto()


class _PrefixInterpolant(Protocol):
    def evaluate(
        self,
        *,
        nodes: FloatND,
        values: FloatND,
        query: FloatND,
        interval_valid: BoolND | None = None,
        valid_count: ScalarInt | None = None,
    ) -> FloatND: ...

    def evaluate_with_derivative(
        self,
        *,
        nodes: FloatND,
        values: FloatND,
        query: FloatND,
        interval_valid: BoolND | None = None,
        valid_count: ScalarInt | None = None,
    ) -> tuple[FloatND, FloatND]: ...

    def evaluate_validity(
        self,
        *,
        nodes: FloatND,
        values: FloatND,
        query: FloatND,
        interval_valid: BoolND | None = None,
        valid_count: ScalarInt | None = None,
    ) -> BoolND: ...


class _PrefixSearch(Protocol):
    def __call__(
        self,
        *,
        objective: Callable[[FloatND], FloatND],
        nodes: FloatND,
        node_values: FloatND,
        golden_iterations: int,
        max_brackets: int | None = None,
        valid_count: ScalarInt | None = None,
    ) -> SafeguardedSearchResult: ...


def _prefix_interpolant(
    *, interpolant: LocalCubicOuterInterpolant
) -> _PrefixInterpolant:
    for method_name in (
        "evaluate",
        "evaluate_with_derivative",
        "evaluate_validity",
    ):
        method = getattr(interpolant, method_name)
        assert "valid_count" in inspect.signature(method).parameters, (
            f"LocalCubicOuterInterpolant.{method_name} needs a runtime "
            "active-prefix count"
        )
    return cast("_PrefixInterpolant", interpolant)


def _prefix_search() -> _PrefixSearch:
    assert (
        "valid_count" in inspect.signature(safeguarded_continuous_argmax).parameters
    ), "The safeguarded outer search needs a runtime active-prefix count"
    return cast("_PrefixSearch", safeguarded_continuous_argmax)


def _surface(*, x: FloatND, surface: _Surface) -> FloatND:
    if surface is _Surface.QUADRATIC:
        return 2.0 - (x - 3.0 / 8.0) ** 2
    if surface is _Surface.INCREASING:
        return 1.0 + 2.0 * x
    return 1.0 - 2.0 * x


def _derivative(*, x: FloatND, surface: _Surface) -> FloatND:
    if surface is _Surface.QUADRATIC:
        return -2.0 * (x - 3.0 / 8.0)
    return jnp.full_like(x, 2.0 if surface is _Surface.INCREASING else -2.0)


def _padded(
    *, coordinates: NodeCoordinates, surface: _Surface
) -> tuple[FloatND, FloatND]:
    nodes = jnp.asarray(coordinates)
    count = len(coordinates)
    pad_nodes = jnp.asarray((9.0, -3.0, 9.0, -3.0, 9.0))[: _CAPACITY - count]
    pad_values = jnp.full((_CAPACITY - count,), 10_000.0)
    return (
        jnp.concatenate((nodes, pad_nodes)),
        jnp.concatenate((_surface(x=nodes, surface=surface), pad_values)),
    )


def _location_atol(*, value: FloatND) -> float:
    eps = float(np.finfo(np.asarray(value).dtype).eps)
    return max(1e-6, 4.0 * eps**0.5)


def _assert_search_oracle(
    *, result: SafeguardedSearchResult, surface: _Surface
) -> None:
    optimum = (
        3.0 / 8.0
        if surface is _Surface.QUADRATIC
        else (1.0 if surface is _Surface.INCREASING else 0.0)
    )
    expected = (
        2.0
        if surface is _Surface.QUADRATIC
        else (3.0 if surface is _Surface.INCREASING else 1.0)
    )
    np.testing.assert_allclose(
        result.x, optimum, rtol=0.0, atol=_location_atol(value=result.x)
    )
    np.testing.assert_array_almost_equal(
        result.value, expected, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(actual=result.valid, desired=True)
    if surface is _Surface.QUADRATIC:
        assert bool(result.refinement_gain > 0.0)
        np.testing.assert_array_equal(actual=result.at_lower_bound, desired=False)
        np.testing.assert_array_equal(actual=result.at_upper_bound, desired=False)
    else:
        np.testing.assert_array_equal(result.x, optimum)
        np.testing.assert_array_equal(
            result.at_upper_bound, surface is _Surface.INCREASING
        )
        np.testing.assert_array_equal(
            result.at_lower_bound, surface is _Surface.DECREASING
        )


def _assert_search_matches(
    *, actual: SafeguardedSearchResult, reference: SafeguardedSearchResult
) -> None:
    for field in fields(SafeguardedSearchResult):
        observed = getattr(actual, field.name)
        expected = getattr(reference, field.name)
        if expected is None:
            assert observed is None, field.name
        elif field.name in ("x", "second_best_x"):
            np.testing.assert_allclose(
                observed,
                expected,
                rtol=0.0,
                atol=_location_atol(value=expected),
            )
        elif field.name in ("coarse_x", "at_lower_bound", "at_upper_bound", "valid"):
            np.testing.assert_array_equal(observed, expected)
        else:
            np.testing.assert_array_almost_equal(
                observed, expected, decimal=DECIMAL_PRECISION
            )


@pytest.mark.parametrize(
    ("coordinates", "surface"),
    [
        (_UNIFORM_TWO, _Surface.INCREASING),
        (_UNIFORM_THREE, _Surface.QUADRATIC),
        (_NONUNIFORM_FOUR, _Surface.QUADRATIC),
    ],
    ids=("n2-affine", "n3-quadratic", "n4-nonuniform-quadratic"),
)
def test_padded_outer_reads_match_the_active_mesh(
    *, coordinates: NodeCoordinates, surface: _Surface
) -> None:
    """Padding cannot change slopes, endpoint reads or the active domain."""
    interp = LocalCubicOuterInterpolant()
    nodes = jnp.asarray(coordinates)
    values = _surface(x=nodes, surface=surface)
    inside = jnp.asarray((0.0, 0.125, 0.375, 0.875, 1.0))
    reference, slope = interp.evaluate_with_derivative(
        nodes=nodes, values=values, query=inside
    )
    np.testing.assert_array_almost_equal(
        reference, _surface(x=inside, surface=surface), decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        slope, _derivative(x=inside, surface=surface), decimal=DECIMAL_PRECISION
    )
    query = jnp.concatenate((jnp.asarray((-0.125,)), inside, jnp.asarray((1.125,))))
    reference_pair = interp.evaluate_with_derivative(
        nodes=nodes, values=values, query=query
    )
    reference_validity = interp.evaluate_validity(
        nodes=nodes, values=values, query=query
    )
    np.testing.assert_array_equal(
        reference_validity, (False, True, True, True, True, True, False)
    )
    np.testing.assert_array_equal(reference_pair[0][jnp.asarray((0, -1))], -jnp.inf)
    np.testing.assert_array_equal(reference_pair[1][jnp.asarray((0, -1))], 0.0)

    prefix = _prefix_interpolant(interpolant=interp)
    np.testing.assert_array_equal(
        prefix.evaluate(nodes=nodes, values=values, query=query, valid_count=None),
        reference_pair[0],
    )
    none_pair = prefix.evaluate_with_derivative(
        nodes=nodes, values=values, query=query, valid_count=None
    )
    for observed, expected in zip(none_pair, reference_pair, strict=True):
        np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        prefix.evaluate_validity(
            nodes=nodes, values=values, query=query, valid_count=None
        ),
        reference_validity,
    )
    padded_nodes, padded_values = _padded(coordinates=coordinates, surface=surface)
    assert padded_nodes.shape == (_CAPACITY,)
    assert bool(jnp.all(jnp.isfinite(padded_nodes)))
    assert bool(jnp.all(jnp.isfinite(padded_values)))
    actual = prefix.evaluate_with_derivative(
        nodes=padded_nodes,
        values=padded_values,
        query=query,
        valid_count=jnp.int32(len(coordinates)),
    )
    for observed, expected in zip(actual, reference_pair, strict=True):
        np.testing.assert_array_almost_equal(
            observed, expected, decimal=DECIMAL_PRECISION
        )
    np.testing.assert_array_almost_equal(
        prefix.evaluate(
            nodes=padded_nodes,
            values=padded_values,
            query=query,
            valid_count=jnp.int32(len(coordinates)),
        ),
        reference_pair[0],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(
        prefix.evaluate_validity(
            nodes=padded_nodes,
            values=padded_values,
            query=query,
            valid_count=jnp.int32(len(coordinates)),
        ),
        reference_validity,
    )


@pytest.mark.parametrize(
    ("coordinates", "surface"),
    [
        (_UNIFORM_TWO, _Surface.INCREASING),
        (_UNIFORM_THREE, _Surface.QUADRATIC),
        (_NONUNIFORM_FOUR, _Surface.QUADRATIC),
        (_NONUNIFORM_FOUR, _Surface.DECREASING),
    ],
    ids=("n2-upper", "n3-interior", "n4-nonuniform-interior", "n4-lower"),
)
def test_padded_outer_search_uses_only_active_nodes_and_brackets(
    *, coordinates: NodeCoordinates, surface: _Surface
) -> None:
    """High-valued dummy pads cannot become candidates or widen probe brackets."""
    nodes = jnp.asarray(coordinates)

    def objective(query: FloatND) -> FloatND:
        return _surface(x=query, surface=surface)

    reference = safeguarded_continuous_argmax(
        objective=objective,
        nodes=nodes,
        node_values=objective(nodes),
        golden_iterations=_GOLDEN_ITERATIONS,
    )
    _assert_search_oracle(result=reference, surface=surface)
    search = _prefix_search()
    none_result = search(
        objective=objective,
        nodes=nodes,
        node_values=objective(nodes),
        golden_iterations=_GOLDEN_ITERATIONS,
        valid_count=None,
    )
    for field in fields(SafeguardedSearchResult):
        np.testing.assert_array_equal(
            getattr(none_result, field.name), getattr(reference, field.name)
        )
    padded_nodes, padded_values = _padded(coordinates=coordinates, surface=surface)
    probes: list[HostArray] = []

    def record_probe(query: HostArray) -> None:
        probes.append(np.asarray(query))

    def observed_objective(query: FloatND) -> FloatND:
        jax.debug.callback(record_probe, query, ordered=True)
        return objective(query)

    actual = search(
        objective=observed_objective,
        nodes=padded_nodes,
        node_values=padded_values,
        golden_iterations=_GOLDEN_ITERATIONS,
        valid_count=jnp.int32(len(coordinates)),
    )
    jax.block_until_ready(actual.value)
    jax.effects_barrier()
    assert probes, "The bracket observer received no objective probes"
    for probe in probes:
        assert bool(np.all(np.greater_equal(probe, coordinates[0])))
        assert bool(np.all(np.less_equal(probe, coordinates[-1])))
    _assert_search_oracle(result=actual, surface=surface)
    _assert_search_matches(actual=actual, reference=reference)


def test_one_compiled_outer_reader_accepts_runtime_prefix_counts() -> None:
    """A fixed padded input shape carries N=2,3,4 as runtime scalar data."""
    interp = LocalCubicOuterInterpolant()
    reference_cases = (
        (_UNIFORM_TWO, _Surface.INCREASING),
        (_UNIFORM_THREE, _Surface.QUADRATIC),
        (_NONUNIFORM_FOUR, _Surface.QUADRATIC),
    )
    references: list[ReadOutputs] = []
    query = jnp.asarray((0.0, 0.125, 0.375, 0.875, 1.0))
    for coordinates, surface in reference_cases:
        nodes = jnp.asarray(coordinates)
        values = _surface(x=nodes, surface=surface)
        value, derivative = interp.evaluate_with_derivative(
            nodes=nodes, values=values, query=query
        )
        np.testing.assert_array_almost_equal(
            value, _surface(x=query, surface=surface), decimal=DECIMAL_PRECISION
        )
        np.testing.assert_array_almost_equal(
            derivative, _derivative(x=query, surface=surface), decimal=DECIMAL_PRECISION
        )

        # keyword-only-exempt: library-callback=safeguarded_continuous_argmax
        def objective(
            probe: FloatND, *, nodes: FloatND = nodes, values: FloatND = values
        ) -> FloatND:
            return interp.evaluate(nodes=nodes, values=values, query=probe)

        result = safeguarded_continuous_argmax(
            objective=objective,
            nodes=nodes,
            node_values=values,
            golden_iterations=_GOLDEN_ITERATIONS,
        )
        _assert_search_oracle(result=result, surface=surface)
        references.append((value, derivative, result.x, result.value, result.coarse_x))

    prefix = _prefix_interpolant(interpolant=interp)
    search = _prefix_search()

    def read(*, nodes: FloatND, values: FloatND, valid_count: ScalarInt) -> ReadOutputs:
        value, derivative = prefix.evaluate_with_derivative(
            nodes=nodes, values=values, query=query, valid_count=valid_count
        )

        def objective(probe: FloatND) -> FloatND:
            return prefix.evaluate(
                nodes=nodes, values=values, query=probe, valid_count=valid_count
            )

        result = search(
            objective=objective,
            nodes=nodes,
            node_values=values,
            golden_iterations=_GOLDEN_ITERATIONS,
            valid_count=valid_count,
        )
        return value, derivative, result.x, result.value, result.coarse_x

    initial_nodes, initial_values = _padded(
        coordinates=_UNIFORM_TWO, surface=_Surface.INCREASING
    )
    compiled = (
        jax.jit(read)
        .lower(nodes=initial_nodes, values=initial_values, valid_count=jnp.int32(2))
        .compile()
    )
    for (coordinates, surface), reference in zip(
        reference_cases, references, strict=True
    ):
        nodes, values = _padded(coordinates=coordinates, surface=surface)
        actual = compiled(
            nodes=nodes, values=values, valid_count=jnp.int32(len(coordinates))
        )
        jax.block_until_ready(actual)
        for index in (0, 1, 3):
            np.testing.assert_array_almost_equal(
                actual[index], reference[index], decimal=DECIMAL_PRECISION
            )
        np.testing.assert_allclose(
            actual[2], reference[2], rtol=0.0, atol=_location_atol(value=reference[2])
        )
        np.testing.assert_array_equal(actual[4], reference[4])
