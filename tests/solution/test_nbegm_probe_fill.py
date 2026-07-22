"""The NB-EGM build probes fill each budget argument to its annotated rank.

The affinity / interval-constancy probes differentiate the composed budget on
synthetic inputs. A budget DAG mixes 0-d scalar parameters (a rate multiplied
onto the liquid state) with array-valued schedule tables (a row indexed by a
discrete code). No single global fill rank satisfies both — a unit-1D fill
violates a scalar parameter's 0-d contract, a 0-d fill cannot be indexed as a
table — so the probe fills each argument at the rank its own annotation
declares.
"""

import jax.numpy as jnp

from _lcm.solution.nbegm import _array_float_arg_names, _probe_fill
from lcm.typing import ContinuousState, Float1D, FloatND, ScalarFloat


def _rate_term(liquid: ContinuousState, rate_of_return: ScalarFloat) -> FloatND:
    return liquid * rate_of_return


def _table_term(schedule: Float1D, code: int) -> FloatND:
    return schedule[code]


def _reads_rate_as_scalar(rate_of_return: ScalarFloat) -> FloatND:
    return jnp.asarray(rate_of_return)


def _reads_rate_as_array(rate_of_return: Float1D) -> FloatND:
    return rate_of_return


def test_array_float_arg_names_includes_an_array_typed_param() -> None:
    """A leaf param annotated as a 1-D array is marked for unit-1D fill."""
    names = _array_float_arg_names(functions={"table_term": _table_term})
    assert "schedule" in names


def test_array_float_arg_names_excludes_a_scalar_typed_param() -> None:
    """A leaf param annotated as a 0-d scalar is never marked for array fill."""
    names = _array_float_arg_names(functions={"rate_term": _rate_term})
    assert "rate_of_return" not in names


def test_array_float_arg_names_lets_a_scalar_annotation_win_on_conflict() -> None:
    """A param any consumer annotates 0-d stays scalar (else its contract breaks)."""
    names = _array_float_arg_names(
        functions={"a": _reads_rate_as_scalar, "b": _reads_rate_as_array}
    )
    assert "rate_of_return" not in names


def test_probe_fill_gives_a_classified_array_arg_unit_1d() -> None:
    """An arg in the array set fills to shape `(1,)` so a scalar index clamps in."""
    table = _probe_fill(
        "schedule", 1.0, frozenset(), array_float_arg_names=frozenset({"schedule"})
    )
    assert jnp.shape(table) == (1,)


def test_probe_fill_keeps_an_unclassified_float_arg_scalar() -> None:
    """A float arg outside the array set stays 0-d, honouring its scalar contract."""
    scalar = _probe_fill(
        "rate_of_return",
        1.0,
        frozenset(),
        array_float_arg_names=frozenset({"schedule"}),
    )
    assert jnp.ndim(scalar) == 0
