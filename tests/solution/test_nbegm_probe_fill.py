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

from _lcm.solution.nbegm import (
    _annotated_int_arg_names,
    _array_float_arg_names,
    _probe_fill,
)
from lcm.typing import (
    ContinuousState,
    Float1D,
    FloatND,
    IntND,
    ScalarFloat,
    ScalarInt,
)


def _rate_term(liquid: ContinuousState, rate_of_return: ScalarFloat) -> FloatND:
    return liquid * rate_of_return


def _table_term(schedule: Float1D, code: int) -> FloatND:
    return schedule[code]


def _reads_rate_as_scalar(rate_of_return: ScalarFloat) -> FloatND:
    return jnp.asarray(rate_of_return)


def _reads_rate_as_array(rate_of_return: Float1D) -> FloatND:
    return rate_of_return


def _reads_insurance_code(insurance_status: IntND) -> FloatND:
    return jnp.asarray(insurance_status, dtype=jnp.float64)


def _reads_repeal_age(repeal_age: ScalarInt) -> FloatND:
    return jnp.asarray(repeal_age, dtype=jnp.float64)


def _reads_repeal_age_as_float(repeal_age: ScalarFloat) -> FloatND:
    return jnp.asarray(repeal_age)


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


def test_annotated_int_arg_names_includes_a_rank_polymorphic_int_param() -> None:
    """A DAG intermediate annotated `IntND` is marked for an integer fill."""
    names = _annotated_int_arg_names(functions={"reads": _reads_insurance_code})
    assert "insurance_status" in names


def test_annotated_int_arg_names_includes_a_scalar_int_param() -> None:
    """A fixed parameter annotated `ScalarInt` is marked for an integer fill.

    Integer-valued parameters need not back a `DiscreteGrid`; an age threshold is
    an ordinary flat param whose only declaration of integer-ness is its
    annotation.
    """
    names = _annotated_int_arg_names(functions={"reads": _reads_repeal_age})
    assert "repeal_age" in names


def test_annotated_int_arg_names_excludes_a_float_param() -> None:
    """A float-annotated parameter is never marked for an integer fill."""
    names = _annotated_int_arg_names(functions={"rate_term": _rate_term})
    assert "rate_of_return" not in names


def test_annotated_int_arg_names_lets_a_float_annotation_win_on_conflict() -> None:
    """A param any consumer annotates float keeps its float fill.

    An integer fill would violate that consumer, so a name whose annotations
    disagree is left to the float default rather than guessed at.
    """
    names = _annotated_int_arg_names(
        functions={"a": _reads_repeal_age, "b": _reads_repeal_age_as_float}
    )
    assert "repeal_age" not in names


def test_probe_fill_gives_an_annotated_int_arg_an_integer_fill() -> None:
    """An arg classified integer by annotation fills as int32, not float."""
    code = _probe_fill("insurance_status", 1.0, frozenset({"insurance_status"}))
    assert code.dtype == jnp.int32
