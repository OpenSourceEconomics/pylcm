"""Negative controls for the parity comparisons the execution-equivalence tests use.

Each control feeds a deliberately wrong output into the assertion entry point a
parity test actually calls and requires it to be rejected, so a comparator that
admits the defect is caught here rather than by a silently green parity test.
"""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lcm import ExecutionConfig
from tests.execution import test_axis_widths_per_regime as axis_parity
from tests.execution.test_axis_widths_per_regime import (
    assert_agrees_to_ulp as value_checker,
)
from tests.simulation._profile_comparison import assert_public_frames, assert_same_bytes
from tests.simulation.test_subject_batching import (
    _assert_columns_invariant as frame_checker,
)


def _steps_below_one(*, dtype: type[np.floating], steps: int) -> np.ndarray:
    value = np.asarray(1.0, dtype=dtype)
    for _ in range(steps):
        value = np.nextafter(value, np.asarray(0.0, dtype=dtype))
    return value


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("steps", [9, 16])
@pytest.mark.parametrize("sign", [-1, 1])
def test_value_gate_rejects_more_than_eight_steps_across_a_binade_edge(
    *, dtype: type[np.floating], steps: int, sign: int
) -> None:
    """Nine or more representable steps below +-1 fail an eight-step gate."""
    reference = _steps_below_one(dtype=dtype, steps=steps)
    with pytest.raises(AssertionError):
        value_checker(
            got=sign * np.asarray(1.0, dtype=dtype), expected=sign * reference, n_ulp=8
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_value_gate_accepts_exactly_eight_steps(*, dtype: type[np.floating]) -> None:
    """Exactly eight representable steps below 1 pass an eight-step gate."""
    reference = _steps_below_one(dtype=dtype, steps=8)
    value_checker(got=np.asarray(1.0, dtype=dtype), expected=reference, n_ulp=8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_value_gate_rejects_maximum_finite_against_one(
    *, dtype: type[np.floating]
) -> None:
    """The largest finite value never yields a tolerance wide enough to match 1."""
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(AssertionError):
        value_checker(
            got=np.asarray(np.finfo(dtype).max, dtype=dtype),
            expected=np.asarray(1.0, dtype=dtype),
            n_ulp=8,
        )


def test_value_gate_rejects_changed_dtype() -> None:
    """Equal numbers in float64 and float32 are not the same published value."""
    with pytest.raises(AssertionError):
        value_checker(
            got=np.asarray([1.0], dtype=np.float64),
            expected=np.asarray([1.0], dtype=np.float32),
            n_ulp=8,
        )


@pytest.mark.parametrize("column", ["wealth", "consumption"])
def test_panel_rejects_one_step_in_a_state_or_action_column(*, column: str) -> None:
    """A float64 state or action column moved by one step fails the panel."""
    reference = pd.DataFrame(
        {"subject_id": [0], "period": [0], column: np.asarray([1.0], dtype=np.float64)}
    )
    actual = reference.copy()
    actual.loc[0, column] = np.nextafter(np.float64(1.0), np.float64(2.0))
    with pytest.raises(AssertionError):
        frame_checker(baseline=reference, batched=actual)


def test_panel_rejects_a_non_value_dtype_change() -> None:
    """A float32 column delivered as float64 fails the panel."""
    reference = pd.DataFrame(
        {
            "subject_id": [0],
            "period": [0],
            "consumption": np.asarray([1.0], dtype=np.float32),
        }
    )
    with pytest.raises(AssertionError):
        frame_checker(
            baseline=reference, batched=reference.astype({"consumption": np.float64})
        )


def _cancellation_pair() -> tuple[np.ndarray, np.ndarray]:
    """A solved-value leaf pair whose near-zero entry moved 16 of its own steps.

    The work-regime period-2 leaf of the two-regime parity model, solved at the
    planned cell width and with the retirement regime pinned to width 2 on an
    AVX-512 CPU: `-0.0246...` moved 16 representable steps of its own format, a
    quarter of one spacing at the leaf's largest value `1.67...`.
    """
    expected = np.asarray([1.67250914, -0.024619740596607903])
    got = np.asarray([1.67250914, -0.024619740596607848])
    return got, expected


def test_value_gate_rejects_the_recorded_cancellation_pair() -> None:
    """Sixteen own value steps remain outside the declared eight-step budget."""
    got, expected = _cancellation_pair()
    with pytest.raises(AssertionError):
        value_checker(got=got, expected=expected, n_ulp=8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_axis_width_gate_never_borrows_another_elements_magnitude(
    *, dtype: type[np.floating]
) -> None:
    """The axis-width operand bound holds each element to its own operands.

    Each element is its own sole operand here, so its bound is its own eight
    steps however large another element of the leaf is. At scale / 64, eight
    scale spacings are 512 of the small value's own steps. The bound must
    accept eight, and reject nine, sixteen and 512 regardless of sign, scale,
    element order or leaf shape.
    """
    for sign, exponent, steps, reverse, shape in product(
        (-1, 1),
        (-16, 0, 16),
        (8, 9, 16, 512),
        (False, True),
        ((2,), (1, 2), (2, 1)),
    ):
        scale = 1.5 * 2.0**exponent
        expected = np.asarray([scale, sign * scale / 64], dtype=dtype)
        got = expected.copy()
        for _ in range(steps):
            got[1] = np.nextafter(got[1], dtype(sign * np.inf))
        if reverse:
            expected, got = expected[::-1], got[::-1]
        expected, got = expected.reshape(shape), got.reshape(shape)

        def check(*, got: np.ndarray, expected: np.ndarray) -> None:
            axis_parity._assert_within_operand_rounding_bound(
                got=got,
                expected=expected,
                flow=expected.astype(np.float64),
                continuation=np.zeros(expected.shape),
                n_ulp=8,
                err_msg="",
            )

        if steps <= 8:
            check(got=got, expected=expected)
        else:
            with pytest.raises(AssertionError):
                check(got=got, expected=expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize(
    ("steps", "accepted"),
    [
        pytest.param(8, True, id="eight-accepted"),
        pytest.param(9, False, id="nine-rejected"),
        pytest.param(12, False, id="twelve-rejected"),
        pytest.param(13, False, id="thirteen-rejected"),
    ],
)
def test_axis_width_gate_holds_an_ordinary_leaf_to_eight_of_its_own_steps(
    *,
    dtype: type[np.floating],
    steps: int,
    accepted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary value may move eight of its own steps, never nine.

    The leaf `(1, "work")` holds `3/2`, the exact sum of a flow of `1` and a
    continuation of `1/2`. Eight spacings of each operand are twelve steps of
    the value, so an operand-rounding allowance would admit nine and twelve
    steps; the real parity test must not.
    """
    reference = np.asarray([1.5], dtype=dtype)
    actual = reference.copy()
    for _ in range(steps):
        actual = np.nextafter(actual, np.asarray(np.inf, dtype=dtype))
    solutions = iter(
        SimpleNamespace(_engine_view=SimpleNamespace(values={1: {"work": leaf}}))
        for leaf in (reference, actual)
    )
    monkeypatch.setattr(
        axis_parity, "_solve_and_collect_widths", lambda **_: ({}, next(solutions))
    )
    monkeypatch.setattr(
        axis_parity,
        "_bellman_operands",
        lambda **_: (np.asarray([1.0]), np.asarray([0.5])),
    )
    if accepted:
        axis_parity.test_a_per_regime_width_preserves_the_solved_values()
    else:
        with pytest.raises(AssertionError):
            axis_parity.test_a_per_regime_width_preserves_the_solved_values()


@pytest.mark.parametrize(("steps", "accepted"), [(16, True), (512, False)])
def test_axis_width_gate_bounds_the_cancellation_entry_by_its_operands(
    *, steps: int, accepted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cancellation entry may move by roundings of its two operands, no more.

    The low-income, bad-health work value at period 2 is about -0.0246, the sum
    of a flow utility near -0.217 and a continuation near 0.192. Eight spacings
    of each operand are 128 of the entry's own steps: a 16-step move passes the
    real parity test and a 512-step move fails it.
    """
    _, solution = axis_parity._solve_and_collect_widths(config=ExecutionConfig())
    expected_values = solution._engine_view.values
    period, regime = 2, "work"
    leaf = np.array(expected_values[period][regime])
    for _ in range(steps):
        leaf[0, 0, 0] = np.nextafter(leaf[0, 0, 0], leaf.dtype.type(np.inf))
    got_values = {
        p: {
            r: (leaf if (p, r) == (period, regime) else value)
            for r, value in by.items()
        }
        for p, by in expected_values.items()
    }
    solutions = iter(
        SimpleNamespace(_engine_view=SimpleNamespace(values=values))
        for values in (expected_values, got_values)
    )
    monkeypatch.setattr(
        axis_parity,
        "_solve_and_collect_widths",
        lambda **_: ({}, next(solutions)),
    )
    if accepted:
        axis_parity.test_a_per_regime_width_preserves_the_solved_values()
    else:
        with pytest.raises(AssertionError):
            axis_parity.test_a_per_regime_width_preserves_the_solved_values()


def test_panel_accepts_two_steps_in_a_derived_target_column() -> None:
    """A float32 `utility` target recomputed in another chunk may move a few steps.

    Subject chunks of 1, 2 and 4 rows evaluate `log(consumption)` in a differently
    vectorized kernel, which moves `utility` by up to two representable steps on
    rows whose consumption is identical.
    """
    reference = pd.DataFrame(
        {"subject_id": [0, 1], "utility": np.asarray([1.0, -0.21710844], np.float32)}
    )
    actual = reference.copy()
    actual.loc[1, "utility"] = np.float32(-0.21710841)
    frame_checker(baseline=reference, batched=actual)


def test_panel_rejects_a_float_action_one_step_from_one() -> None:
    """An action `1.0` delivered as `1.0000000000000002` fails the panel."""
    reference = pd.DataFrame(
        {"subject_id": [0], "utility": [0.5], "consumption": [1.0], "value": [2.0]}
    )
    actual = reference.copy()
    actual.loc[0, "consumption"] = 1.0000000000000002
    with pytest.raises(AssertionError):
        frame_checker(baseline=reference, batched=actual)


def test_panel_accepts_eight_steps_in_the_value_column() -> None:
    """The published `value` column receives the eight-step allowance."""
    reference = pd.DataFrame(
        {"subject_id": [0], "value": _steps_below_one(dtype=np.float64, steps=8)[None]}
    )
    actual = reference.copy()
    actual.loc[0, "value"] = 1.0
    frame_checker(baseline=reference, batched=actual)


@pytest.mark.parametrize(
    ("got", "expected"),
    [
        (np.asarray([-0.0]), np.asarray([0.0])),
        (
            np.asarray([0x7FF8000000000001], dtype=np.uint64).view(np.float64),
            np.asarray([np.nan]),
        ),
    ],
    ids=["signed_zero", "nan_payload"],
)
def test_same_bytes_rejects_signed_zero_and_nan_payload(
    *, got: np.ndarray, expected: np.ndarray
) -> None:
    """Same-program comparison distinguishes signed zeros and NaN payloads."""
    with pytest.raises(AssertionError):
        assert_same_bytes(got=got, expected=expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_derived_utility_keeps_the_eight_step_boundary_and_same_program_bytes(
    *, dtype: type[np.floating]
) -> None:
    """A named derived value never grants tolerance to same-program replay."""
    reference = pd.DataFrame(
        {"subject_id": [0], "utility": np.asarray([1.0], dtype=dtype)}
    )
    for steps in (8, 9):
        actual = reference.copy()
        value = dtype(1.0)
        for _ in range(steps):
            value = np.nextafter(value, dtype(2.0))
        actual.loc[0, "utility"] = value
        if steps == 8:
            frame_checker(baseline=reference, batched=actual)
        else:
            with pytest.raises(AssertionError):
                frame_checker(baseline=reference, batched=actual)
        with pytest.raises(AssertionError):
            assert_public_frames(
                got=actual,
                expected=reference,
                mode="same_program",
                value_columns=("value", "utility"),
            )
