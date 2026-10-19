"""The finite-outer NEGM query traces its native row reader once, not per candidate.

`read_finite_outer_choices` reads the native rows of every outer candidate `O` of a
nested child. Those reads differ only in the candidate's index and are the bulk of
the query's program, so they are one batched program over the candidate axis.
Tracing the row reader once per candidate instead unrolls it `outer_capacity`
times, and the compiler's time and memory grow with it.
"""

import pytest

from _lcm.egm import finite_negm_query
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.test_models.ds2024_housing import build_model, build_params


class _QueryTracedError(Exception):
    """Stop the solve once the first finite-outer query has been traced."""


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_read_finite_outer_choices_traces_native_row_reader_once(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One trace of the finite query traces the native row reader exactly once."""
    traces = []
    original_reader = finite_negm_query._read_finite_native_row
    original_query = finite_negm_query.read_finite_outer_choices

    def counted_reader(**kwargs):
        traces.append(None)
        return original_reader(**kwargs)

    def stop_after_query(**kwargs):
        original_query(**kwargs)
        raise _QueryTracedError

    monkeypatch.setattr(finite_negm_query, "_read_finite_native_row", counted_reader)
    monkeypatch.setattr(
        finite_negm_query, "read_finite_outer_choices", stop_after_query
    )
    model = build_model(
        variant="negm", n_grid=4, n_periods=3, n_consumption=10, n_outer_grid=4
    )
    with pytest.raises(_QueryTracedError):
        model.solve(params=build_params(variant="negm", delta=0.0), log_level="off")

    assert len(traces) == 1
