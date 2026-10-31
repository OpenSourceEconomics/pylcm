"""A NaN value names the exact envelope's run capacity where that envelope solves.

`ExactEnvelope` NaN-poisons a row whose candidate chain stacks more runs over one
resource cell than `max_runs` holds, so the NaN error of a regime it solves names
that capacity and how to raise it. Regimes solved any other way do not.
"""

import dataclasses

import pytest

from lcm import AgeGrid, Model
from lcm.exceptions import InvalidValueFunctionError
from lcm.solvers import ExactEnvelope, RFCEnvelope
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    DCEGM_SOLVER,
    dcegm_retirement,
    dead,
    get_retirement_only_params,
)

_N_PERIODS = 3


def _dcegm_model(*, envelope: ExactEnvelope | RFCEnvelope) -> Model:
    """Build the DC-EGM retirement model under one upper-envelope backend."""
    ages = AgeGrid(start=40, inclusive_stop=40 + (_N_PERIODS - 1) * 10, step="10Y")
    retirement = dataclasses.replace(
        dcegm_retirement, solver=dataclasses.replace(DCEGM_SOLVER, envelope=envelope)
    )
    return Model(
        regimes={"retirement": retirement, "dead": dead},
        ages=ages,
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
        initial_nodes={ages.exact_values[0]: "retirement"},
        edges=retirement_only.retirement_edges(ages),
    )


def _nan_error_text(*, model: Model) -> str:
    """Solve under a NaN discount factor; return the error with its notes."""
    params = get_retirement_only_params(
        n_periods=_N_PERIODS, discount_factor=float("nan")
    )
    with pytest.raises(InvalidValueFunctionError, match="are NaN") as error:
        model.solve(params=params, log_level="off")
    return "\n".join((str(error.value), *getattr(error.value, "__notes__", ())))


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_exact_envelope_nan_error_names_its_run_capacity() -> None:
    """The error names `max_runs=7` and the remedy of raising it."""
    text = _nan_error_text(model=_dcegm_model(envelope=ExactEnvelope(max_runs=7)))

    assert "max_runs=7" in text
    assert "ExactEnvelope(max_runs=...)" in text


def test_other_envelope_nan_error_omits_the_run_capacity() -> None:
    """A DC-EGM regime on another envelope gets no capacity hint."""
    text = _nan_error_text(model=_dcegm_model(envelope=RFCEnvelope()))

    assert "max_runs" not in text


def test_grid_search_nan_error_omits_the_run_capacity() -> None:
    """A brute-force regime gets no capacity hint."""
    text = _nan_error_text(model=retirement_only.get_model(_N_PERIODS))

    assert "max_runs" not in text
