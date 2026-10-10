import pytest
from beartype import beartype

from lcm._solver_api.beartype_conf import SOLVER_API_CONF
from lcm.exceptions import SolverAPITypeError


@beartype(conf=SOLVER_API_CONF)
def _echo_int(*, value: int) -> int:
    return value


@beartype(conf=SOLVER_API_CONF)
def _misreport_int(*, value: int) -> int:
    return str(value)  # ty: ignore[invalid-return-type]


def test_solver_api_conf_reports_a_wrong_argument_as_a_type_error() -> None:
    """A wrongly typed argument at a solver-API boundary is a `TypeError`."""
    with pytest.raises(TypeError, match="value"):
        _echo_int(value="1")  # ty: ignore[invalid-argument-type]


def test_solver_api_conf_reports_a_wrong_return_as_the_solver_api_error() -> None:
    """A wrongly typed return at a solver-API boundary is a `SolverAPITypeError`."""
    with pytest.raises(SolverAPITypeError):
        _misreport_int(value=1)
