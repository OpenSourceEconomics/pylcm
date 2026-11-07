"""The solve program a public `Model` stores carries the requested lottery contraction.

The check walks the stored program without running it, so it sees exactly the
continuation the solve will use.
"""

from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass
from functools import partial
from types import FunctionType, MethodType
from typing import Literal

import pytest

from _lcm.execution.core_program import CoreProgramGraphAware
from _lcm.regime_building.Q_and_F import _ComputeCE, _TargetContinuation
from lcm.solvers import GridSearch
from tests.regime_building.test_lottery_contraction import _model


@pytest.mark.parametrize("contract", [False, True])
@pytest.mark.parametrize("final_z", ["same", "cross"])
@pytest.mark.parametrize("income_kind", ["markov", "iid"])
def test_stored_solve_program_contracts_exactly_when_requested(
    *,
    contract: bool,
    final_z: Literal["same", "cross"],
    income_kind: Literal["markov", "iid"],
) -> None:
    """With the option on, `z` and `income` are contracted and keep no node axes.

    With the option off, nothing is contracted and both keep their node axes.
    """
    model = _model(
        final_z=final_z,
        income_kind=income_kind,
        solver=GridSearch(lottery_contraction=contract),
    )
    kernel = model._regimes["alive"].solution.period_kernels[0]
    programs = (
        kernel.core_programs() if isinstance(kernel, CoreProgramGraphAware) else {}
    )
    observed = {
        _summarize(owned.continuations["final"])
        for program in programs.values()
        for owned in _owned_objects(root=program.function)
        if isinstance(owned, _ComputeCE) and "final" in owned.continuations
    }

    unread_axes = frozenset({"next_q", "next_w", "next_y"})
    expected = (
        (frozenset({"z", "income"}), unread_axes)
        if contract
        else (frozenset(), unread_axes | {"next_z", "next_income"})
    )
    assert observed == {expected}


def _summarize(
    continuation: _TargetContinuation,
) -> tuple[frozenset[str], frozenset[str]]:
    contraction = continuation.lottery_contraction
    contracted = (
        frozenset()
        if contraction is None
        else frozenset(lottery.state_name for lottery in contraction.lotteries)
    )
    return contracted, frozenset(continuation.lottery_axis_names)


def _owned_objects(*, root: object) -> Iterator[object]:  # noqa: PAN001 - walks any stored Python object
    """Yield every object reachable through stored fields and closure bindings."""
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if is_dataclass(current) and not isinstance(current, type):
            pending.extend(getattr(current, field.name) for field in fields(current))
        elif isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, (tuple, list)):
            pending.extend(current)
        elif isinstance(current, partial):
            pending.extend((current.func, current.args, current.keywords))
        elif isinstance(current, FunctionType):
            pending.extend(cell.cell_contents for cell in current.__closure__ or ())
            pending.extend((current.__defaults__, current.__kwdefaults__))
        elif isinstance(current, MethodType):
            pending.extend((current.__self__, current.__func__))
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
