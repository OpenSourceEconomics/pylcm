"""NB-EGM metadata collection sees phased declarations and rejects bad schedules."""

import pytest

import lcm
from _lcm.egm.nbegm import collect_nbegm_metadata
from lcm.exceptions import NBEGMCaseError
from lcm.phased import Phased


@lcm.piecewise_affine(
    output="subsidy",
    variable="income",
    breakpoints=(lcm.affine_breakpoint(threshold="cutoff", kind="jump"),),
)
def subsidy(income, cutoff):
    return income - cutoff


@lcm.piecewise_affine(
    output="subsidy",
    variable="income",
    breakpoints=(
        lcm.affine_breakpoint(threshold="other_cutoff", kind="continuous_kink"),
    ),
)
def subsidy_again(income, other_cutoff):
    return income - other_cutoff


@lcm.piecewise_affine(output="rebate", variable="income", breakpoints=())
def rebate(income):
    return income


def simulate_subsidy(income, cutoff):
    return income - cutoff


def test_collect_reads_a_schedule_declared_on_a_phased_solve_variant() -> None:
    """A `Phased` entry contributes the declaration its solve variant carries."""
    registry = collect_nbegm_metadata(
        functions={"subsidy": Phased(solve=subsidy, simulate=simulate_subsidy)}
    )
    assert registry.piecewise_affine_schedules[0].output == "subsidy"


def test_collect_ignores_a_none_entry() -> None:
    """A `None` entry (a model-level broadcast mask) contributes no declaration."""
    registry = collect_nbegm_metadata(functions={"mask": None, "subsidy": subsidy})
    assert len(registry.piecewise_affine_schedules) == 1


def test_collect_rejects_an_uninspectable_entry() -> None:
    """An entry that is neither callable nor phased is named and refused."""
    with pytest.raises(NBEGMCaseError, match=r"'threshold_table'.*cannot inspect"):
        collect_nbegm_metadata(functions={"threshold_table": (1.0, 2.0)})


def test_collect_rejects_two_schedules_declaring_the_same_output() -> None:
    """Two schedules on one output would collide in the threshold parameter names."""
    with pytest.raises(NBEGMCaseError, match=r"both declare the output 'subsidy'"):
        collect_nbegm_metadata(
            functions={"subsidy": subsidy, "subsidy_again": subsidy_again}
        )


def test_collect_rejects_a_schedule_without_breakpoints() -> None:
    """An empty schedule still routes the regime through the breakpoint kernels."""
    with pytest.raises(NBEGMCaseError, match=r"'rebate' declares no breakpoint"):
        collect_nbegm_metadata(functions={"rebate": rebate})
