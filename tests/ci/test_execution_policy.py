"""Executable contract for the orthogonal CI test policy."""

import pytest

from tests.ci.execution_policy import (
    Capability,
    Coverage,
    ExecutionContract,
    Policy,
    Profile,
    classify,
    contract_from_marker_kwargs,
)


def test_an_unmarked_test_is_owned_by_the_cpu_pr_lane():
    """Cheap tests run once on CPU instead of being replayed on every GPU."""
    contract = ExecutionContract()

    assert classify(contract, profile=Profile.CPU, policy=Policy.PR).selected
    assert classify(
        contract, profile=Profile.GPU_SMALL, policy=Policy.PR
    ).matrix_deselected


def test_representative_precision_is_fp64_on_cpu_and_fp32_on_small_gpu():
    """Representative coverage follows the profile instead of replaying both."""
    cpu_contract = ExecutionContract()
    gpu_contract = ExecutionContract(
        requires=Capability(device="gpu"),
        coverage=Coverage(backends=(Profile.GPU_SMALL,)),
    )

    assert classify(
        cpu_contract, profile=Profile.CPU, policy=Policy.PR, precision="64"
    ).selected
    assert classify(
        cpu_contract, profile=Profile.CPU, policy=Policy.PR, precision="32"
    ).matrix_deselected
    assert classify(
        gpu_contract, profile=Profile.GPU_SMALL, policy=Policy.PR, precision="32"
    ).selected
    assert classify(
        gpu_contract, profile=Profile.GPU_SMALL, policy=Policy.PR, precision="64"
    ).matrix_deselected


def test_full_suite_lifts_tiers_but_not_hardware_requirements():
    """Full means every supported test on this machine, not imaginary hardware."""
    contract = ExecutionContract(
        requires=Capability(device="gpu"),
        coverage=Coverage(backends=(Profile.GPU_SMALL,)),
        tier=Policy.NIGHTLY,
    )

    assert classify(contract, profile=Profile.GPU_SMALL, policy=Policy.FULL).selected
    assert classify(
        contract, profile=Profile.CPU, policy=Policy.FULL
    ).capability_skipped


def test_marker_dimensions_are_parsed_independently():
    """Eligibility, coverage, cost, isolation, and tier retain separate meanings."""
    contract = contract_from_marker_kwargs(
        {
            "requires": {"device": "gpu", "native": ("exact_affine",)},
            "coverage": {
                "backends": ("gpu-small",),
                "precisions": "both",
            },
            "resources": {"wall": "production", "gpu_mem_gb": 16},
            "isolation": {"process": "fresh", "gpu": "exclusive"},
            "ci": {"tier": "nightly", "paths": ("src/_lcm/egm/**",)},
        }
    )

    assert contract.requires.device == "gpu"
    assert contract.coverage.precisions == "both"
    assert contract.resources.wall == "production"
    assert contract.isolation.process == "fresh"
    assert contract.tier is Policy.NIGHTLY


def test_unknown_marker_arguments_are_rejected():
    """A misspelled policy dimension cannot silently change test coverage."""
    with pytest.raises(ValueError, match=r"requires.*unknown"):
        contract_from_marker_kwargs({"requires": {"devcie": "gpu"}})


@pytest.mark.parametrize(
    ("policy", "selected"),
    [
        (Policy.PR, False),
        (Policy.RELEVANT, False),
        (Policy.EXTENDED, False),
        (Policy.NIGHTLY, True),
        (Policy.FULL, True),
    ],
)
def test_production_tests_enter_only_nightly_or_full(policy, selected):
    """Production-scale witnesses stay out of bounded pull-request policies."""
    contract = ExecutionContract(tier=Policy.NIGHTLY)

    assert classify(contract, profile=Profile.CPU, policy=policy).selected is selected
