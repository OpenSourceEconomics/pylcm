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


def test_full_suite_lifts_tiers_and_precision_ownership_but_not_capabilities():
    """Full runs every supported node at the child precision on this machine."""
    gpu_contract = ExecutionContract(
        requires=Capability(device="gpu"),
        coverage=Coverage(backends=(Profile.GPU_SMALL,), precisions="representative"),
        tier=Policy.NIGHTLY,
    )

    assert classify(
        gpu_contract,
        profile=Profile.GPU_SMALL,
        policy=Policy.FULL,
        precision="64",
    ).selected
    assert classify(
        gpu_contract, profile=Profile.CPU, policy=Policy.FULL, precision="64"
    ).capability_skipped


def test_enforced_marker_dimensions_are_parsed_independently():
    """Capability, ownership, isolation, and tier retain separate meanings."""
    contract = contract_from_marker_kwargs(
        {
            "requires": {"device": "gpu", "min_devices": 2},
            "coverage": {
                "backends": ("multi-gpu",),
                "precisions": "both",
            },
            "isolation": {"process": "fresh"},
            "ci": {"tier": "nightly"},
        }
    )

    assert contract.requires.device == "gpu"
    assert contract.requires.min_devices == 2
    assert contract.coverage.precisions == "both"
    assert contract.isolation.process == "fresh"
    assert contract.tier is Policy.NIGHTLY


@pytest.mark.parametrize(
    "markers",
    [
        {"requires": {"native": ("exact_affine",)}},
        {"resources": {"gpu_mem_gb": 16}},
        {"isolation": {"gpu": "exclusive"}},
        {"ci": {"paths": ("src/**",)}},
    ],
)
def test_unenforced_policy_fields_are_rejected(markers):
    """Public policy metadata contains only guarantees the launcher enforces."""
    with pytest.raises(ValueError, match="unknown"):
        contract_from_marker_kwargs(markers)


def test_unknown_marker_arguments_are_rejected():
    """A misspelled policy dimension cannot silently change test coverage."""
    with pytest.raises(ValueError, match=r"requires.*unknown"):
        contract_from_marker_kwargs({"requires": {"devcie": "gpu"}})  # codespell:ignore


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
