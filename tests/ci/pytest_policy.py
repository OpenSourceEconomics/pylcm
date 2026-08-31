"""Pytest adapter for the pure CI execution-policy classifier."""

import argparse
import json
from pathlib import Path

import jax
import pytest

from tests.ci.execution_policy import (
    Capability,
    Coverage,
    DispositionKind,
    ExecutionContract,
    Policy,
    Profile,
    classify,
    contract_from_marker_kwargs,
)

_POLICY_MARKERS = ("requires", "coverage", "isolation", "ci")
_REPORT_KEY: pytest.StashKey[list[dict[str, object]]] = pytest.StashKey()
_PROFILE_KEY: pytest.StashKey[Profile | None] = pytest.StashKey()
_POLICY_KEY: pytest.StashKey[Policy | None] = pytest.StashKey()


def add_options(parser: pytest.Parser) -> None:
    """Register policy options without changing ordinary pytest selection."""
    group = parser.getgroup("pylcm execution policy")
    group.addoption(
        "--full-suite",
        action="store_true",
        help="Run every declared tier supported by the current machine.",
    )
    group.addoption(
        "--ci-policy",
        choices=("pr", "relevant", "extended", "nightly", "full"),
        default=None,
    )
    group.addoption(
        "--hardware-profile",
        choices=("auto", "cpu", "gpu-small", "gpu-large", "multi-gpu"),
        default="auto",
    )
    group.addoption("--selection-report", default=None, metavar="PATH")
    group.addoption(
        "--isolation-mode",
        choices=("all", "shared", "fresh"),
        default="all",
        help="Run only tests assigned to one process-isolation group.",
    )
    group.addoption("--policy-child", action="store_true", help=argparse.SUPPRESS)


def configure(config: pytest.Config) -> None:
    """Resolve the concrete policy/profile and reject contradictory options."""
    config.stash[_REPORT_KEY] = []
    config.stash[_PROFILE_KEY] = None
    config.stash[_POLICY_KEY] = None
    full_suite = config.getoption("--full-suite")
    policy_name = config.getoption("--ci-policy")
    if full_suite and policy_name is not None:
        msg = "--full-suite and --ci-policy are mutually exclusive"
        raise pytest.UsageError(msg)
    if not full_suite and policy_name is None:
        return
    if policy_name == "full" and not config.getoption("--policy-child"):
        msg = "use --full-suite; --ci-policy=full is private to the launcher"
        raise pytest.UsageError(msg)

    policy = Policy.FULL if full_suite else Policy(policy_name)
    profile = _resolve_profile(config.getoption("--hardware-profile"))
    _validate_explicit_profile(
        requested=config.getoption("--hardware-profile"), resolved=profile
    )
    config.stash[_POLICY_KEY] = policy
    config.stash[_PROFILE_KEY] = profile


def apply(*, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Classify, reconcile, and select all collected nodes for one child."""
    policy = config.stash[_POLICY_KEY]
    profile = config.stash[_PROFILE_KEY]
    if policy is None or profile is None:
        return

    precision = config.getoption("--precision")
    isolation_mode = config.getoption("--isolation-mode")
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    records = config.stash[_REPORT_KEY]
    for item in items:
        try:
            contract = _contract_from_item(item)
        except (TypeError, ValueError) as error:
            msg = f"{item.nodeid}: {error}"
            raise pytest.UsageError(msg) from error
        if isolation_mode not in {"all", contract.isolation.process}:
            records.append(
                {
                    "nodeid": item.nodeid,
                    "disposition": "isolation-deselected",
                    "reason": (
                        f"belongs to {contract.isolation.process!r} process group"
                    ),
                    "profile": profile.value,
                    "precision": precision,
                    "policy": policy.value,
                }
            )
            deselected.append(item)
            continue
        disposition = classify(
            contract=contract,
            profile=profile,
            policy=policy,
            precision=precision,
        )
        records.append(
            {
                "nodeid": item.nodeid,
                "disposition": disposition.kind.value,
                "reason": disposition.reason,
                "profile": profile.value,
                "precision": precision,
                "policy": policy.value,
            }
        )
        if disposition.kind is DispositionKind.CAPABILITY_SKIPPED:
            item.add_marker(pytest.mark.skip(reason=disposition.reason))
            selected.append(item)
        elif disposition.selected:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def write_report(config: pytest.Config) -> None:
    """Write one exhaustive, deterministic selection report when requested."""
    path_value = config.getoption("--selection-report")
    if path_value is None or config.stash[_POLICY_KEY] is None:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(
        config.stash[_REPORT_KEY], key=lambda record: str(record["nodeid"])
    )
    payload = {"schema": 1, "tests": records}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _contract_from_item(item: pytest.Item) -> ExecutionContract:
    marker_kwargs: dict[str, dict[str, object]] = {}
    for name in _POLICY_MARKERS:
        markers = list(item.iter_markers(name=name))
        if len(markers) > 1:
            msg = f"@pytest.mark.{name} is declared more than once"
            raise ValueError(msg)
        if not markers:
            continue
        marker = markers[0]
        if marker.args:
            msg = f"@pytest.mark.{name} takes keyword arguments only"
            raise ValueError(msg)
        marker_kwargs[name] = dict(marker.kwargs)

    contract = contract_from_marker_kwargs(marker_kwargs)
    if item.get_closest_marker("gpu") is not None and "requires" not in marker_kwargs:
        contract = ExecutionContract(
            requires=Capability(device="gpu"),
            coverage=Coverage(
                backends=(Profile.GPU_SMALL, Profile.GPU_LARGE),
                precisions=contract.coverage.precisions,
            ),
            isolation=contract.isolation,
            tier=contract.tier,
        )
    return contract


def _resolve_profile(requested: str) -> Profile:
    if requested != "auto":
        return Profile(requested)
    return Profile.CPU if jax.default_backend() == "cpu" else Profile.GPU_SMALL


def _validate_explicit_profile(*, requested: str, resolved: Profile) -> None:
    if requested == "auto":
        return
    actual_has_gpu = jax.default_backend() != "cpu"
    if resolved.has_gpu and not actual_has_gpu:
        msg = f"hardware profile {requested!r} requires a GPU backend"
        raise pytest.UsageError(msg)
    if resolved is Profile.CPU and actual_has_gpu:
        msg = "hardware profile 'cpu' requires JAX to select the CPU backend"
        raise pytest.UsageError(msg)
    if resolved is Profile.MULTI_GPU and len(jax.devices()) < 2:
        msg = "hardware profile 'multi-gpu' requires at least two JAX devices"
        raise pytest.UsageError(msg)
