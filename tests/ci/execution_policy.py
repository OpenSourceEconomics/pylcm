"""Pure classification rules for pylcm's test execution policy.

The policy keeps four decisions separate:

- capability: whether this machine can execute a test truthfully;
- matrix ownership: which backend owes routine coverage;
- tier: which bounded CI policy includes the test; and
- full-suite expansion: every tier supported by the current machine.

Process isolation and resource admission are represented by pytest markers too,
but scheduling them belongs to the launcher rather than this classifier.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, cast

Precision = Literal["32", "64"]


class Profile(StrEnum):
    """Hardware profiles understood by the CI planner."""

    CPU = "cpu"
    GPU_SMALL = "gpu-small"
    GPU_LARGE = "gpu-large"
    MULTI_GPU = "multi-gpu"

    @property
    def has_gpu(self) -> bool:
        """Return whether this profile provides at least one GPU."""
        return self is not Profile.CPU


class Policy(StrEnum):
    """Increasing CI coverage tiers plus the current-machine full switch."""

    PR = "pr"
    RELEVANT = "relevant"
    EXTENDED = "extended"
    NIGHTLY = "nightly"
    FULL = "full"


_POLICY_ORDER = {
    Policy.PR: 0,
    Policy.RELEVANT: 1,
    Policy.EXTENDED: 2,
    Policy.NIGHTLY: 3,
    Policy.FULL: 4,
}


@dataclass(frozen=True)
class Capability:
    """Hard requirements that decide whether execution would be meaningful."""

    device: Literal["any", "cpu", "gpu"] = "any"
    native: tuple[str, ...] = ()
    min_devices: int = 1
    min_compute_capability: int | None = None
    min_gpu_mem_gb: float | None = None
    os: tuple[str, ...] = ()
    arch: tuple[str, ...] = ()


@dataclass(frozen=True)
class Coverage:
    """Backends and precision legs that owe routine coverage."""

    backends: tuple[Profile, ...] = (Profile.CPU,)
    precisions: Literal["representative", "32", "64", "both", "internal"] = (
        "representative"
    )


@dataclass(frozen=True)
class Resources:
    """Measured resource needs used by admission and budget planning."""

    wall: Literal["xs", "s", "m", "l", "production"] = "s"
    wall_seconds: int | None = None
    host_mem_gb: float | None = None
    gpu_mem_gb: float | None = None
    cpu_cores: int = 1
    compile: Literal["light", "heavy"] = "light"


@dataclass(frozen=True)
class Isolation:
    """Process-global state and physical-device isolation requirements."""

    process: Literal["shared", "fresh"] = "shared"
    gpu: Literal["shared", "exclusive", "multi-device"] = "shared"
    cache: Literal["ordinary", "isolated", "cold", "warm"] = "ordinary"
    env_group: str = "ordinary"
    serial_group: str | None = None


@dataclass(frozen=True)
class ExecutionContract:
    """The execution metadata relevant to static test classification."""

    requires: Capability = field(default_factory=Capability)
    coverage: Coverage = field(default_factory=Coverage)
    resources: Resources = field(default_factory=Resources)
    isolation: Isolation = field(default_factory=Isolation)
    tier: Policy = Policy.PR
    paths: tuple[str, ...] = ()


_MARKER_ARGUMENTS = {
    "requires": {
        "device",
        "native",
        "min_devices",
        "min_compute_capability",
        "min_gpu_mem_gb",
        "os",
        "arch",
    },
    "coverage": {"backends", "precisions"},
    "resources": {
        "wall",
        "wall_seconds",
        "host_mem_gb",
        "gpu_mem_gb",
        "cpu_cores",
        "compile",
    },
    "isolation": {"process", "gpu", "cache", "env_group", "serial_group"},
    "ci": {"tier", "paths"},
}


def contract_from_marker_kwargs(
    markers: Mapping[str, Mapping[str, Any]],
) -> ExecutionContract:
    """Parse and validate the five independent pytest policy dimensions."""
    unknown_markers = set(markers) - set(_MARKER_ARGUMENTS)
    if unknown_markers:
        msg = f"unknown policy markers: {sorted(unknown_markers)!r}"
        raise ValueError(msg)
    for name, kwargs in markers.items():
        unknown = set(kwargs) - _MARKER_ARGUMENTS[name]
        if unknown:
            msg = f"{name} has unknown arguments: {sorted(unknown)!r}"
            raise ValueError(msg)

    requires = markers.get("requires", {})
    coverage = markers.get("coverage", {})
    resources = markers.get("resources", {})
    isolation = markers.get("isolation", {})
    ci = markers.get("ci", {})

    device = cast("Literal['any', 'cpu', 'gpu']", requires.get("device", "any"))
    if device not in {"any", "cpu", "gpu"}:
        msg = f"requires.device has invalid value {device!r}"
        raise ValueError(msg)
    backends = tuple(
        Profile(str(value)) for value in coverage.get("backends", (Profile.CPU,))
    )
    precisions = cast(
        "Literal['representative', '32', '64', 'both', 'internal']",
        coverage.get("precisions", "representative"),
    )
    if precisions not in {"representative", "32", "64", "both", "internal"}:
        msg = f"coverage.precisions has invalid value {precisions!r}"
        raise ValueError(msg)
    tier = Policy(str(ci.get("tier", Policy.PR)))
    if tier is Policy.FULL:
        msg = "ci.tier cannot be 'full'; full is a launcher policy"
        raise ValueError(msg)

    return ExecutionContract(
        requires=Capability(
            device=device,
            native=tuple(str(value) for value in requires.get("native", ())),
            min_devices=int(requires.get("min_devices", 1)),
            min_compute_capability=_optional_int(
                requires.get("min_compute_capability")
            ),
            min_gpu_mem_gb=_optional_float(requires.get("min_gpu_mem_gb")),
            os=tuple(str(value) for value in requires.get("os", ())),
            arch=tuple(str(value) for value in requires.get("arch", ())),
        ),
        coverage=Coverage(backends=backends, precisions=precisions),
        resources=Resources(
            wall=cast(
                "Literal['xs', 's', 'm', 'l', 'production']",
                resources.get("wall", "s"),
            ),
            wall_seconds=_optional_int(resources.get("wall_seconds")),
            host_mem_gb=_optional_float(resources.get("host_mem_gb")),
            gpu_mem_gb=_optional_float(resources.get("gpu_mem_gb")),
            cpu_cores=int(resources.get("cpu_cores", 1)),
            compile=cast(
                "Literal['light', 'heavy']", resources.get("compile", "light")
            ),
        ),
        isolation=Isolation(
            process=cast(
                "Literal['shared', 'fresh']", isolation.get("process", "shared")
            ),
            gpu=cast(
                "Literal['shared', 'exclusive', 'multi-device']",
                isolation.get("gpu", "shared"),
            ),
            cache=cast(
                "Literal['ordinary', 'isolated', 'cold', 'warm']",
                isolation.get("cache", "ordinary"),
            ),
            env_group=str(isolation.get("env_group", "ordinary")),
            serial_group=(
                None
                if isolation.get("serial_group") is None
                else str(isolation["serial_group"])
            ),
        ),
        tier=tier,
        paths=tuple(str(value) for value in ci.get("paths", ())),
    )


def _optional_int(value: Any | None) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any | None) -> float | None:
    return None if value is None else float(value)


class DispositionKind(StrEnum):
    """The exhaustive outcomes of classifying one node in one lane."""

    SELECTED = "selected"
    POLICY_DESELECTED = "policy-deselected"
    MATRIX_DESELECTED = "matrix-deselected"
    CAPABILITY_SKIPPED = "capability-skipped"


@dataclass(frozen=True)
class Disposition:
    """One reconciled classification and its stable reason."""

    kind: DispositionKind
    reason: str

    @property
    def selected(self) -> bool:
        """Return whether this node enters the concrete pytest child."""
        return self.kind is DispositionKind.SELECTED

    @property
    def policy_deselected(self) -> bool:
        """Return whether the node belongs to a broader policy tier."""
        return self.kind is DispositionKind.POLICY_DESELECTED

    @property
    def matrix_deselected(self) -> bool:
        """Return whether routine coverage belongs to another backend."""
        return self.kind is DispositionKind.MATRIX_DESELECTED

    @property
    def capability_skipped(self) -> bool:
        """Return whether this machine cannot execute the node truthfully."""
        return self.kind is DispositionKind.CAPABILITY_SKIPPED


def classify(  # noqa: PLR0911
    contract: ExecutionContract,
    *,
    profile: Profile,
    policy: Policy,
    precision: Precision | None = None,
) -> Disposition:
    """Classify one test on one hardware profile under one CI policy."""
    required_device = contract.requires.device
    if required_device == "gpu" and not profile.has_gpu:
        return Disposition(DispositionKind.CAPABILITY_SKIPPED, "requires a GPU")
    if required_device == "cpu" and profile.has_gpu:
        return Disposition(
            DispositionKind.CAPABILITY_SKIPPED,
            "requires a CPU backend",
        )
    if contract.requires.min_devices > 1 and profile is not Profile.MULTI_GPU:
        return Disposition(
            DispositionKind.CAPABILITY_SKIPPED,
            f"requires {contract.requires.min_devices} physical devices",
        )

    if policy is not Policy.FULL and profile not in contract.coverage.backends:
        return Disposition(
            DispositionKind.MATRIX_DESELECTED,
            f"routine coverage belongs to {contract.coverage.backends!r}",
        )

    if precision is not None and not _precision_matches(
        coverage=contract.coverage,
        profile=profile,
        precision=precision,
    ):
        return Disposition(
            DispositionKind.MATRIX_DESELECTED,
            f"the {precision}-bit leg does not own this precision obligation",
        )

    if _POLICY_ORDER[policy] < _POLICY_ORDER[contract.tier]:
        return Disposition(
            DispositionKind.POLICY_DESELECTED,
            f"requires the {contract.tier.value} policy",
        )

    return Disposition(DispositionKind.SELECTED, "selected")


def _precision_matches(
    *, coverage: Coverage, profile: Profile, precision: Precision
) -> bool:
    """Return whether one concrete child owns the declared precision obligation."""
    obligation = coverage.precisions
    if obligation == "both":
        return True
    if obligation in {"32", "64"}:
        return precision == obligation
    representative = "64" if profile is Profile.CPU else "32"
    return precision == representative
