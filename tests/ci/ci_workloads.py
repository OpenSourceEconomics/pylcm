"""Read-only accessors over the source-bound CPU workload manifest.

`ci-workloads.json` is generated evidence, not hand-maintained: it records every
pytest invocation `cpu.yml` makes (lane, OS, precision, worker count, isolation
kind, selection expression, JUnit filename) together with the test files each
invocation selects, resolved by real `pytest --collect-only` runs (or, for the
hash-sharded `tests/solution` battery, by `tests.ci.shard_test_files`, which is
itself deterministic and needs no collection run). Per-file observed weight
comes from the reviewer's per-file CSV keyed to the frozen head
`29396117e6ff6cc90cc0a1bdad7669554c7c32d9`; a file absent from that CSV is never
assigned a weight of zero, it is listed in `unweighted_files`.

This module only reads the manifest back. Regenerating it is a deliberate,
reviewed step, not something a test or this module does implicitly, and each
kind of regeneration has its own route:

- a new test file ⇒ `tests/ci/generate_ci_workloads.py`, which registers it in
  `general_shard_universe` and `unweighted_files` and re-derives the general
  shards. `--check` exits non-zero when the committed manifest differs from
  regeneration;
- a new or changed `cpu.yml` invocation ⇒ a reviewed hand edit, checked by
  `tests/ci/test_cpu_workflow_contract.py`;
- re-measured root `file_weights` ⇒ a new per-file CSV and a new `frozen_head`;
- re-measured `leg_weights` ⇒ a matching per-leg CSV and an updated leg's
  `junit_source`, with its tested source, selection and input provenance.

`frozen_head` binds the root cross-leg CSV weights. Each `leg_weights` entry
records its own measurement provenance in `junit_source`; updating one leg does
not advance the root freeze or imply that other legs were re-measured.
"""

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

type SourcePath = str


class _InvocationEnvironment(TypedDict, total=False):
    runner_os: str
    precision: int
    backend: str
    devices: int
    workers: int
    isolation: str
    policy: dict[str, bool | str]


class _InvocationShard(TypedDict):
    leg: str
    index: int
    count: int
    assignment: str


class Invocation(TypedDict, total=False):
    id: str
    job: str
    step: str
    environment: _InvocationEnvironment
    selection: str
    shard: _InvocationShard
    deselect: list[str]
    ignore: list[str]
    junit: str
    files: list[str]
    nodeids: list[str]
    no_skips_required: bool
    note: str
    portability_controls: list[str]
    predicted_payload_minutes: float
    reclassified_files: list[str]
    source_only: bool
    unchained: bool


class _FileWeight(TypedDict):
    seconds: float
    count: int


class _LegWeight(TypedDict):
    junit_source: SourcePath
    measurement: str
    seconds: dict[str, float]
    counts: dict[str, int]


class _ShardConfig(TypedDict):
    shards: int
    workers: int


class _LaneSummary(TypedDict):
    job: str
    workers: int
    weighted_minutes: float
    unweighted_file_count: int
    largest_file: str | None
    largest_file_seconds: float | None


class _CeilingBacklog(TypedDict):
    file: str
    seconds_per_test: float
    reason: str


class WorkloadManifest(TypedDict):
    schema_version: str
    frozen_head: str
    source: dict[str, str]
    guardrails: dict[str, float]
    shard_layout: dict[str, dict[str, _ShardConfig]]
    coverage_contributors: list[str]
    general_shard_universe: list[str]
    per_test_ceiling_backlog: list[_CeilingBacklog]
    invocations: list[Invocation]
    file_weights: dict[str, _FileWeight]
    leg_weights: dict[str, _LegWeight]
    unweighted_files: list[str]
    excluded_files: dict[str, str]


MANIFEST_PATH = Path(__file__).with_name("ci-workloads.json")


@lru_cache(maxsize=1)
def load_manifest(*, path: Path = MANIFEST_PATH) -> WorkloadManifest:
    """Return the parsed manifest, cached for the process lifetime."""
    return json.loads(path.read_text(encoding="utf-8"))


def invocations(*, path: Path = MANIFEST_PATH) -> tuple[Invocation, ...]:
    """Return every recorded pytest invocation, in manifest order."""
    return tuple(load_manifest(path=path)["invocations"])


def invocation_by_id(*, invocation_id: str, path: Path = MANIFEST_PATH) -> Invocation:
    """Return one invocation record by its manifest `id`."""
    for inv in invocations(path=path):
        if inv["id"] == invocation_id:
            return inv
    raise KeyError(invocation_id)


def files_for(*, invocation_id: str, path: Path = MANIFEST_PATH) -> tuple[str, ...]:
    """Return the test files one invocation selects, repository-relative POSIX."""
    return tuple(invocation_by_id(invocation_id=invocation_id, path=path)["files"])


def all_manifest_files(*, path: Path = MANIFEST_PATH) -> frozenset[str]:
    """Return the union of every file any invocation selects."""
    return frozenset(f for inv in invocations(path=path) for f in inv["files"])


def excluded_files(*, path: Path = MANIFEST_PATH) -> Mapping[str, str]:
    """Return files under `tests/` deliberately excluded, mapped to their reason."""
    return load_manifest(path=path)["excluded_files"]


def unweighted_files(*, path: Path = MANIFEST_PATH) -> tuple[str, ...]:
    """Return files with no observed weight (never assigned a weight of zero)."""
    return tuple(load_manifest(path=path)["unweighted_files"])


def file_weight_seconds(*, file_: str, path: Path = MANIFEST_PATH) -> float | None:
    """Return one file's observed weight in seconds, or None if unweighted."""
    row = load_manifest(path=path)["file_weights"].get(file_)
    return None if row is None else float(row["seconds"])


def guardrails(*, path: Path = MANIFEST_PATH) -> Mapping[str, float]:
    """Return the manifest's recorded operating budgets (minutes/seconds)."""
    return load_manifest(path=path)["guardrails"]


def invocations_for_job(
    *, job: str, path: Path = MANIFEST_PATH
) -> tuple[Invocation, ...]:
    """Return every invocation recorded under one `cpu.yml` job name."""
    return tuple(inv for inv in invocations(path=path) if inv["job"] == job)


def lane_summary(*, path: Path = MANIFEST_PATH) -> dict[str, _LaneSummary]:
    """Return per-invocation worker count and weighted-minutes total.

    Files with no observed weight are counted separately (`unweighted_files`
    count) rather than folded into the weighted total as zero.
    """
    manifest = load_manifest(path=path)
    weights = manifest["file_weights"]
    summary: dict[str, _LaneSummary] = {}
    for inv in manifest["invocations"]:
        weighted_seconds = 0.0
        unweighted = 0
        largest_file = None
        largest_seconds = -1.0
        for file_ in inv["files"]:
            row = weights.get(file_)
            if row is None:
                unweighted += 1
                continue
            seconds = float(row["seconds"])
            weighted_seconds += seconds
            if seconds > largest_seconds:
                largest_seconds = seconds
                largest_file = file_
        summary[inv["id"]] = {
            "job": inv["job"],
            "workers": inv["environment"]["workers"],
            "weighted_minutes": weighted_seconds / 60.0,
            "unweighted_file_count": unweighted,
            "largest_file": largest_file,
            "largest_file_seconds": None if largest_file is None else largest_seconds,
        }
    return summary


def leg_weights(*, leg: str, path: Path = MANIFEST_PATH) -> Mapping[str, float]:
    """Return observed per-file seconds for one leg of the frozen head.

    A "leg" is one OS/precision combination (`fp64-windows`, `fp32-linux`,
    `fp64-solution`, ...). These are the weights a shard layout must use: the
    cross-leg totals in `file_weights` sum four general legs, so the program
    certificate's 6,966 s there is 37 min on Windows but 27 on Linux fp64.
    Dividing a cross-leg sum by the leg count would understate the expensive
    platform and overstate the cheap one.
    """
    legs = load_manifest(path=path)["leg_weights"]
    if leg not in legs:
        raise KeyError(f"{leg!r} is not a recorded leg: {sorted(legs)}")
    return {f: float(seconds) for f, seconds in legs[leg]["seconds"].items()}


def leg_names(*, path: Path = MANIFEST_PATH) -> tuple[str, ...]:
    """Return every leg name the manifest records observed weights for."""
    return tuple(sorted(load_manifest(path=path)["leg_weights"]))


def general_shard_universe(*, path: Path = MANIFEST_PATH) -> tuple[str, ...]:
    """Return the file universe every general (`notslow`) lane shards over.

    Recorded once in the manifest rather than derived per lane, so the three
    fp64 legs and the fp32 leg provably shard the same set and only their
    weights differ.
    """
    return tuple(load_manifest(path=path)["general_shard_universe"])


def coverage_contributors(*, path: Path = MANIFEST_PATH) -> tuple[str, ...]:
    """Return the artifact name of every lane that contributes to coverage."""
    return tuple(load_manifest(path=path)["coverage_contributors"])


def shard_layout(
    *, path: Path = MANIFEST_PATH
) -> Mapping[str, Mapping[str, _ShardConfig]]:
    """Return the recorded shard counts and worker counts per lane family."""
    return load_manifest(path=path)["shard_layout"]


def node_ids(*, invocation_id: str, path: Path = MANIFEST_PATH) -> tuple[str, ...]:
    """Return explicit node IDs for an invocation that selects by node ID."""
    inv = invocation_by_id(invocation_id=invocation_id, path=path)
    return tuple(inv.get("nodeids", ()))


__all__: Sequence[str] = (
    "MANIFEST_PATH",
    "all_manifest_files",
    "coverage_contributors",
    "excluded_files",
    "file_weight_seconds",
    "files_for",
    "general_shard_universe",
    "guardrails",
    "invocation_by_id",
    "invocations",
    "invocations_for_job",
    "lane_summary",
    "leg_names",
    "leg_weights",
    "load_manifest",
    "node_ids",
    "shard_layout",
    "unweighted_files",
)
