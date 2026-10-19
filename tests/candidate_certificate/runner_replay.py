"""Share direct-flow verifications across calls to the aggregate mutation runner.

`run_direct_flow_mutation_controls` verifies the whole certified corridor once per
mutation, so one serial call costs tens of minutes, and the campaign makes many
calls. Every call verifies the same trees: the fault-injection tests change only
how the runner aggregates verdicts (a case predicate, or one family's population),
never what `verify_direct_candidate_flow` reads. That verifier reads exactly the
bytes of the corridor sources under its root and the module's `_SOURCE_SEALS`, so
a verdict keyed by those is the verdict of every tree with that key.

The first call records the trees the runner will verify, verifies them across a
process pool, and then runs the real runner against those verdicts. A tree that
was not recorded -- a later call with a different population -- is verified on
the spot.
"""

import copy
import functools
import hashlib
import os
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tests.candidate_certificate import direct_flow

type VerdictKey = tuple[tuple[tuple[str, str | None], ...], tuple[tuple[str, str], ...]]
type Seals = tuple[tuple[str, str], ...]
type Verifier = Callable[..., direct_flow.DirectFlowReport]

_MAX_PROCESSES = 4


class DirectFlowControlsReplay:
    """Run the actual aggregate runner, verifying each distinct tree once."""

    def __init__(self) -> None:
        self._verdicts: dict[VerdictKey, direct_flow.DirectFlowReport] = {}
        self.real_verifications = 0

    def key(self, *, repo_root: Path) -> VerdictKey:
        """Return the identity of every input the verifier reads from this tree."""
        root = repo_root.resolve()
        return (
            tuple(
                (relative, _sha256(root / relative))
                for relative in direct_flow._CERTIFIED_CORRIDOR_SOURCES
            ),
            tuple(sorted(direct_flow._SOURCE_SEALS.items())),
        )

    def run(self, *, repo_root: Path) -> direct_flow.DirectMutationControls:
        """Run `run_direct_flow_mutation_controls` with shared verdicts."""
        verify = direct_flow.verify_direct_candidate_flow
        if not self._verdicts:
            self._prewarm(repo_root=repo_root, verify=verify)

        def replay(*, repo_root: Path) -> direct_flow.DirectFlowReport:
            key = self.key(repo_root=repo_root)
            if key not in self._verdicts:
                self._verdicts[key] = verify(repo_root=repo_root)
                self.real_verifications += 1
            return copy.deepcopy(self._verdicts[key])

        direct_flow.verify_direct_candidate_flow = replay  # ty: ignore[invalid-assignment]
        try:
            return direct_flow.run_direct_flow_mutation_controls(repo_root=repo_root)
        finally:
            direct_flow.verify_direct_candidate_flow = verify

    def _prewarm(self, *, repo_root: Path, verify: Verifier) -> None:
        """Record every tree one runner call verifies and verify them in parallel."""
        clean_key = self.key(repo_root=repo_root)
        clean = verify(repo_root=repo_root)
        self._verdicts[clean_key] = clean
        self.real_verifications += 1
        originals = {
            relative: (repo_root / relative).read_bytes()
            for relative in direct_flow._CERTIFIED_CORRIDOR_SOURCES
        }
        recorded: dict[VerdictKey, tuple[dict[str, bytes], Seals]] = {}

        def record(*, repo_root: Path) -> direct_flow.DirectFlowReport:
            key = self.key(repo_root=repo_root)
            if key not in self._verdicts and key not in recorded:
                changed = {
                    relative: (repo_root / relative).read_bytes()
                    for (relative, digest), (_, clean_digest) in zip(
                        key[0], clean_key[0], strict=True
                    )
                    if digest != clean_digest
                }
                recorded[key] = (changed, key[1])
            return copy.deepcopy(clean)

        direct_flow.verify_direct_candidate_flow = record  # ty: ignore[invalid-assignment]
        try:
            direct_flow.run_direct_flow_mutation_controls(repo_root=repo_root)
        finally:
            direct_flow.verify_direct_candidate_flow = verify  # ty: ignore[invalid-assignment]
        keys = tuple(recorded)
        processes = max(1, min(_MAX_PROCESSES, os.cpu_count() or 1))
        with (
            tempfile.TemporaryDirectory(prefix="direct-flow-replay-") as base,
            ProcessPoolExecutor(
                max_workers=processes,
                initializer=functools.partial(
                    _start_worker, originals=originals, base=Path(base)
                ),
            ) as pool,
        ):
            verdicts = pool.map(
                _verify_recorded_tree,
                [recorded[key] for key in keys],
                chunksize=8,
            )
            for key, verdict in zip(keys, verdicts, strict=True):
                self._verdicts[key] = verdict
                self.real_verifications += 1


def _sha256(path: Path) -> str | None:
    """Return the file's digest, or `None` when the tree does not contain it."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


_WORKER_ROOT: Path | None = None
_WORKER_ORIGINALS: Mapping[str, bytes] = {}


def _start_worker(*, originals: Mapping[str, bytes], base: Path) -> None:
    """Materialize one private clean tree for this pool process under `base`."""
    global _WORKER_ROOT, _WORKER_ORIGINALS  # noqa: PLW0603
    root = Path(tempfile.mkdtemp(dir=base)) / "repo"
    for relative, content in originals.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    _WORKER_ROOT = root
    _WORKER_ORIGINALS = originals


def _verify_recorded_tree(
    tree: tuple[Mapping[str, bytes], Seals],
) -> direct_flow.DirectFlowReport:
    """Verify the clean tree with `tree`'s files and seals in place, then restore."""
    changed, seals = tree
    root = _WORKER_ROOT
    if root is None:
        msg = "the replay pool process has no materialized tree"
        raise RuntimeError(msg)
    saved_seals = dict(direct_flow._SOURCE_SEALS)
    try:
        for relative, content in changed.items():
            (root / relative).write_bytes(content)
        direct_flow._SOURCE_SEALS.clear()
        direct_flow._SOURCE_SEALS.update(seals)
        return direct_flow.verify_direct_candidate_flow(repo_root=root)
    finally:
        direct_flow._SOURCE_SEALS.clear()
        direct_flow._SOURCE_SEALS.update(saved_seals)
        for relative in changed:
            (root / relative).write_bytes(_WORKER_ORIGINALS[relative])
