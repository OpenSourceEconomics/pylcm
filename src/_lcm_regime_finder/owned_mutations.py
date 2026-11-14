"""Immutable admission authority for every checker-owned filesystem mutation.

Adapted from RF10-REF1. Admission covers supplied names, resolved targets and
existing inode aliases before mutation. Each actual writer consumes its role.
The contract assumes a single owner without concurrent namespace changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .input_reads import InputIdentity, discover_inputs


OPERATIONS = frozenset({"mkdir", "unlink", "open_w", "open_wb", "write_text", "replace"})


def absolute_name(path):
    """Keep spelling without collapsing '..' across a possible symlink."""
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def inode_if_present(path):
    try:
        record = path.stat()
    except FileNotFoundError:
        return None
    return record.st_dev, record.st_ino


@dataclass(frozen=True)
class Mutation:
    role: str
    operation: str
    path: Path


@dataclass(frozen=True)
class AdmittedMutation:
    role: str
    operation: str
    supplied: Path
    resolved: Path


@dataclass(frozen=True)
class OwnedMutationPlan:
    inputs: tuple[InputIdentity, ...]
    roots: tuple[tuple[Path, Path], ...]
    mutations: tuple[AdmittedMutation, ...]
    reads: object

    @classmethod
    def admit(cls, *, inputs, owned_roots, mutations, reads=None):
        """Either validate the entire finite operation tuple or mutate nothing.

        Each mutation has a unique role. Multiple operations on one pathname
        receive separate roles; a caller cannot silently change the registered
        operation. The upstream adapter is responsible for a complete tuple,
        and every downstream writer must consume this object exclusively.
        """
        reads = discover_inputs(inputs) if reads is None else reads
        original = reads.identities
        roots = tuple((absolute_name(root), absolute_name(root).resolve())
                      for root in owned_roots)
        for item in original:
            for supplied_root, resolved_root in roots:
                if (item.supplied.is_relative_to(supplied_root)
                        or item.resolved.is_relative_to(resolved_root)):
                    raise ValueError(f"Owned output namespace contains original input: {item.supplied}")

        admitted = []
        roles = set()
        for mutation in mutations:
            if mutation.role in roles:
                raise ValueError(f"Duplicate mutation role: {mutation.role}")
            roles.add(mutation.role)
            if mutation.operation not in OPERATIONS:
                raise ValueError(f"Unregistered mutation operation: {mutation.operation}")
            supplied = absolute_name(mutation.path)
            resolved = supplied.resolve()
            candidate_inode = inode_if_present(supplied)
            for item in original:
                if (supplied == item.supplied or resolved == item.resolved
                        or (candidate_inode is not None and candidate_inode == item.inode)):
                    raise ValueError(f"Mutation {mutation.role} aliases original input: {item.supplied}")
            admitted.append(AdmittedMutation(mutation.role, mutation.operation, supplied, resolved))
        return cls(tuple(original), roots, tuple(admitted), reads)

    def path_for(self, role, operation):
        """Writers request an admitted operation; missing or changed roles fail."""
        for mutation in self.mutations:
            if mutation.role == role:
                if mutation.operation != operation:
                    raise ValueError(f"Operation mismatch for {role}: {operation}")
                return mutation.supplied
        raise KeyError(f"Writer role is absent from the admitted plan: {role}")


def resource_mutations(output):
    output = Path(output)
    return (
        Mutation('resource.mkdir', 'mkdir', output),
        Mutation('resource.remove_json', 'unlink', output/'report.json'),
        Mutation('resource.remove_md', 'unlink', output/'report.md'),
        Mutation('resource.log', 'open_wb', output/'analysis-execution.log'),
        Mutation('resource.incomplete_json', 'write_text', output/'report.json'),
        Mutation('resource.incomplete_md', 'write_text', output/'report.md'),
        Mutation('resource.receipt', 'write_text', output/'resource.json'),
    )


def resource_plan(output, *, inputs=()):
    return OwnedMutationPlan.admit(inputs=inputs, owned_roots=[output],
                                   mutations=resource_mutations(output))
