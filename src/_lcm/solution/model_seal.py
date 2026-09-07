"""The bindings a model's callables read, captured when the model is built.

A model's durable identity is computed once, at build, from the user callables
and everything they reference. That identity stays truthful only while those
references stay put, so the walk that computes it also records every global and
closure binding it read. Before a solve or simulation the model checks that each
name is still bound to the very object it saw — an identity comparison per
binding, no hashing — and refuses to run when one has moved.

What the seal does not see: an in-place mutation of a referenced mutable object
(a NumPy array written to, a list appended to) leaves the binding in place and is
unsupported rather than detected.
"""

from dataclasses import dataclass

from _lcm.solution.fingerprint import SealedBinding
from lcm.exceptions import ModelSealError


@dataclass(frozen=True)
class SealedBindings:
    """Every binding the identity walk read, in the order it read them."""

    bindings: tuple[SealedBinding, ...]

    def fail_if_moved(self) -> None:
        """Raise `ModelSealError` naming the first binding that no longer holds."""
        for binding in self.bindings:
            if binding.has_moved():
                kind = "closure variable" if binding.cell is not None else "global"
                msg = (
                    f"{binding.owner} reads the {kind} {binding.name!r}, which was "
                    "rebound after the model was built. A model captures its "
                    "callables and everything they reference at construction; "
                    "build a new Model to run against the new binding."
                )
                raise ModelSealError(msg)


class BindingRecorder:
    """Collect the bindings one identity walk reports."""

    def __init__(self) -> None:
        self._bindings: list[SealedBinding] = []
        self._seen: set[tuple[int, str]] = set()

    def __call__(self, binding: SealedBinding) -> None:
        """Record one binding, once per (container, name)."""
        container = binding.cell if binding.cell is not None else binding.namespace
        key = (id(container), binding.name)
        if key in self._seen:
            return
        self._seen.add(key)
        self._bindings.append(binding)

    def sealed(self) -> SealedBindings:
        """Return the recorded bindings as an immutable seal."""
        return SealedBindings(bindings=tuple(self._bindings))
