"""Declared JAX options forwarded by compilation observers."""

from typing import TypedDict

import jax
from jax._src.interpreters.mlir import LoweringParameters

type CompilerOptions = dict[str, bool | int | float | str]


class LowerOptions(TypedDict, total=False):
    lowering_platforms: tuple[str, ...] | None
    _private_parameters: LoweringParameters | None


class CompileOptions(TypedDict, total=False):
    device_assignment: tuple[jax.Device, ...] | None
