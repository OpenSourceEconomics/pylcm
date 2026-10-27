"""Retain original declaration/callback buffers before constructor producers run."""

import inspect
from dataclasses import fields, is_dataclass

import jax
import numpy as np

from _lcm.egm.validation_stages import (
    ValidationCaptureHasher,
    _require_original_jit_policy,
)
from _lcm.engine import placed_devices_for_ids
from _lcm.execution.execution_plan import ExecutionHardware
from _lcm.grids import DiscreteGrid
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.residency import require_transfer_headroom
from _lcm.solution.fingerprint import (
    _is_captured_jax_pjit,
    _Reachable,
    _SemanticHasher,
)
from _lcm.typing import DataclassInstance, ScalarInt
from lcm.ages import AgeGrid
from lcm.shocks import Discrete as DiscreteShock


class _DeclarationRootObserver(ValidationCaptureHasher):
    """Reuse exact referenced bindings, without array content hashing or factories."""

    def visit(
        self, *, value: _Reachable, _ignore_beartype_guards: bool = False
    ) -> None:
        if isinstance(value, jax.Array):
            if self._array_recorder is not None:
                self._array_recorder(value)
            return
        if isinstance(value, np.ndarray | np.generic):
            return
        if isinstance(value, DiscreteGrid):
            for nodes in value.materialized_array_roots:
                self.visit(value=nodes.values)
                self.visit(value=nodes.midpoints)
            return
        if isinstance(value, AgeGrid):
            self.visit(value=value.values)
            return
        if type(value) is DiscreteShock:
            self.visit(value=value.categories)
            self.visit(value=value.probs)
            return
        if _is_captured_jax_pjit(value):
            # Passive custody seals the original callback without tracing it;
            # its eager dispatch boundary is left intact for actual execution.
            _require_original_jit_policy(value=value, preserve_eager_boundary=False)
            _SemanticHasher.visit(
                self, value=value, _ignore_beartype_guards=_ignore_beartype_guards
            )
            return
        super().visit(value=value, _ignore_beartype_guards=_ignore_beartype_guards)


def open_declaration_allocations(
    *,
    hardware: ExecutionHardware,
    declarations: tuple[_Reachable, ...],
    regime_id_class: type[DataclassInstance] | None = None,
) -> SimulationEntryAllocations | None:
    """Admit existing roots before the first original declaration producer."""
    if hardware.device_memory_bytes is None:
        return None
    arrays: list[jax.Array] = []
    roots = tuple(
        _declared_category_code_roots(category_class=regime_id_class)
        if declaration is regime_id_class and regime_id_class is not None
        else declaration
        for declaration in declarations
    )
    _DeclarationRootObserver(array_recorder=arrays.append).visit(value=roots)
    inputs = SimulationEntryInputs(arrays=tuple(arrays))
    devices = placed_devices_for_ids(
        submesh_device_ids=(), visible_device_ids=hardware.device_ids
    )
    require_transfer_headroom(
        live=inputs.footprint(solution=None),
        destination_bytes={},
        scratch_bytes={},
        budget_bytes=hardware.device_memory_bytes,
        devices=devices,
    )
    return SimulationEntryAllocations(
        original_inputs=inputs,
        solution=None,
        model_roots=(),
        operations=ProfiledSimulationOperations(),
        devices=devices,
        budget_bytes=hardware.device_memory_bytes,
    )


def _declared_category_code_roots(
    *, category_class: type[DataclassInstance]
) -> tuple[ScalarInt, ...]:
    """Observe only the original code fields of the explicit Model category slot.

    This is resource custody, not class behavior admission. The original Model
    validator still checks code values/order, and referenced callback attributes
    retain the semantic walk's separate static attribute/dependency checks.
    """
    if type(category_class) is not type or not is_dataclass(category_class):
        raise TypeError("Model category custody requires its original dataclass.")
    roots = []
    for category_field in fields(category_class):
        value = inspect.getattr_static(category_class, category_field.name, None)
        if (
            not isinstance(value, jax.Array)
            or value.shape != ()
            or value.dtype != np.dtype(np.int32)
        ):
            raise TypeError(
                "Model category custody requires existing scalar int32 codes."
            )
        roots.append(value)
    return tuple(roots)
