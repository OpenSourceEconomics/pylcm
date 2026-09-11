"""Own admitted Cartesian action products through preflight and serial diagnostics."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from _lcm.simulation.memory import SimulationMemory
from lcm.typing import ActionName, FloatND, IntND


@dataclass(kw_only=True)
class PreflightActionGrids:
    """Reuse exact action products while retaining their coordinates and outputs."""

    memory: SimulationMemory | None
    """Current preflight accounting and pure-operation compiler owner."""
    build: Callable[..., Mapping[ActionName, FloatND | IntND]]
    """Module-level numerical producer using the declared Cartesian order."""
    bindings: dict[
        tuple[object, ...],
        tuple[
            MappingProxyType[ActionName, FloatND | IntND],
            Mapping[ActionName, FloatND | IntND],
        ],
    ] = field(default_factory=dict)
    """Exact identity/layout keys with input owners and admitted output owners."""

    def resolve(
        self,
        *,
        action_names: tuple[ActionName, ...],
        grids: MappingProxyType[ActionName, FloatND | IntND],
        retained_arrays: object,
    ) -> Mapping[ActionName, FloatND | IntND]:
        """Admit each distinct Cartesian product once within the validation call."""
        if self.memory is None:
            return self.build(action_names=action_names, grids=grids)
        self.memory.set_derived(retained_arrays)
        if len(action_names) == 1:
            return self.build(action_names=action_names, grids=grids)
        key = (
            action_names,
            tuple((id(grids[name]), grids[name].sharding) for name in action_names),
        )
        if key not in self.bindings:
            flat = self.memory.run(
                function=self.build,
                arguments={"grids": grids},
                static_arguments={"action_names": action_names},
            )
            self.bindings[key] = (grids, flat)
            self.memory.publish(tree=(grids, flat))
        return self.bindings[key][1]

    def close(self) -> None:
        """Release products after every validation exit, including exceptions."""
        self.bindings.clear()
        if self.memory is not None:
            self.memory.close_unit()
            self.memory.replace_outputs(tree=())
