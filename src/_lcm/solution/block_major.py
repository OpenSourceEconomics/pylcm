"""Block-major lifetime execution: one invariant component at a time.

Under `InvariantBlockSchedule.BLOCK_MAJOR`, a model whose every regime carries
the blocked state is solved one *component* — one code of that state — at a
time. A component runs through the ordinary backward-induction engine on a view
of the canonical regimes in which the state's grid holds only that code, so
every period of the component is solved before the next code starts:

- `component_regimes` narrows each regime's grid of the state to the
  component's code, and each bound program to the one evaluating that code, at
  position zero of the component's values. Nothing else of a regime changes:
  functions, parameters, transitions, grids, labels and fingerprints are the
  model's own, and the code keeps its value.
- `ComponentSchedule` runs one component through `backward_induction.solve`,
  sharing one `ExecutableCache` across components, so every component runs the
  executables the first one compiled.
- `RetainedComponentValues` copies a finished component's values to the host
  and deletes its device buffers. Its `ComponentCoverage` publishes a complete
  result only once every code is covered exactly once over the whole solved
  domain, and its value store assembles one value, on the layout the
  period-major solve publishes, only when that value is read.

Forward simulation reads a component's values through a `ComponentValueSource`:
`SolvingComponentValues` solves each component as its subjects are simulated,
and `UploadedComponentValues` places a retained component back on the device.

A component job solves a selection of the codes: `selected_components` scopes
the schedules a solve or a simulation starts to those codes. A selected
schedule retains its codes on the host and never publishes a result; its
simulation holds the rows of its codes' subjects alone.
"""

import dataclasses
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias

import jax
import numpy as np

from _lcm.engine import Regime
from _lcm.execution.core_program import InvariantBinding, core_program_graph
from _lcm.execution.footprint import layout_footprint
from _lcm.execution.invariant_blocks import block_state_action_space
from _lcm.solution import backward_induction
from _lcm.solution.backward_induction import ExecutableCache
from _lcm.solution.contract import BackwardInductionResult
from _lcm.solution.grid_search import _GridSearchPeriodKernel
from _lcm.solution.v_topology import _get_regime_V_shapes_and_shardings
from _lcm.typing import FlatParams, RegimeName, StateName
from lcm._solver_api.entries import _LazyEntry
from lcm._solver_api.identity import LoadState
from lcm._solver_api.stores import ValueStore, _ValueStoreBoundary
from lcm.exceptions import ExecutionPlanningError

if TYPE_CHECKING:
    _ComponentBlocks: TypeAlias = Mapping[int, Mapping[RegimeName, jax.Array]]  # noqa: UP040
else:
    # A component's blocks are read and deleted by the schedule itself; the
    # runtime annotation check must not walk them while they are being retired.
    _ComponentBlocks = object

_REMEDY = (
    "Keep the default InvariantBlockSchedule.PERIOD_MAJOR, or remove the state "
    "from ExecutionConfig.invariant_block_widths."
)

type _Coordinate = tuple[int, RegimeName]


@dataclass(frozen=True, kw_only=True)
class InvariantComponent:
    """One code of the blocked state, solved as an independent problem."""

    state_name: StateName
    """Name of the blocked state."""

    start: int
    """Position of the code on the state's grid."""

    code: int
    """Canonical code at that position; it keeps its value inside the component."""


def invariant_components(
    *, regimes: Mapping[RegimeName, Regime], state_name: StateName
) -> tuple[InvariantComponent, ...]:
    """Return one component per code of the blocked state, in grid order.

    Raises:
        ExecutionPlanningError: Two regimes hold different codes for the state.

    """
    grids = {
        name: tuple(
            int(code)
            for code in regime.solution._base_state_action_space.states[  # noqa: SLF001
                state_name
            ].tolist()
        )
        for name, regime in regimes.items()
    }
    codes = next(iter(grids.values()))
    if any(other != codes for other in grids.values()):
        msg = (
            f"The block-major schedule takes one component per code of "
            f"{state_name!r}, but the regimes hold different codes: {grids!r}. "
            f"{_REMEDY}"
        )
        raise ExecutionPlanningError(msg)
    return tuple(
        InvariantComponent(state_name=state_name, start=start, code=code)
        for start, code in enumerate(codes)
    )


def component_regimes(
    *, regimes: Mapping[RegimeName, Regime], component: InvariantComponent
) -> MappingProxyType[RegimeName, Regime]:
    """Return the regimes as one component solves them.

    Each regime's grid of the state holds the component's code alone, built on
    the host in the grid's dtype and placed like the grid, so binding a code
    compiles no program. Each kernel bound to the state keeps only the program
    evaluating that code, rebound to position zero: inside the component every
    value carrying the state holds that one code there.

    Raises:
        ExecutionPlanningError: A kernel binds the state but is not a GridSearch
            kernel, or holds no program for the code.

    """
    return MappingProxyType(
        {
            name: dataclasses.replace(
                regime,
                solution=dataclasses.replace(
                    regime.solution,
                    _base_state_action_space=block_state_action_space(
                        space=regime.solution._base_state_action_space,  # noqa: SLF001
                        binding=_component_binding(component=component),
                    ),
                    period_kernels=MappingProxyType(
                        {
                            period: _component_kernel(
                                kernel=kernel, component=component, regime_name=name
                            )
                            for period, kernel in regime.solution.period_kernels.items()
                        }
                    ),
                ),
            )
            for name, regime in regimes.items()
        }
    )


def _component_binding(*, component: InvariantComponent) -> InvariantBinding:
    """Return the binding that narrows a grid to the component's code."""
    return InvariantBinding(
        state_name=component.state_name,
        start=component.start,
        code=component.code,
        family="component",
    )


def _component_kernel(
    *, kernel: object, component: InvariantComponent, regime_name: RegimeName
) -> object:
    """Keep the kernel's program for the component's code, at position zero."""
    graph = core_program_graph(kernel=kernel)
    if all(program.invariant_binding is None for program in graph.values()):
        return kernel
    if not isinstance(kernel, _GridSearchPeriodKernel):
        msg = (
            f"Regime {regime_name!r} binds {component.state_name!r} in a kernel the "
            f"block-major schedule cannot narrow to one code. {_REMEDY}"
        )
        raise ExecutionPlanningError(msg)
    selected = {
        name: dataclasses.replace(
            program,
            invariant_binding=dataclasses.replace(binding, start=0),
        )
        for name, program in graph.items()
        if (binding := program.invariant_binding) is not None
        and binding.state_name == component.state_name
        and binding.code == component.code
    }
    if len(selected) != 1:
        msg = (
            f"Regime {regime_name!r} holds {len(selected)} programs for code "
            f"{component.code} of {component.state_name!r}; a component needs "
            f"exactly one. {_REMEDY}"
        )
        raise ExecutionPlanningError(msg)
    return dataclasses.replace(kernel, _core_programs=MappingProxyType(selected))


@dataclass(frozen=True, kw_only=True)
class ComponentCoverage:
    """Which codes a block-major result holds, and over which solved domain.

    A result is published only from a complete coverage: every code of the
    state, each exactly once, each over exactly `entries`.
    """

    state_name: StateName
    """Name of the blocked state."""

    codes: tuple[int, ...]
    """Every code of the state, in grid order."""

    entries: frozenset[_Coordinate]
    """The solved `(period, regime)` coordinates every code must cover."""

    covered: tuple[int, ...] = ()
    """Codes covered so far, in the order they were retained."""

    def with_code(
        self, *, code: int, entries: frozenset[_Coordinate]
    ) -> ComponentCoverage:
        """Return the coverage with `code` added.

        Raises:
            ExecutionPlanningError: The code is not one of the state's, is already
                covered, or covers other coordinates than the solved domain.

        """
        if code not in self.codes:
            msg = (
                f"Code {code} is not a code of {self.state_name!r}, whose codes are "
                f"{self.codes!r}."
            )
            raise ExecutionPlanningError(msg)
        if code in self.covered:
            msg = f"Code {code} of {self.state_name!r} is covered twice."
            raise ExecutionPlanningError(msg)
        if entries != self.entries:
            msg = (
                f"Code {code} of {self.state_name!r} covers "
                f"{sorted(entries)!r}, not the solved domain "
                f"{sorted(self.entries)!r}; missing {sorted(self.entries - entries)!r},"
                f" extra {sorted(entries - self.entries)!r}."
            )
            raise ExecutionPlanningError(msg)
        return dataclasses.replace(self, covered=(*self.covered, code))

    def fail_if_incomplete(self) -> None:
        """Refuse to publish a result missing a code.

        Raises:
            ExecutionPlanningError: Naming the codes not yet covered.

        """
        missing = tuple(code for code in self.codes if code not in self.covered)
        if missing:
            msg = (
                f"The block-major result of {self.state_name!r} is missing codes "
                f"{missing!r}; a partial result is never published."
            )
            raise ExecutionPlanningError(msg)


@dataclass(frozen=True, kw_only=True)
class ValueLayout:
    """Where and how one complete value is published."""

    shape: tuple[int, ...]
    """Shape of the complete value."""

    axis: int
    """Position of the blocked state's axis in the value."""

    sharding: jax.sharding.Sharding
    """Layout the period-major solve publishes the value on."""


@dataclass(frozen=True, kw_only=True)
class ComponentRetentionRecord:
    """What one block-major result retained and moved, in bytes."""

    state_name: StateName
    """Name of the blocked state."""

    codes: tuple[int, ...]
    """Codes retained, in grid order."""

    host_bytes_by_code: MappingProxyType[int, int]
    """Host bytes each code's retained values occupy."""

    device_to_host_bytes: int
    """Bytes copied from devices to the host when components were retained."""

    host_to_device_bytes: int
    """Bytes placed back on devices for simulation, summed over every upload."""

    full_value_bytes_by_device: MappingProxyType[int, int]
    """Bytes every complete value would occupy on each device at once."""

    def to_json(self) -> str:
        """Return the record as one JSON line."""
        return json.dumps(
            {
                "state_name": self.state_name,
                "codes": list(self.codes),
                "host_bytes_by_code": {
                    str(code): count for code, count in self.host_bytes_by_code.items()
                },
                "device_to_host_bytes": self.device_to_host_bytes,
                "host_to_device_bytes": self.host_to_device_bytes,
                "full_value_bytes_by_device": {
                    str(device): count
                    for device, count in self.full_value_bytes_by_device.items()
                },
            },
            sort_keys=True,
        )


class RetainedComponentValues:
    """The host-retained values of one block-major solve, and their owner.

    Blocks are exact host copies of each component's device values. Nothing
    here is cached on a device: an assembled value or an uploaded component is
    a fresh array owned by its reader. The owner is referenced only by the
    result it backs, so dropping that result frees every block.
    """

    def __init__(
        self,
        *,
        state_name: StateName,
        codes: tuple[int, ...],
        layouts: Mapping[_Coordinate, ValueLayout],
        budget_bytes: int | None,
    ) -> None:
        """Hold an empty retention for `codes` over the solved `layouts`."""
        self._coverage = ComponentCoverage(
            state_name=state_name, codes=codes, entries=frozenset(layouts)
        )
        self._layouts = MappingProxyType(dict(layouts))
        self._coordinates: tuple[_Coordinate, ...] = ()
        self._budget_bytes = budget_bytes
        self._blocks: dict[int, MappingProxyType[_Coordinate, np.ndarray]] = {}
        self._device_to_host_bytes = 0
        self._host_to_device_bytes = 0

    @property
    def coverage(self) -> ComponentCoverage:
        """Return the codes retained so far and the domain each covers."""
        return self._coverage

    @property
    def layouts(self) -> MappingProxyType[_Coordinate, ValueLayout]:
        """Return the published layout of every solved value."""
        return self._layouts

    @property
    def coordinates(self) -> tuple[_Coordinate, ...]:
        """Return every solved coordinate in the order the solve published it.

        Every component publishes in the same order, the period-major loop's,
        which the first retained component records.
        """
        return self._coordinates

    def retain(self, *, code: int, blocks: _ComponentBlocks) -> None:
        """Copy one component's values to the host, then delete them on the device.

        The host copy completes before any buffer is deleted, so a reader still
        in flight on the device finishes first.
        """
        flat = {
            (period, regime): block
            for period, regimes in blocks.items()
            for regime, block in regimes.items()
        }
        coverage = self._coverage.with_code(code=code, entries=frozenset(flat))
        host = jax.device_get(flat)
        self._keep(code=code, coverage=coverage, host=host)
        self._device_to_host_bytes += sum(array.nbytes for array in host.values())
        if not self._coordinates:
            self._coordinates = tuple(flat)
        delete_blocks(blocks=blocks)

    def retain_host(
        self, *, code: int, blocks: Mapping[_Coordinate, np.ndarray]
    ) -> None:
        """Retain one component's host blocks, listed in the solve's publication order.

        The blocks come from where a component job wrote them; they are held as
        given, never placed on a device.

        Raises:
            ExecutionPlanningError: The code is not one of the state's or is
                already covered, the blocks cover other coordinates than the
                solved domain or list them in another order than the codes
                retained before, or a block does not have the component shape.

        """
        coverage = self._coverage.with_code(code=code, entries=frozenset(blocks))
        coordinates = tuple(blocks)
        if self._coordinates and coordinates != self._coordinates:
            msg = (
                f"Component {code} lists its values in the order "
                f"{coordinates!r}, not the order {self._coordinates!r} of the "
                "codes retained before it."
            )
            raise ExecutionPlanningError(msg)
        self._keep(code=code, coverage=coverage, host=blocks)
        if not self._coordinates:
            self._coordinates = coordinates

    def host_blocks(self, *, code: int) -> MappingProxyType[_Coordinate, np.ndarray]:
        """Return one retained component's host blocks in publication order."""
        blocks = self._blocks[code]
        return MappingProxyType(
            {coordinate: blocks[coordinate] for coordinate in self._coordinates}
        )

    def _keep(
        self,
        *,
        code: int,
        coverage: ComponentCoverage,
        host: Mapping[_Coordinate, np.ndarray],
    ) -> None:
        """Hold one component's host blocks once each has the component shape."""
        for coordinate, array in host.items():
            layout = self._layouts[coordinate]
            expected = (
                *layout.shape[: layout.axis],
                1,
                *layout.shape[layout.axis + 1 :],
            )
            if array.shape != expected:
                msg = (
                    f"Component {code} published {coordinate!r} with shape "
                    f"{array.shape}, not the component shape {expected}."
                )
                raise ExecutionPlanningError(msg)
        self._blocks[code] = MappingProxyType(
            {coordinate: np.asarray(array) for coordinate, array in host.items()}
        )
        self._coverage = coverage

    def assemble_host_value(self, *, period: int, regime: RegimeName) -> np.ndarray:
        """Return one complete value on the host, codes in grid order."""
        self._coverage.fail_if_incomplete()
        layout = self._layouts[(period, regime)]
        return np.concatenate(
            [self._blocks[code][(period, regime)] for code in self._coverage.codes],
            axis=layout.axis,
        )

    def assemble_value(self, *, period: int, regime: RegimeName) -> jax.Array:
        """Return one complete value on the layout the period-major solve publishes."""
        return jax.device_put(
            self.assemble_host_value(period=period, regime=regime),
            self._layouts[(period, regime)].sharding,
        )

    def upload(
        self, *, code: int
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]:
        """Place one retained component back on the devices it was solved on."""
        blocks = self._blocks[code]
        placed = {
            coordinate: jax.device_put(array, self._layouts[coordinate].sharding)
            for coordinate, array in blocks.items()
        }
        self._host_to_device_bytes += sum(array.nbytes for array in blocks.values())
        return _nested(flat=placed)

    def value_store(self) -> _ValueStoreBoundary:
        """Return the complete logical store, one lazy entry per solved value."""
        self._coverage.fail_if_incomplete()
        entries: dict[object, object] = {
            coordinate: _ComponentValueEntry(
                owner=self, period=coordinate[0], regime=coordinate[1]
            )
            for coordinate in self._coordinates
        }
        return ValueStore(entries)

    def admit_full_materialization(
        self, *, coordinates: tuple[_Coordinate, ...]
    ) -> None:
        """Refuse to place `coordinates` on the devices at once above the budget.

        Raises:
            ExecutionPlanningError: Naming each device whose need exceeds the
                budget, with the remedies.

        """
        if self._budget_bytes is None:
            return
        need = _bytes_by_device(
            layouts=tuple(self._layouts[coordinate] for coordinate in coordinates),
            item_bytes=self._item_bytes(),
        )
        over = {
            device: count
            for device, count in need.items()
            if count > self._budget_bytes
        }
        if over:
            msg = (
                f"Materializing {len(coordinates)} block-major values at once needs "
                f"{dict(sorted(over.items()))!r} bytes on devices whose budget is "
                f"{self._budget_bytes} bytes. Read the values one at a time with "
                "SolutionResult.value(period=..., regime=...), save the result "
                "with SolutionResult.save, or raise "
                "ExecutionConfig.device_memory_bytes."
            )
            raise ExecutionPlanningError(msg)

    def retention_record(self) -> ComponentRetentionRecord:
        """Return what this retention holds on the host and has moved."""
        return ComponentRetentionRecord(
            state_name=self._coverage.state_name,
            codes=tuple(code for code in self._coverage.codes if code in self._blocks),
            host_bytes_by_code=MappingProxyType(
                {
                    code: sum(array.nbytes for array in self._blocks[code].values())
                    for code in self._coverage.codes
                    if code in self._blocks
                }
            ),
            device_to_host_bytes=self._device_to_host_bytes,
            host_to_device_bytes=self._host_to_device_bytes,
            full_value_bytes_by_device=MappingProxyType(
                _bytes_by_device(
                    layouts=tuple(self._layouts.values()),
                    item_bytes=self._item_bytes(),
                )
            ),
        )

    def _item_bytes(self) -> int:
        """Return the bytes of one value element, from any retained block."""
        for blocks in self._blocks.values():
            for array in blocks.values():
                return array.dtype.itemsize
        return 0


def _bytes_by_device(
    *, layouts: tuple[ValueLayout, ...], item_bytes: int
) -> dict[int, int]:
    """Sum the per-device bytes of complete values on their layouts."""
    totals: dict[int, int] = {}
    for layout in layouts:
        footprint = layout_footprint(
            sharding=layout.sharding, shape=layout.shape, item_bytes=item_bytes
        )
        for device in footprint.device_ids:
            totals[device] = totals.get(device, 0) + footprint.bytes_per_device
    return totals


@dataclass(frozen=True, kw_only=True, eq=False)
class _ComponentValueEntry(_LazyEntry):
    """One complete value of a block-major result, assembled when read."""

    owner: RetainedComponentValues
    """The retention holding every code's block of the value."""

    period: int
    """Period of the value."""

    regime: RegimeName
    """Regime of the value."""

    @property
    def load_state(self) -> LoadState:
        """Nothing of a block-major value is resident until it is read."""
        return LoadState.UNLOADED

    def materialize(self, *, template: object | None = None) -> object:
        """Assemble the value on the layout the period-major solve publishes."""
        del template
        return self.owner.assemble_value(period=self.period, regime=self.regime)

    def host_value(self) -> np.ndarray:
        """Assemble the value on the host, without placing it on a device."""
        return self.owner.assemble_host_value(period=self.period, regime=self.regime)

    @classmethod
    def _admit_joint_materialization(cls, *, entries: tuple[_LazyEntry, ...]) -> None:
        """Refuse to materialize the entries of one owner together above its budget."""
        by_owner: dict[int, tuple[RetainedComponentValues, list[_Coordinate]]] = {}
        for entry in entries:
            if not isinstance(entry, _ComponentValueEntry):
                continue
            owner, coordinates = by_owner.setdefault(id(entry.owner), (entry.owner, []))
            coordinates.append((entry.period, entry.regime))
        for owner, coordinates in by_owner.values():
            owner.admit_full_materialization(coordinates=tuple(coordinates))


def delete_blocks(*, blocks: _ComponentBlocks) -> None:
    """Delete every device buffer of one component's values that is still alive."""
    for regimes in blocks.values():
        for block in regimes.values():
            if isinstance(block, jax.Array) and not block.is_deleted():
                block.delete()


def _nested(
    *, flat: Mapping[_Coordinate, jax.Array]
) -> MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]:
    """Return `(period, regime) -> array` as `period -> regime -> array`."""
    nested: dict[int, dict[RegimeName, jax.Array]] = {}
    for (period, regime), array in sorted(flat.items()):
        nested.setdefault(period, {})[regime] = array
    return MappingProxyType(
        {period: MappingProxyType(regimes) for period, regimes in nested.items()}
    )


class ComponentSchedule:
    """Solve one invariant component at a time and retain each on the host.

    Every component calls the ordinary backward-induction engine with the same
    arguments except the component view of the regimes, and shares one
    executable cache, so codes after the first compile nothing. A component's
    device values belong to the schedule until they are retained; whatever
    fails while they are alive, they are deleted before the error propagates.
    """

    def __init__(
        self,
        *,
        regimes: MappingProxyType[RegimeName, Regime],
        state_name: StateName,
        flat_params: FlatParams,
        device_ids: tuple[int, ...],
        process_grid_resolver: object | None,
        budget_bytes: int | None,
        solve: Callable[..., BackwardInductionResult],
        logger: logging.Logger,
        codes: tuple[int, ...] | None = None,
    ) -> None:
        """Prepare the schedule; no component is solved yet.

        Args:
            regimes: The canonical regimes.
            state_name: The blocked state.
            flat_params: The solve's parameters, which complete runtime grids.
            device_ids: The model's device ids, ascending.
            process_grid_resolver: The solve's process-grid resolver, or `None`.
            budget_bytes: The model's device budget, for full materialization.
            solve: `backward_induction.solve` bound to every argument but the
                regimes and the executable cache.
            logger: Logger receiving the retention record at debug level.
            codes: The codes to solve, or `None` for every code. A selection
                solves its codes in grid order; its retention holds those codes
                alone and so never publishes a result.

        Raises:
            ExecutionPlanningError: A selected code is not a code of the state.

        """
        self._regimes = regimes
        every_component = invariant_components(regimes=regimes, state_name=state_name)
        unknown = (
            ()
            if codes is None
            else tuple(
                code
                for code in codes
                if code not in {component.code for component in every_component}
            )
        )
        if unknown:
            msg = (
                f"Codes {unknown!r} are not codes of {state_name!r}, whose codes "
                f"are {tuple(component.code for component in every_component)!r}."
            )
            raise ExecutionPlanningError(msg)
        self._subject_codes = codes
        self._components = tuple(
            component
            for component in every_component
            if codes is None or component.code in codes
        )
        self._solve = solve
        self._cache = ExecutableCache()
        self._logger = logger
        self._retained = RetainedComponentValues(
            state_name=state_name,
            codes=tuple(component.code for component in every_component),
            layouts=value_layouts(
                regimes=regimes,
                state_name=state_name,
                topology=_get_regime_V_shapes_and_shardings(
                    regimes=regimes,
                    flat_params=flat_params,
                    device_ids=device_ids,
                    process_grid_resolver=process_grid_resolver,  # ty: ignore[invalid-argument-type]
                ),
            ),
            budget_bytes=budget_bytes,
        )

    @property
    def components(self) -> tuple[InvariantComponent, ...]:
        """Return the components in the order they are solved."""
        return self._components

    @property
    def subject_codes(self) -> tuple[int, ...] | None:
        """Return the selected codes, or `None` when every code is solved."""
        return self._subject_codes

    @property
    def retained(self) -> RetainedComponentValues:
        """Return the host retention the components are copied into."""
        return self._retained

    def solve_component(
        self, *, code: int
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]:
        """Run one component through every period; its values stay on the device.

        Raises:
            ExecutionPlanningError: The component published anything but values,
                which this schedule does not retain.

        """
        (component,) = (
            component for component in self._components if component.code == code
        )
        result = self._solve(
            regimes=component_regimes(regimes=self._regimes, component=component),
            executable_cache=self._cache,
        )
        values = result.value_functions
        published = {
            "simulation policies": result.simulation_policies,
            "replay artifacts": result.replay_artifacts,
            "retained continuations": result.retained_continuations,
            "auxiliary artifacts": result.auxiliary_artifacts,
            "solver diagnostics": result.diagnostics,
        }
        extra = [name for name, store in published.items() if _has_payload(store)]
        if extra:
            delete_blocks(blocks=values)
            msg = (
                f"Component {code} of {component.state_name!r} published "
                f"{extra!r}; the block-major schedule retains values only. "
                f"{_REMEDY}"
            )
            raise ExecutionPlanningError(msg)
        return values

    def retain_component(self, *, code: int, blocks: _ComponentBlocks) -> None:
        """Retain one solved component on the host; its device buffers are deleted.

        The buffers are deleted whether or not the retention succeeds.
        """
        try:
            self._retained.retain(code=code, blocks=blocks)
        finally:
            delete_blocks(blocks=blocks)

    def run(self) -> RetainedComponentValues:
        """Solve and retain every component in grid order, then log the record."""
        for component in self._components:
            blocks = self.solve_component(code=component.code)
            self.retain_component(code=component.code, blocks=blocks)
        return self.finish()

    def finish(self) -> RetainedComponentValues:
        """Return the complete retention after logging its record.

        Raises:
            ExecutionPlanningError: A code was never retained.

        """
        self._retained.coverage.fail_if_incomplete()
        if self._logger.isEnabledFor(logging.DEBUG):
            record = self._retained.retention_record()
            self._logger.debug(
                "component retention record %s",
                record.to_json(),
                extra={"component_retention_record": record},
            )
        return self._retained


def _has_payload(store: object) -> bool:
    """Return whether a published artifact mapping holds any payload."""
    if isinstance(store, Mapping):
        return any(
            bool(inner) if isinstance(inner, Mapping) else True
            for inner in store.values()
        )
    return bool(len(store))  # ty: ignore[invalid-argument-type]


class SolvingComponentValues:
    """Solve each component as forward simulation reaches its code.

    `acquire` solves the code's component and hands its device values to the
    simulation; `release` retains them on the host once the code's subjects
    are simulated. Every code is acquired and released, including codes no
    subject holds, so the retained result is complete.
    """

    def __init__(self, *, schedule: ComponentSchedule) -> None:
        """Wrap a schedule whose components have not been solved."""
        self._schedule = schedule
        self._live: dict[
            int, MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]
        ] = {}

    @property
    def codes(self) -> tuple[int, ...]:
        """Return the codes in the order they are solved."""
        return tuple(component.code for component in self._schedule.components)

    @property
    def retained(self) -> RetainedComponentValues:
        """Return the host retention the solved components are copied into."""
        return self._schedule.retained

    @property
    def subject_codes(self) -> tuple[int, ...] | None:
        """Return the selected codes, or `None` when every code is solved."""
        return self._schedule.subject_codes

    def acquire(
        self, *, code: int
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]:
        """Solve the code's component and return its values on the device."""
        values = self._schedule.solve_component(code=code)
        self._live[code] = values
        return values

    def release(self, *, code: int) -> None:
        """Retain the code's values on the host and delete them on the device."""
        self._schedule.retain_component(code=code, blocks=self._live.pop(code))

    def abandon(self, *, code: int) -> None:
        """Delete the code's device values without retaining them."""
        live = self._live.pop(code, None)
        if live is not None:
            delete_blocks(blocks=live)

    def values(self) -> _ValueStoreBoundary:
        """Return the complete logical value store once every code is retained."""
        return self._schedule.finish().value_store()


class UploadedComponentValues:
    """Place each retained component back on the device for its subjects."""

    def __init__(
        self, *, retained: RetainedComponentValues, store: _ValueStoreBoundary
    ) -> None:
        """Wrap a complete retention and the value store the result exposes."""
        retained.coverage.fail_if_incomplete()
        self._retained = retained
        self._store = store
        self._live: dict[
            int, MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]
        ] = {}

    @property
    def codes(self) -> tuple[int, ...]:
        """Return the codes in grid order."""
        return self._retained.coverage.codes

    @property
    def subject_codes(self) -> None:
        """Every code's subjects are simulated from a complete retention."""
        return

    def acquire(
        self, *, code: int
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, jax.Array]]:
        """Upload the code's retained values onto their solve layout."""
        values = self._retained.upload(code=code)
        self._live[code] = values
        return values

    def release(self, *, code: int) -> None:
        """Delete the uploaded values; the host blocks stay retained."""
        self.abandon(code=code)

    def abandon(self, *, code: int) -> None:
        """Delete the uploaded values."""
        live = self._live.pop(code, None)
        if live is not None:
            delete_blocks(blocks=live)

    def values(self) -> _ValueStoreBoundary:
        """Return the value store of the result being simulated."""
        return self._store


class ComponentSelection:
    """The codes that the block-major schedules started in one scope solve.

    Each schedule started in the scope records its retention here, so the
    caller that opened the scope reads the selected codes' host blocks after a
    solve or a simulation that does not publish them.
    """

    def __init__(self, *, codes: tuple[int, ...]) -> None:
        """Select `codes`; no schedule has started yet."""
        self._codes = codes
        self._retained: list[RetainedComponentValues] = []

    @property
    def codes(self) -> tuple[int, ...]:
        """Return the selected codes."""
        return self._codes

    def record(self, *, retained: RetainedComponentValues) -> None:
        """Record the retention of a schedule started in the scope."""
        self._retained.append(retained)

    @property
    def retained(self) -> RetainedComponentValues:
        """Return the retention of the one schedule the scope started.

        Raises:
            ExecutionPlanningError: The scope started no schedule, or several.

        """
        if len(self._retained) != 1:
            msg = (
                f"A component selection expects one block-major schedule, but "
                f"{len(self._retained)} were started in its scope."
            )
            raise ExecutionPlanningError(msg)
        return self._retained[0]


_SELECTION: ContextVar[ComponentSelection | None] = ContextVar(
    "component_selection", default=None
)


@contextmanager
def selected_components(*, codes: tuple[int, ...]) -> Iterator[ComponentSelection]:
    """Scope the block-major schedules started inside to `codes`.

    A selected solve retains its codes and publishes no result; a selected
    simulation simulates the subjects of its codes alone. The active selection
    scope ends with the block; its returned retention can remain owned by the caller.
    """
    selection = ComponentSelection(codes=codes)
    token = _SELECTION.set(selection)
    try:
        yield selection
    finally:
        _SELECTION.reset(token)


def active_component_selection() -> ComponentSelection | None:
    """Return the selection of the enclosing `selected_components` scope, if any."""
    return _SELECTION.get()


def value_layouts(
    *,
    regimes: Mapping[RegimeName, Regime],
    state_name: StateName,
    topology: Mapping[RegimeName, object],
) -> MappingProxyType[_Coordinate, ValueLayout]:
    """Return the published layout of every value the solve publishes.

    A regime publishes a value in every period it is active, as the
    period-major loop does.

    Args:
        regimes: The canonical regimes.
        state_name: The blocked state.
        topology: Each regime's complete value shape and layout, from the
            canonical regimes.

    """
    axes = backward_induction._value_axis_names(regimes=regimes)  # noqa: SLF001
    return MappingProxyType(
        {
            (period, name): ValueLayout(
                shape=tuple(topology[name].shape),  # ty: ignore[unresolved-attribute]
                axis=axes[name].index(state_name),
                sharding=topology[name].sharding,  # ty: ignore[unresolved-attribute]
            )
            for name, regime in regimes.items()
            for period in sorted(regime.active_periods)
        }
    )
