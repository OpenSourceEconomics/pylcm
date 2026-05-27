"""V_arr chunk-spec derivation for memory-bounded save/load.

The engine scans over states with `batch_size > 0` during solve+simulate,
keeping only one slice of the splayed axis live on device at a time. Save
mirrors that pattern: it materialises the V_arr one chunk at a time along
the same axis, so the peak save-time device buffer never exceeds the
compute-time peak. `_ChunkSpec` records which axis to chunk and at what
width; `_build_chunk_specs` derives one per regime from the canonical
state-action space ordering.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from _lcm.engine import Regime
from _lcm.typing import FlatParams, RegimeName, StateName


@dataclass(frozen=True)
class _ChunkSpec:
    """Axis and width for chunked V_arr save / load."""

    chunk_axis: int | None
    """Index of the V_arr axis to chunk over; `None` if no state was batched."""

    chunk_size: int
    """Slice width along `chunk_axis`. Zero (and unused) when `chunk_axis is None`."""


def _chunk_spec_from_state_order(
    *,
    state_names: tuple[StateName, ...],
    batch_sizes: Mapping[StateName, int],
) -> _ChunkSpec:
    """Pick the outermost V_arr axis with `batch_size > 0`.

    Iteration order in `state_names` matches the V_arr's axis order;
    the first state with `batch_size > 0` is the axis along which the
    engine scanned, so chunking the save along that axis bounds the
    materialisation to one scan iteration's worth.

    Args:
        state_names: V_arr axis order (one state per axis).
        batch_sizes: Mapping of state name to engine `batch_size`.

    Returns:
        `_ChunkSpec` with `chunk_axis` set to the first positively-batched
        axis (and `chunk_size` to that state's batch_size), or
        `(None, 0)` when every state has `batch_size == 0`.

    """
    for axis, name in enumerate(state_names):
        size = batch_sizes.get(name, 0)
        if size > 0:
            return _ChunkSpec(chunk_axis=axis, chunk_size=size)
    return _ChunkSpec(chunk_axis=None, chunk_size=0)


def _build_chunk_specs(
    *,
    regimes: Mapping[RegimeName, Regime],
    flat_params: FlatParams,
) -> MappingProxyType[RegimeName, _ChunkSpec]:
    """Derive one `_ChunkSpec` per regime from its V_arr's state axis order."""
    specs: dict[RegimeName, _ChunkSpec] = {}
    for regime_name, regime in regimes.items():
        state_action_space = regime.state_action_space(
            regime_params=flat_params[regime_name],
        )
        state_names = tuple(state_action_space.states.keys())
        batch_sizes = {
            name: grid.batch_size
            for name, grid in regime.grids.items()
            if name in state_action_space.states
        }
        specs[regime_name] = _chunk_spec_from_state_order(
            state_names=state_names,
            batch_sizes=batch_sizes,
        )
    return MappingProxyType(specs)
