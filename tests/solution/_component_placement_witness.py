"""Compare collected component-job raw results with the single-process reference.

Run as `python -m tests.solution._component_placement_witness` in a fresh process
with eight CPU host devices. It prints one JSON line mapping each case to the leaves
whose placement differs from the reference and to any error raised, after checking
values, raw bytes and the public panel exactly.
"""

import json
import tempfile
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    InvariantBlockSchedule,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.component_jobs import (
    collect_component_jobs,
    plan_component_jobs,
    run_component_job,
)
from lcm.result import SimulationResult
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)

# Each case names the selected devices, chunk width, population codes and job
# layout. They cover a non-default selected CPU, one chunk, the width boundary
# and its tail, several codes whose total fits one width, empty codes, permuted
# rows and job completion, one versus several jobs, eager execution, the default
# device and an eight-device subject layout.
CASES: tuple[dict[str, object], ...] = (
    {"devices": (1,), "width": 1, "codes": (2, 0, 2)},
    {"devices": (1,), "width": 1, "codes": (0,)},
    {"devices": (1,), "width": 2, "codes": (2, 2)},
    {"devices": (1,), "width": 2, "codes": (2, 2, 2)},
    {"devices": (1,), "width": 8, "codes": (2, 0)},
    {"devices": (1,), "width": 2, "codes": (2, 0, 2, 0), "reverse_workers": True},
    {"devices": (1,), "width": 2, "codes": (0, 2, 0, 2), "assignment": ((2, 0, 1),)},
    {"devices": (1,), "width": 2, "codes": (2, 0, 2), "enable_jit": False},
    {"devices": (0,), "width": 1, "codes": (2, 0, 2)},
    {
        "devices": tuple(range(8)),
        "width": 8,
        "codes": (2,) * 8,
        "simulation_sharding": "subjects",
    },
)


@categorical(ordered=True)
class _PreferenceCode:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


@categorical(ordered=True)
class _Choice:
    zero: ScalarInt
    one: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    live: ScalarInt
    dead: ScalarInt


def _flow(
    *, wealth: ContinuousState, pref_type: DiscreteState, choice: DiscreteAction
) -> FloatND:
    return wealth + pref_type + choice


def _terminal(*, wealth: ContinuousState, pref_type: DiscreteState) -> FloatND:
    return wealth + pref_type


def _model(
    *,
    devices: tuple[int, ...],
    width: int,
    enable_jit: bool,
    simulation_sharding: Literal["legacy", "subjects"],
) -> Model:
    states = {
        "wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
        "pref_type": DiscreteGrid(category_class=_PreferenceCode),
    }
    live = Regime(
        states=states,
        state_transitions={
            "wealth": fixed_transition(state_name="wealth"),
            "pref_type": fixed_transition(state_name="pref_type"),
        },
        actions={"choice": DiscreteGrid(category_class=_Choice)},
        functions={"utility": _flow},
    )
    dead = Regime(states=states, functions={"utility": _terminal})
    return Model(
        regimes={"live": live, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        edges={"live": {"dead": (0,)}},
        initial_nodes={0: "live"},
        execution_config=ExecutionConfig(
            devices=devices,
            device_memory_bytes=None,
            axis_widths={"subject": width},
            invariant_block_widths={"pref_type": 1},
            invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR,
            simulation_sharding=simulation_sharding,
        ),
        enable_jit=enable_jit,
    )


def _initial_population(*, codes: tuple[int, ...]) -> dict[str, jax.Array]:
    n_subjects = len(codes)
    return {
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(n_subjects, _RegimeId.live, dtype=jnp.int32),
        "wealth": jnp.linspace(1.0, 2.0, n_subjects),
        "pref_type": jnp.asarray(codes, dtype=jnp.int32),
    }


def _bytes(array: jax.Array) -> tuple[str, tuple[int, ...], bytes]:
    host = np.asarray(jax.device_get(array))
    return host.dtype.str, host.shape, host.tobytes()


def _layout(array: jax.Array) -> dict[str, object]:
    sharding = array.sharding
    mesh = (
        None
        if not isinstance(sharding, jax.sharding.NamedSharding)
        else {
            "shape": list(sharding.mesh.devices.shape),
            "axis_names": list(sharding.mesh.axis_names),
            "devices": [device.id for device in sharding.mesh.devices.flat],
        }
    )
    return {
        "devices": sorted(device.id for device in array.devices()),
        "memory_kind": sharding.memory_kind,
        "mesh": mesh,
    }


def _placement_mismatches(
    *, reference: SimulationResult, collected: SimulationResult
) -> list[str]:
    """Check values, raw bytes and panels exactly; return misplaced leaf paths."""
    reference_values = reference.solution.values  # ty: ignore[unresolved-attribute]
    collected_values = collected.solution.values  # ty: ignore[unresolved-attribute]
    for period, regimes in reference_values.items():
        for name, value in regimes.items():
            if _bytes(value) != _bytes(collected_values[period][name]):
                msg = f"Value bytes differ at {(period, name)}"
                raise AssertionError(msg)
    expected, structure = jax.tree.flatten_with_path(reference.raw_results)
    actual, other_structure = jax.tree.flatten_with_path(collected.raw_results)
    if structure != other_structure:
        msg = "Raw result trees differ"
        raise AssertionError(msg)
    mismatches = []
    for (path, left), (_, right) in zip(expected, actual, strict=True):
        if _bytes(left) != _bytes(right):
            msg = f"Raw bytes differ at {jax.tree_util.keystr(path)}"
            raise AssertionError(msg)
        if not (
            right.sharding.is_equivalent_to(left.sharding, left.ndim)
            and _layout(left) == _layout(right)
        ):
            mismatches.append(jax.tree_util.keystr(path))
    pd.testing.assert_frame_equal(
        reference.to_dataframe(), collected.to_dataframe(), check_exact=True
    )
    return mismatches


def _run_case(
    *,
    directory: Path,
    devices: tuple[int, ...],
    width: int,
    codes: tuple[int, ...],
    enable_jit: bool = True,
    simulation_sharding: Literal["legacy", "subjects"] = "legacy",
    assignment: tuple[tuple[int, ...], ...] = ((0,), (1,), (2,)),
    reverse_workers: bool = False,
) -> list[str]:
    params = {"discount_factor": 0.9}
    initial = _initial_population(codes=codes)

    def build() -> Model:
        return _model(
            devices=devices,
            width=width,
            enable_jit=enable_jit,
            simulation_sharding=simulation_sharding,
        )

    reference = build().simulate(
        params=params, initial_conditions=initial, seed=7, log_level="off"
    )
    model = build()
    plan = plan_component_jobs(
        model=model,
        params=params,
        directory=directory,
        assignment=assignment,
        initial_conditions=initial,
        seed=7,
    )
    jobs = list(range(len(plan.jobs)))
    if reverse_workers:
        jobs.reverse()
    for job in jobs:
        run_component_job(
            model=model,
            params=params,
            directory=directory,
            job=job,
            initial_conditions=initial,
            log_level="off",
        )
    collected = collect_component_jobs(
        model=model, params=params, directory=directory, log_level="off"
    )
    return _placement_mismatches(
        reference=reference, collected=cast("SimulationResult", collected.simulation)
    )


def main() -> None:
    """Run every case and print the misplaced leaves and errors of each."""
    report: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory() as work:
        for index, case in enumerate(CASES):
            try:
                mismatches: list[str] = _run_case(
                    directory=Path(work) / f"case-{index}",
                    **case,  # ty: ignore[invalid-argument-type]
                )
                error = None
            except Exception as exception:  # noqa: BLE001
                mismatches = []
                error = f"{type(exception).__name__}: {exception}"
            report[str(index)] = {"mismatches": mismatches, "error": error}
    print(json.dumps(report, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
