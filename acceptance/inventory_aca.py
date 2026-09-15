"""Inventory the frozen ACA factory without solving or simulating.

Run with eight visible devices in a fresh fp32 process. The caller supplies
source imports through PYTHONPATH and the original sealed input packet.
Capacity output is a necessary lower bound, never an admission certificate.
"""

# Standalone acceptance probe: defer native imports to preserve failure receipts;
# inspect internal phase metadata and assert the literal frozen workload.
# Git subprocess arguments are fixed, apart from the inspected source directory.
# ruff: noqa: ANN401, INP001, PLC0415, PLR0915, PLR2004, S101, S603, S607, SLF001

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    """Construct the exact economic factory and write phase and capacity records."""
    if not __debug__:
        raise RuntimeError("Inventory assertions require Python without optimization.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "status": "unsupported-native",
        "scope": "factory construction and parameter/grid preparation only",
        "numerical_dispatch": False,
        "argv": sys.argv,
        "driver_sha256": _sha(Path(__file__)),
    }
    try:
        receipt.update(_construct(args.packet_root))
        _write(path=args.output_dir / "aca-phase-inventory.json", value=receipt)
        _write(path=args.output_dir / "aca-capacity.json", value=_capacity(receipt))
    except BaseException as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        _write(path=args.output_dir / "aca-phase-inventory.json", value=receipt)
        _write(
            path=args.output_dir / "aca-capacity.json",
            value={
                "status": "unmeasured",
                "admitted": False,
                "reason": "Native phase inventory did not complete.",
                "error": receipt["error"],
            },
        )
        raise


def _construct(packet_root: Path) -> dict[str, Any]:
    import aca_model
    import aca_slurm
    import jax
    import numpy as np
    import pandas as pd
    from aca_model.agent.preferences import PrefType
    from aca_model.baseline.model import create_model
    from aca_slurm._simulate import (
        _assemble,
        _derived_categoricals,
        _load_inputs,
        _model_params,
    )
    from aca_slurm._type_prediction import (
        replicate_for_draws,
        triple_initdist_by_pref_type,
    )
    from aca_slurm.config import (
        A40_GRID_CONFIG,
        N_DRAWS_PER_INDIVIDUAL,
        SIMULATION_SEED,
    )

    import lcm
    from _lcm.solution.v_topology import _get_regime_V_shapes_and_shardings
    from lcm import DiscreteGrid, ExecutionConfig

    assert os.environ.get("ACA_JAX_ENABLE_X64") == "0"
    assert not jax.config.jax_enable_x64
    assert len(jax.devices()) == 8
    manifest = json.loads((packet_root / "manifest.json").read_text())
    assert len(manifest["inputs"]) == 11
    inputs_identity = {
        name: _sha(packet_root / "inputs" / name) for name in manifest["inputs"]
    }
    assert inputs_identity == manifest["inputs"]
    roots = {
        "pylcm": Path(lcm.__file__).resolve().parents[2],
        "aca-model": Path(aca_model.__file__).resolve().parents[2],
        "aca-slurm": Path(aca_slurm.__file__).resolve().parents[2],
    }
    sources = {name: _source(root) for name, root in roots.items()}
    files = {
        "ss": "social_security_params",
        "tax": "tax_params",
        "ssi": "ssi_medicaid_params",
        "hi": "health_insurance_params",
        "pension": "pension_params",
        "wage": "wage_params",
        "transition": "transition_probs",
        "pref": "preference_start_values",
        "env": "environment_constants",
        "hcc_insurer": "hcc_insurer_params",
        "initdist": "initial_conditions",
    }
    inputs = _load_inputs(
        **{
            f"{key}_path": packet_root / "inputs" / f"{value}.pkl"
            for key, value in files.items()
        }
    )
    grid = A40_GRID_CONFIG
    policy = ExecutionConfig(
        devices=tuple(range(8)),
        sharded_states=("assets",),
        axis_widths={"subject": 2048},
    )
    fixed, params = _assemble(inputs=inputs, grid_config=grid)
    model = create_model(
        fixed_params=fixed,
        wage_params=inputs.wage,
        derived_categoricals=_derived_categoricals(),
        grid_config=grid,
        pref_type_grid=DiscreteGrid(PrefType),
        execution_config=policy,
        solver="brute_force",
        consumption_dollars_points=None,
    )
    params = _model_params(
        params=params, inputs=inputs, model=model, solver="brute_force"
    )
    flat_params = model._process_params(params)
    topology = _get_regime_V_shapes_and_shardings(
        regimes=model._regimes, flat_params=flat_params
    )
    initial = replicate_for_draws(
        triple_initdist_by_pref_type(inputs.initdist_df), n_draws=N_DRAWS_PER_INDIVIDUAL
    )
    assert len(inputs.initdist_df) == 9452
    assert len(initial) == 226848
    assert N_DRAWS_PER_INDIVIDUAL == 8
    assert SIMULATION_SEED == 20260903
    rows = []
    for name, regime in model._regimes.items():
        space = regime.solution.state_action_space(regime_params=flat_params[name])
        state_names = list(space.states)
        stored_names = [
            state for state in state_names if state not in regime.fold_state_names
        ]
        shape = list(topology[name].shape)
        assert shape == [len(space.states[state]) for state in stored_names]
        assert len(space.states["assets"]) == 24
        assert len(space.states["pref_type"]) == 3
        assert not regime.fold_state_names
        assert "pension_wealth" not in state_names
        if name == "dead":
            assert set(state_names) == {"assets", "pref_type"}
        else:
            assert len(space.states["aime"]) == 38
            assert "hcc_transitory" in state_names
            assert model.user_regimes[name].states["hcc_transitory"].fold is False
        index_map = topology[name].sharding.devices_indices_map(tuple(shape))
        assets_axis = stored_names.index("assets")
        indices = [index[assets_axis] for index in index_map.values()]
        assert len(indices) == 8
        assert sorted((index.start, index.stop) for index in indices) == [
            (i, i + 3) for i in range(0, 24, 3)
        ]
        classes: dict[str, dict[str, Any]] = {}
        for period in regime.active_periods:
            targets = {
                phase: list(graph.targets(period=period, source=name))
                if period < model.n_periods - 1
                else []
                for phase, graph in (
                    ("solution", model.reachability.solution),
                    ("simulation", model.reachability.simulation),
                )
            }
            key = repr(
                (
                    regime.solution.period_signatures.get(period),
                    regime.solution.solver_period_group_keys.get(period),
                    targets,
                )
            )
            entry = classes.setdefault(
                key, {"periods": [], "ages": [], "targets": targets}
            )
            entry["periods"].append(period)
            entry["ages"].append(51 + period)
        rows.append(
            {
                "regime": name,
                "terminal": regime.terminal,
                "declared_effective_grids": {
                    state: repr(spec)
                    for state, spec in model.user_regimes[name].states.items()
                },
                "solve_state_names": state_names,
                "stored_value_state_names": stored_names,
                "stored_value_shape": shape,
                "stored_value_dtype": "float32",
                "stored_value_bytes": math.prod(shape) * np.dtype(np.float32).itemsize,
                "simulation_state_names": list(regime.simulation.state_names),
                "carried_simulation_names": list(regime.simulation.carried_grids),
                "folded_names": list(regime.fold_state_names),
                "state_coordinates": {
                    state: _array(value) for state, value in space.states.items()
                },
                "action_names": list(regime.solution.action_names),
                "action_coordinates": {
                    action: _array(value) for action, value in space.actions.items()
                },
                "constraints": list(regime.solution.constraints),
                "active_period_classes": list(classes.values()),
                "value_sharding": str(topology[name].sharding),
                "per_device_indices": {
                    str(device): [str(index) for index in indices]
                    for device, indices in index_map.items()
                },
            }
        )
    schemas = {
        row["regime"]: {
            "state_names": row["stored_value_state_names"],
            "shape": row["stored_value_shape"],
        }
        for row in rows
    }
    for row in rows:
        for period_class in row["active_period_classes"]:
            period_class["target_value_schemas"] = {
                target: schemas[target]
                for target in {
                    target
                    for targets in period_class["targets"].values()
                    for target in targets
                }
            }
    return {
        "status": "native-construction-verified",
        "sources": sources,
        "inputs_sha256": inputs_identity,
        "original_manifest_sha256": _sha(packet_root / "manifest.json"),
        "imports": {
            "lcm": lcm.__file__,
            "aca_model": aca_model.__file__,
            "aca_slurm": aca_slurm.__file__,
        },
        "jax_version": jax.__version__,
        "python": sys.version,
        "devices": [
            {"id": device.id, "platform": device.platform, "kind": device.device_kind}
            for device in jax.devices()
        ],
        "grid_config": dataclasses.asdict(grid),
        "execution_config": {
            field.name: repr(getattr(policy, field.name))
            for field in dataclasses.fields(policy)
        },
        "hardware_acceptance": False,
        "policy_note": (
            "Unbudgeted construction inventory; "
            "no allocator ceiling or admission claim."
        ),
        "workload": manifest["workload"],
        "initial_population": len(initial),
        "initial_hash": hashlib.sha256(
            pd.util.hash_pandas_object(initial, index=True).to_numpy().tobytes()
        ).hexdigest(),
        "regimes": rows,
    }


def _capacity(receipt: dict[str, Any]) -> dict[str, Any]:
    rows = receipt["regimes"]
    retained = sum(
        row["stored_value_bytes"]
        * sum(len(group["periods"]) for group in row["active_period_classes"])
        for row in rows
    )
    largest = max(row["stored_value_bytes"] for row in rows)
    single_artifact_source_destination_scratch = largest * 17 // 8
    lower_bound = max(retained // 8, single_artifact_source_destination_scratch)
    return {
        "schema_version": 2,
        "status": "necessary-lower-bound-only",
        "admitted": False,
        "phase_inventory_driver_sha256": receipt["driver_sha256"],
        "source_python_tree_sha256": {
            name: source["python_tree_sha256"]
            for name, source in receipt["sources"].items()
        },
        "inputs_sha256": receipt["inputs_sha256"],
        "dtype": "float32",
        "n_assets_shards": 8,
        "retained_horizon_value_bytes_global": retained,
        "retained_horizon_value_bytes_per_device": retained // 8,
        "largest_single_full_value_replica_bytes": largest,
        "necessary_per_device_lower_bound_bytes": lower_bound,
        "formula": (
            "max(retained horizon values / 8, one-artifact 17S/8); "
            "overlaps are not added"
        ),
        "P_compiler_peak_bytes": None,
        "H_retained_bank_per_device_bytes": retained // 8,
        "H_all_retained_device_owners_peak_bytes": None,
        "host_owned_inputs_bytes": None,
        "host_completed_chunks_bytes": None,
        "host_assembly_bytes": None,
        "R_single_full_replica_bytes": largest,
        "T_single_replica_scratch_bytes": largest,
        "one_artifact_source_destination_scratch_lower_bound": {
            "artifact_global_bytes_S": largest,
            "source_shard_bytes_S_over_8": largest // 8,
            "destination_full_replica_bytes_S": largest,
            "scratch_bytes_S": largest,
            "per_device_bytes_17S_over_8": single_artifact_source_destination_scratch,
            "scope": (
                "One retained eight-way sharded artifact and its full "
                "destination/scratch; not a global concurrent total"
            ),
        },
        "R_peak_simultaneous_replicas_bytes": None,
        "T_transfers_bytes": None,
        "replay_artifacts_bytes": None,
        "pending_outputs_bytes": None,
        "external_owner_bytes": None,
        "compiler_peak_bytes": None,
        "selected_allocator_limit_bytes": None,
        "lower_bound_screen_passed": None,
        "limitations": [
            "No numerical solve or simulation",
            "No GPU allocator measurement",
            "No replay or compiler profiling",
            "No complete P/H/R/T admission accounting",
            (
                "P is compiler storage; H is retained device owners; "
                "host memory is separate"
            ),
            "Single-replica R/T terms are not peak concurrent phase requirements",
        ],
    }


def _array(value: Any) -> dict[str, Any]:
    import numpy as np

    host = np.asarray(value)
    return {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "points": host.tolist(),
        "sha256": hashlib.sha256(host.tobytes()).hexdigest(),
    }


def _source(root: Path) -> dict[str, Any]:
    files = {
        str(path.relative_to(root)): _sha(path)
        for path in sorted((root / "src").rglob("*.py"))
    }
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "root": str(root),
        "git_head": result.stdout.strip() if result.returncode == 0 else None,
        "git_status": status.stdout if status.returncode == 0 else None,
        "python_files_sha256": files,
        "environment_files_sha256": {
            name: _sha(root / name)
            for name in ("pyproject.toml", "pixi.lock")
            if (root / name).is_file()
        },
        "python_tree_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(*, path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
