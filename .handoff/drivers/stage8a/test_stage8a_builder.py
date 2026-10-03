"""The production builder receives its execution policy before model creation."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("mode", ["explicit", "default"])
def test_production_builder_forwards_execution_policy_without_changing_inputs(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    """An explicit policy reaches the owner builder; omission retains its default."""
    default = SimpleNamespace(policy="owner_default")
    explicit = SimpleNamespace(
        invariant_block_widths={"pref_type": 1},
        invariant_block_schedule="block_major",
        device_memory_bytes=None,
    )
    grid = SimpleNamespace(points="full_canonical_A100")
    raw = object()
    tripled = object()
    replicated = object()
    model = object()
    params = {"literal_parameter": 17}
    calls: dict[str, list[Any]] = {
        "make_execution": [],
        "build": [],
        "load": [],
        "triple": [],
        "replicate": [],
    }

    def make_execution_config(**kwargs: Any) -> object:
        calls["make_execution"].append(kwargs)
        return default

    def build_aca_policy_model(**kwargs: Any) -> tuple[object, dict[str, int]]:
        calls["build"].append(kwargs)
        return model, params

    def load_inputs(**kwargs: Any) -> object:
        calls["load"].append(kwargs)
        return SimpleNamespace(initdist_df=raw)

    def triple_initial(frame: object) -> object:
        calls["triple"].append(frame)
        return tripled

    # keyword-only-exempt: library-callback=replicate_for_draws
    def replicate_initial(frame: object, *, n_draws: int) -> object:
        calls["replicate"].append((frame, n_draws))
        return replicated

    modules = {
        name: ModuleType(name)
        for name in (
            "aca_model",
            "aca_model.aca",
            "aca_model.aca.health_insurance",
            "aca_slurm",
            "aca_slurm._simulate",
            "aca_slurm._type_prediction",
            "aca_slurm.config",
        )
    }
    for name, module in modules.items():
        if name in {"aca_model", "aca_model.aca", "aca_slurm"}:
            module.__dict__["__path__"] = []
        monkeypatch.setitem(sys.modules, name, module)
    modules["aca_model.aca.health_insurance"].__dict__["PolicyVariant"] = (
        SimpleNamespace(ACA="ACA")
    )
    simulate = modules["aca_slurm._simulate"]
    simulate.__dict__["_production_input_paths"] = lambda: {
        "literal_input": "unchanged"
    }
    simulate.__dict__["_load_inputs"] = load_inputs
    simulate.__dict__["build_aca_policy_model"] = build_aca_policy_model
    prediction = modules["aca_slurm._type_prediction"]
    prediction.__dict__["triple_initdist_by_pref_type"] = triple_initial
    prediction.__dict__["replicate_for_draws"] = replicate_initial
    config = modules["aca_slurm.config"]
    config.__dict__["_GRID_CONFIG_BY_GPU"] = {"nvidia_a100_sxm4_80gb": grid}
    config.__dict__["N_DRAWS_PER_INDIVIDUAL"] = 4
    config.__dict__["make_execution_config"] = make_execution_config
    path = Path(__file__).parents[1] / "stage5b/stage3_arms.py"
    spec = importlib.util.spec_from_file_location("stage8a_builder_harness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The inherited production builder cannot be loaded")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    arguments = {} if mode == "default" else {"production_execution_config": explicit}
    monkeypatch.setattr(sys, "path", list(sys.path))

    built_model, built_params, initial, description = harness._build(
        workload="production", aca_slurm_src=tmp_path, n_subjects=1, **arguments
    )

    assert (
        built_model is model,
        built_params is params,
        initial is replicated,
        calls["build"][0]["execution_config"]
        is (explicit if mode == "explicit" else default),
        description["pref_types"],
        calls,
    ) == (
        True,
        True,
        True,
        True,
        3,
        {
            "make_execution": []
            if mode == "explicit"
            else [{"solver": "brute_force", "continuous_sharding": True}],
            "build": [
                {
                    "policy": "ACA",
                    "grid_config": grid,
                    "solver": "brute_force",
                    "execution_config": explicit if mode == "explicit" else default,
                }
            ],
            "load": [{"literal_input": "unchanged"}],
            "triple": [raw],
            "replicate": [(tripled, 4)],
        },
    )
