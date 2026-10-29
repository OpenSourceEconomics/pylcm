"""Capture the S10 cost of one current-IID EGM workload against GridSearch.

Two SYNTHETIC-REPRESENTATIVE workloads, scaled copies of the public acceptance
fixtures (no production workload uses these producers yet):

- `dc-r`: scalar DCEGM parent -> current-IID resources-only DCEGM child ->
  action-free terminal (`tests/egm/test_current_iid_call_owned_generation.py`).
- `nnbegm-u`: finite NNBEGM keeper/adjuster child with a current shock in
  utility -> action-free terminal (`tests/egm/test_current_iid_negm_model.py`,
  with the NNBEGM solver of `tests/simulation/test_ordinary_current_iid_replay.py`).

`--solver gridsearch` keeps the same regimes, state grids, shock nodes, dtype and
subjects and swaps every EGM solver for `GridSearch()`. GridSearch searches the
declared consumption grids, so both arms declare the same `--consumption-points`
grids, and the GridSearch arm adds the borrowing limit the EGM savings grid
imposes as a `savings >= lowest saving` constraint. The `dc-r` parent's grid spans
the fixture's [0.25, 2.0]: its EGM replay chooses among declared grid actions,
and from initial wealth in [2.5, 4] each lands inside the child's domain.

Run one arm per process so the first call is cold. The receipt records cold and
warm solve, cold and warm physical replay (`simulate`), the phase records of
every call, peak RSS, device peak and live-array bytes, the `[mem]` compiler
lines, per-period means of the simulated panel for an ex-post comparison, and
provenance.

Usage:
    pixi run -e tests-cpu python benchmarks/capture_s10_cost.py \\
        --workload dc-r --solver egm --precision 64 --out receipt.json
"""

import argparse
import json
import os
import re
import resource
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Opt into the compiler-memory `[mem]` lines before pylcm reads the variable.
os.environ.setdefault("LCM_LOG_KERNEL_MEMORY", "1")

import jax
import numpy as np

from _lcm.typing import JSONValue
from benchmarks.warm_solve_phases import _collecting_lcm_records, parse_phase_records
from lcm import IrregSpacedGrid, Model, Regime
from lcm.typing import BoolND, FloatND, UserParams

type _Built = tuple[Model, UserParams, dict[str, np.ndarray], dict[str, JSONValue]]

_MEM = re.compile(
    r"\[mem\] (?P<label>.+?): temp=(?P<temp>[0-9.]+) GiB\s+args=(?P<args>[0-9.]+) GiB"
    r"\s+output=(?P<output>[0-9.]+) GiB\s+peak=(?P<peak>[0-9.]+) GiB"
)


def _gridsearch(
    *,
    regimes: dict[str, Regime],
    borrowing: float,
    **constraints: Callable[..., BoolND],
) -> dict[str, Regime]:
    """Swap EGM for GridSearch; `constraints` add what the EGM grids imply."""
    from lcm.solvers import GridSearch

    def feasible(*, savings: FloatND) -> BoolND:
        return savings >= borrowing

    return {
        name: (
            replace(
                regime,
                solver=GridSearch(),
                # GridSearch reads no inverse marginal utility.
                functions={
                    k: v
                    for k, v in regime.functions.items()
                    if k != "inverse_marginal_utility"
                },
                constraints={
                    **regime.constraints,
                    "feasible": feasible,
                    **constraints,
                },
            )
            if "consumption" in regime.actions
            else regime
        )
        for name, regime in regimes.items()
    }


def _dc_r(args: argparse.Namespace) -> _Built:
    from lcm import ConsumptionSavingsRegime
    from lcm.shocks import Normal, ShockGroup, UserNodes
    from lcm.solvers import DCEGM, ExactEnvelope, FUESEnvelope, RFCEnvelope
    from tests.egm.test_current_iid_call_owned_generation import (
        _declarations,
        _params,
    )

    envelope = {"exact": ExactEnvelope, "rfc": RFCEnvelope, "fues": FUESEnvelope}[
        args.envelope
    ]()
    z, w = np.polynomial.hermite_e.hermegauss(args.shock_nodes)
    eps, weights = 0.5 + 0.2 * z, w / w.sum()
    child_wealth = np.linspace(1.0, 12.5, args.wealth_points)
    savings = np.geomspace(0.25, 12.75, args.savings_points)
    consumption = IrregSpacedGrid(
        points=np.geomspace(0.05, 12.0, args.consumption_points)
    )
    parent_consumption = IrregSpacedGrid(
        points=np.geomspace(0.25, 2.0, args.consumption_points)
    )
    base = _declarations()
    parent = base.regimes["parent"]
    child = base.regimes["child"]
    assert isinstance(parent, ConsumptionSavingsRegime)
    assert isinstance(child, ConsumptionSavingsRegime)
    regimes = {
        "parent": replace(
            parent,
            states={
                "wealth": IrregSpacedGrid(
                    points=np.linspace(1.0, 4.0, args.wealth_points)
                )
            },
            actions={"consumption": parent_consumption},
            solver=DCEGM(
                savings_grid=IrregSpacedGrid(
                    points=np.linspace(0.5, 4.0, max(4, args.savings_points // 4))
                ),
                envelope=envelope,
            ),
        ),
        "child": replace(
            child,
            states={"wealth": IrregSpacedGrid(points=child_wealth)},
            actions={"consumption": consumption},
            solver=DCEGM(
                savings_grid=IrregSpacedGrid(points=savings),
                envelope=envelope,
                current_iid_projection_capacity=args.projection_capacity,
            ),
        ),
        "done": Regime(
            states={"wealth": IrregSpacedGrid(points=savings)},
            functions=base.regimes["done"].functions,
        ),
    }
    if args.solver == "gridsearch":
        regimes = _gridsearch(regimes=regimes, borrowing=float(savings[0]))
    inputs = {f.name: getattr(base, f.name) for f in fields(base) if f.init}
    inputs["regimes"] = regimes
    inputs["shocks"] = {
        "income": ShockGroup(
            shocks={"eps": Normal(mu=0.5, sigma=0.2)},
            nodes=UserNodes(nodes=eps[:, None], weights=weights),
        )
    }
    rng = np.random.default_rng(args.seed)
    initial = {
        "regime_id": np.zeros(args.subjects, dtype=int),
        "age": np.zeros(args.subjects, dtype=int),
        "wealth": rng.uniform(2.5, 4.0, args.subjects),
    }
    sizes = {"child_savings_points": int(savings.size), "shock_nodes_eps": list(eps)}
    return Model(**inputs), _params(), initial, sizes


def _nnbegm_u(args: argparse.Namespace) -> _Built:
    from lcm import NestedConsumptionSavingsRegime
    from lcm.shocks import ShockGroup, Uniform, UserNodes
    from lcm.solvers import NBEGM, NNBEGM, FiniteOuterGrid
    from tests.egm.test_current_iid_negm_model import _declarations, _params

    base = _declarations()
    child = base.regimes["child"]
    assert isinstance(child, NestedConsumptionSavingsRegime)
    savings = IrregSpacedGrid(points=np.geomspace(0.25, 6.0, args.savings_points))
    housing = IrregSpacedGrid(points=np.linspace(0.0, 2.0, args.housing_points))
    outer = IrregSpacedGrid(points=np.linspace(0.0, 2.0, args.outer_points))
    consumption = IrregSpacedGrid(
        points=np.geomspace(0.05, 5.0, args.consumption_points)
    )
    regimes = {
        "child": replace(
            child,
            states={
                "wealth": IrregSpacedGrid(
                    points=np.linspace(1.5, 3.0, args.wealth_points)
                ),
                "housing": housing,
            },
            actions={
                "consumption": consumption,
                "investment": IrregSpacedGrid(
                    points=np.linspace(-2.0, 2.0, 2 * args.outer_points - 1)
                ),
            },
            solver=NNBEGM(
                inner=NBEGM(savings_grid=savings, jump_read="bridged"),
                outer_search=FiniteOuterGrid(grid=outer),
            ),
        ),
        "done": replace(
            base.regimes["done"], states={"wealth": savings, "housing": housing}
        ),
    }
    if args.solver == "gridsearch":
        # NNBEGM targets only the finite outer grid inside [0, 2].
        def outer_domain(*, new_housing: FloatND) -> BoolND:
            return (new_housing >= 0.0) & (new_housing <= 2.0)

        regimes = _gridsearch(
            regimes=regimes, borrowing=0.25, outer_domain=outer_domain
        )
    eps = np.linspace(0.0, 1.0, args.shock_nodes)
    inputs = {f.name: getattr(base, f.name) for f in fields(base) if f.init}
    inputs["regimes"] = regimes
    inputs["shocks"] = {
        "income": ShockGroup(
            shocks={"eps": Uniform(start=0.0, stop=3.0)},
            nodes=UserNodes(
                nodes=eps[:, None],
                weights=np.full(args.shock_nodes, 1.0 / args.shock_nodes),
            ),
        )
    }
    model = Model(**inputs)
    rng = np.random.default_rng(args.seed)
    initial = {
        "regime_id": np.full(args.subjects, model.regime_names_to_ids["child"]),
        "age": np.ones(args.subjects, dtype=int),
        "wealth": rng.uniform(1.5, 3.0, args.subjects),
        "housing": rng.choice(np.asarray(housing.points), args.subjects),
    }
    return model, _params(), initial, {"shock_nodes_eps": list(eps)}


def _memory() -> dict[str, JSONValue]:
    stats = jax.devices()[0].memory_stats() or {}
    return {
        "live_array_bytes": int(sum(a.nbytes for a in jax.live_arrays())),
        "device_peak_bytes": stats.get("peak_bytes_in_use"),
        "device_in_use_bytes": stats.get("bytes_in_use"),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def _git(*args: str) -> str:
    repo = Path(__file__).resolve().parent.parent
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", choices=("dc-r", "nnbegm-u"), required=True)
    parser.add_argument("--solver", choices=("egm", "gridsearch"), default="egm")
    parser.add_argument("--envelope", choices=("exact", "rfc", "fues"), default="exact")
    parser.add_argument("--wealth-points", type=int, default=4)
    parser.add_argument("--housing-points", type=int, default=3)
    parser.add_argument("--outer-points", type=int, default=2)
    parser.add_argument("--savings-points", type=int, default=24)
    parser.add_argument("--consumption-points", type=int, default=24)
    parser.add_argument("--shock-nodes", type=int, default=2)
    parser.add_argument("--subjects", type=int, default=16)
    parser.add_argument("--projection-capacity", type=int, default=4096)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", args.precision == 64)

    import _lcm
    import lcm

    receipt: dict[str, JSONValue] = {
        "workload_kind": "synthetic-representative",
        "workload": args.workload,
        "solver": args.solver,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "args": {k: str(v) for k, v in vars(args).items()},
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "devices": [str(d) for d in jax.devices()],
        "cpu_count": os.cpu_count(),
        "lcm_file": lcm.__file__,
        "_lcm_file": _lcm.__file__,
    }
    build = _dc_r if args.workload == "dc-r" else _nnbegm_u
    start = time.perf_counter()
    model, params, initial, receipt["sizes"] = build(args)
    receipt["model_build_seconds"] = time.perf_counter() - start

    calls: dict[str, JSONValue] = {}
    with _collecting_lcm_records() as lines:
        result = None
        for label in ("solve_cold", "solve_warm"):
            del result
            start = time.perf_counter()
            result = model.solve(params=params, log_level="progress")
            jax.block_until_ready(result.retained_continuations)
            calls[label] = {"wall_seconds": time.perf_counter() - start, **_memory()}
        frame = None
        for label in ("replay_cold", "replay_warm"):
            start = time.perf_counter()
            frame = model.simulate(
                params=params,
                solution=result,
                seed=args.seed,
                log_level="progress",
                initial_conditions=initial,
            )
            data = frame.to_dataframe()
            calls[label] = {"wall_seconds": time.perf_counter() - start, **_memory()}
    receipt["calls"] = calls
    numeric = data.select_dtypes("number").drop(columns=["subject_id"], errors="ignore")
    receipt["panel_means_by_period"] = json.loads(
        numeric.groupby("period").mean().to_json(orient="index")
    )
    records = parse_phase_records(lines=lines)
    receipt["phases"] = [
        {"call_id": call.call_id, "phases": [vars(p) for p in call.phases]}
        for call in records
    ]
    # One public call per label, in call order: cold compilation is the
    # `compilation_waves` phase of `solve_cold`.
    for label, call in zip(calls, records, strict=True):
        calls[label]["phase_seconds"] = {
            p.name: p.seconds for p in call.phases if p.depth <= 1
        }
    receipt["compiler_memory_gib"] = [
        m.groupdict() for line in lines if (m := _MEM.search(line))
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".log").write_text("\n".join(lines) + "\n")
    args.out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps({"calls": calls, "sizes": receipt["sizes"]}, indent=2))


if __name__ == "__main__":
    main()
