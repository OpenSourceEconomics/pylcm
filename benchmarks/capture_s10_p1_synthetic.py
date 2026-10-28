"""Capture one SYNTHETIC-REPRESENTATIVE solve + simulate of the scalar DC-R shape.

The scalar DC-R workload is a public scalar DCEGM parent -> current-IID resources-only
DCEGM child -> action-free terminal. No production workload uses this producer,
so this is a scaled-up copy of the public P1 fixture
(`tests/egm/test_current_iid_call_owned_generation.py::_declarations/_params`),
labelled synthetic everywhere. It is not a production capture.

Economics are the fixture's: log utility, log terminal, parent landing
`2 s + 0.5 + 0.125 period + 0.0625 age`, child resources `wealth + eps`, identity
terminal law. The child's log/log conditional optimum is `(w + eps) / 2`; with
`--optima-on-grid` (default) every such optimum at a child grid point and a
quadrature node is inserted into the child saving grid.

The receipt holds per-phase solve seconds from the `lcm` logger's phase records,
simulate seconds, peak RSS, the `[mem]` compiler-memory lines, the projection
capacity and the retained projected continuation's row length / valid lengths,
plus provenance. The raw `lcm` log lines go next to it as `<out>.log`.

Usage:
    pixi run -e tests-cpu python benchmarks/capture_s10_p1_synthetic.py \\
        --parent-wealth-points 2 --child-wealth-points 2 --savings-points 22 \\
        --shock-nodes 2 --subjects 8 --out receipt.json
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
from dataclasses import fields
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Opt into the compiler-memory `[mem]` lines before pylcm reads the variable.
os.environ.setdefault("LCM_LOG_KERNEL_MEMORY", "1")

import jax
import numpy as np

from _lcm.typing import JSONValue
from benchmarks.warm_solve_phases import _collecting_lcm_records, parse_phase_records
from lcm import Model
from lcm.typing import UserParams

_MEM = re.compile(
    r"\[mem\] (?P<label>.+?): temp=(?P<temp>[0-9.]+) GiB\s+args=(?P<args>[0-9.]+) GiB"
    r"\s+output=(?P<output>[0-9.]+) GiB\s+peak=(?P<peak>[0-9.]+) GiB"
)

SHOCK_MU = 0.5
SHOCK_SIGMA = 0.2


def _build(
    args: argparse.Namespace,
) -> tuple[Model, UserParams, dict[str, JSONValue]]:
    from lcm import (
        ConsumptionSavingsRegime,
        IrregSpacedGrid,
        LiquidMargin,
        Model,
        Regime,
    )
    from lcm.shocks import Normal, ShockGroup, UserNodes
    from lcm.solvers import DCEGM, ExactEnvelope
    from tests.egm.test_current_iid_call_owned_generation import (
        _declarations,
        _params,
    )
    from tests.egm.test_current_iid_parent_demands import (
        _current_resources,
        _identity,
        _inverse,
        _landing,
        _resources,
        _savings,
        _terminal,
        _utility,
    )

    # Probabilists' Gauss-Hermite rule for Normal(mu, sigma).
    z, w = np.polynomial.hermite_e.hermegauss(args.shock_nodes)
    eps = SHOCK_MU + SHOCK_SIGMA * z
    weights = w / w.sum()

    child_wealth = np.linspace(1.0, 12.5, args.child_wealth_points)
    savings = np.geomspace(0.25, 12.75, args.savings_points)
    if args.optima_on_grid:
        savings = np.union1d(
            savings, ((child_wealth[:, None] + eps[None, :]) / 2).ravel()
        )
    savings = np.unique(np.round(savings, 12))

    margin = LiquidMargin(
        state="wealth",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    functions = {
        "utility": _utility,
        "resources": _resources,
        "savings": _savings,
        "inverse_marginal_utility": _inverse,
    }
    consumption = IrregSpacedGrid(points=np.array([0.25, 2.0]))
    parent = ConsumptionSavingsRegime(
        states={
            "wealth": IrregSpacedGrid(
                points=np.linspace(1.0, 4.0, args.parent_wealth_points)
            )
        },
        actions={"consumption": consumption},
        functions=functions,
        state_transitions={"wealth": {"child": _landing}},
        liquid=margin,
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(
                points=np.linspace(0.5, 4.0, max(4, args.savings_points // 4))
            ),
            envelope=ExactEnvelope(),
        ),
    )
    child = ConsumptionSavingsRegime(
        states={"wealth": IrregSpacedGrid(points=child_wealth)},
        actions={"consumption": consumption},
        functions={**functions, "resources": _current_resources},
        state_transitions={"wealth": {"done": _identity}},
        liquid=margin,
        solver=DCEGM(
            savings_grid=IrregSpacedGrid(points=savings),
            envelope=ExactEnvelope(),
            current_iid_projection_capacity=args.projection_capacity,
        ),
        shocks=("income",),
    )
    # Every other constructor field (ages, edges, regime ids, ...) is the fixture's.
    base = _declarations()
    inputs = {f.name: getattr(base, f.name) for f in fields(base) if f.init}
    inputs["regimes"] = {
        "parent": parent,
        "child": child,
        "done": Regime(
            states={"wealth": IrregSpacedGrid(points=savings)},
            functions={"utility": _terminal},
        ),
    }
    inputs["shocks"] = {
        "income": ShockGroup(
            shocks={"eps": Normal(mu=SHOCK_MU, sigma=SHOCK_SIGMA)},
            nodes=UserNodes(nodes=eps[:, None], weights=weights),
        )
    }
    sizes = {
        "parent_wealth_points": args.parent_wealth_points,
        "child_wealth_points": args.child_wealth_points,
        "savings_points_requested": args.savings_points,
        "child_savings_points_actual": int(savings.size),
        "parent_savings_points": max(4, args.savings_points // 4),
        "shock_nodes": args.shock_nodes,
        "subjects": args.subjects,
        "optima_on_grid": args.optima_on_grid,
        "shock_law": {"mu": SHOCK_MU, "sigma": SHOCK_SIGMA},
    }
    return Model(**inputs), _params(), sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent-wealth-points", type=int, default=2)
    parser.add_argument("--child-wealth-points", type=int, default=2)
    parser.add_argument("--savings-points", type=int, default=22)
    parser.add_argument("--shock-nodes", type=int, default=2)
    parser.add_argument("--subjects", type=int, default=8)
    parser.add_argument("--projection-capacity", type=int, default=4096)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument(
        "--optima-on-grid", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", args.precision == 64)

    import _lcm
    import lcm
    from _lcm.dtypes import canonical_float_dtype
    from _lcm.egm.projected_continuation import PROJECTED_EGM_CONTINUATION
    from lcm.solver_api import ArtifactRef

    repo = Path(__file__).resolve().parent.parent
    receipt: dict[str, JSONValue] = {
        "workload_kind": "synthetic-representative",
        "workload": ("Scalar DC-R (scaled public fixture; NOT production)"),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "hostname": socket.gethostname(),
        "argv": sys.argv,
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "precision": args.precision,
        "devices": [str(d) for d in jax.devices()],
        "cpu_count": os.cpu_count(),
        "lcm_file": lcm.__file__,
        "_lcm_file": _lcm.__file__,
        "projection_capacity": args.projection_capacity,
    }

    model, params, receipt["sizes"] = _build(args)
    rng = np.random.default_rng(args.seed)
    initial_conditions = {
        "regime_id": np.zeros(args.subjects, dtype=int),
        "age": np.zeros(args.subjects, dtype=int),
        "wealth": rng.uniform(2.5, 4.0, args.subjects).astype(canonical_float_dtype()),
    }

    with _collecting_lcm_records() as lines:
        start = time.perf_counter()
        result = model.solve(params=params, log_level="progress")
        jax.block_until_ready(result.retained_continuations)
        receipt["solve_wall_seconds"] = time.perf_counter() - start
        carry = result.retained_continuations[
            ArtifactRef(period=1, regime="child", key=PROJECTED_EGM_CONTINUATION)
        ]
        valid_length = np.asarray(carry.valid_length)
        receipt["retained_projection"] = {
            "carry_type": type(carry).__name__,
            "grid_shape": list(carry.grid.shape),
            "grid_dtype": str(carry.grid.dtype),
            "row_length": int(carry.grid.shape[-1]),
            "valid_length_max": int(valid_length.max()),
            "valid_length_min": int(valid_length.min()),
            "valid_all": bool(np.asarray(carry.valid).all()),
        }
        start = time.perf_counter()
        try:
            frame = model.simulate(
                params=params,
                solution=result,
                seed=args.seed,
                log_level="progress",
                initial_conditions=initial_conditions,
            )
            receipt["simulate_rows"] = len(frame.to_dataframe())
            receipt["simulate_wall_seconds"] = time.perf_counter() - start
        except Exception as exc:
            receipt["simulate_error"] = f"{type(exc).__name__}: {exc}"

    receipt["phases"] = [
        {"call_id": call.call_id, "phases": [vars(p) for p in call.phases]}
        for call in parse_phase_records(lines=lines)
    ]
    receipt["compiler_memory_gib"] = [
        m.groupdict() for line in lines if (m := _MEM.search(line))
    ]
    receipt["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".log").write_text("\n".join(lines) + "\n")
    args.out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    summary = (
        "solve_wall_seconds",
        "simulate_wall_seconds",
        "simulate_error",
        "peak_rss_mib",
        "retained_projection",
        "sizes",
    )
    print(json.dumps({k: receipt.get(k) for k in summary}, indent=2, default=str))
    if "simulate_error" in receipt:
        sys.exit(1)


if __name__ == "__main__":
    main()
