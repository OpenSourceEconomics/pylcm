"""Capture the S10 cost of one current-IID EGM workload against GridSearch.

Two SYNTHETIC-REPRESENTATIVE workloads, scaled copies of the public acceptance
fixtures (no production workload uses these producers yet):

- `dc-aca`: the ACA-shaped DCEGM consumer of
  `tests/solution/test_current_iid_consumer_parity.py` (`shape="aca"`): a carried
  health state, a work choice, a literal savings floor and a declared health-cost
  shock, with `alive` repeating until a terminal `done`. The shock is uniform with
  unit variance rather than the fixture's normal: simulated draws from a normal tail
  push the cost of a poor, unhealthy subject past every action's savings floor.
- `nnbegm`: the finite NNBEGM keeper/adjuster housing regime of
  `tests/egm/_nnbegm_toy.py`, whose resources read a declared current shock.

`--solver nbegm` (dc-aca only) solves the `alive` regime with `NBEGM` on the same
savings grid instead of `DCEGM`.

`--solver gridsearch` keeps the same regimes, state grids, shock nodes, dtype and
subjects and swaps every EGM solver for `GridSearch()`, which searches the declared
consumption (and investment) grids. Both arms therefore declare the same
`--consumption-points` grids. For `nnbegm` the GridSearch arm adds the borrowing limit
the EGM savings grid imposes and the outer-grid domain as constraints; `dc-aca`
already declares its floor.

Run one arm per process so the first call is cold. The receipt records cold and
warm solve, cold and warm `simulate`, the phase records of every call, peak RSS,
device peak and live-array bytes, per-period means of the simulated panel for an
ex-post comparison, and provenance.

Usage:
    pixi run -e tests-cpu python benchmarks/capture_s10_cost.py \\
        --workload dc-aca --solver egm --precision 64 --out receipt.json
"""

import argparse
import json
import os
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

import jax
import numpy as np

from _lcm.typing import JSONValue
from benchmarks.warm_solve_phases import _collecting_lcm_records, parse_phase_records
from lcm import IrregSpacedGrid, LinSpacedGrid, Model, Regime
from lcm.solvers import GridSearch
from lcm.typing import BoolND, FloatND, UserParams

type _Built = tuple[Model, UserParams, dict[str, np.ndarray], dict[str, JSONValue]]


def _gridsearch(regime: Regime, **constraints: Callable[..., BoolND]) -> Regime:
    """Swap EGM for GridSearch; `constraints` add what the EGM grids imply."""
    return replace(
        regime,
        solver=GridSearch(),
        # GridSearch reads no inverse marginal utility.
        functions={
            k: v for k, v in regime.functions.items() if k != "inverse_marginal_utility"
        },
        constraints={**regime.constraints, **constraints},
    )


def _dc_aca(args: argparse.Namespace) -> _Built:
    from lcm import (
        ConsumptionSavingsRegime,
        DiscreteGrid,
        InitialNodes,
        Periods,
        fixed_transition,
        post_decision_lower_bound,
    )
    from lcm.shocks import EqualSpacedUniform, Product, ShockGroup, Uniform
    from lcm.solvers import DCEGM, NBEGM
    from tests.solution.test_current_iid_consumer_parity import (
        _MARGIN,
        _SAVINGS_FLOOR,
        _CostRegimeId,
        _Health,
        _health_bequest,
        _health_costs,
        _inverse_marginal_utility,
        _next_wealth_from_savings,
        _savings,
        _Work,
        _work_utility,
        _working_resources,
    )

    # `_aca_shaped_model(shape="aca")` with size knobs and `alive` repeating. Wealth
    # and savings start at the floor, so every landing is on the wealth grid;
    # consumption reaches the largest resources.
    top = args.wealth_top
    savings = IrregSpacedGrid(
        points=np.linspace(_SAVINGS_FLOOR, 1.2 * top, args.savings_points)
    )
    health = DiscreteGrid(category_class=_Health)
    alive = ConsumptionSavingsRegime(
        states={
            "wealth": LinSpacedGrid(
                start=_SAVINGS_FLOOR, stop=top, n_points=args.wealth_points
            ),
            "good_health": health,
        },
        actions={
            "consumption": LinSpacedGrid(
                start=0.05, stop=top + 2.0, n_points=args.consumption_points
            ),
            "work": DiscreteGrid(category_class=_Work),
        },
        functions={
            "utility": _work_utility,
            "total_costs": _health_costs,
            "resources": _working_resources,
            "savings": _savings,
            "inverse_marginal_utility": _inverse_marginal_utility,
        },
        state_transitions={
            "wealth": dict.fromkeys(("alive", "done"), _next_wealth_from_savings),
            "good_health": dict.fromkeys(
                ("alive", "done"), fixed_transition("good_health")
            ),
        },
        constraints={
            "borrowing_limit": post_decision_lower_bound(
                margin=_MARGIN, lower=_SAVINGS_FLOOR
            )
        },
        liquid=_MARGIN,
        solver=(NBEGM if args.solver == "nbegm" else DCEGM)(savings_grid=savings),
        shocks=("hcc",),
    )
    if args.solver == "gridsearch":
        alive = _gridsearch(alive)
    last = args.periods - 2
    model = Model(
        regimes={
            "alive": alive,
            "done": Regime(
                states={
                    "wealth": savings,
                    "good_health": health,
                },
                functions={"utility": _health_bequest},
            ),
        },
        n_periods=args.periods,
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        edges={
            "alive": {
                "alive": Periods(values=tuple(range(last))),
                "done": Periods(values=(last,)),
            }
        },
        regime_id_class=_CostRegimeId,
        fixed_params={"discount_factor": 0.95},
        shocks={
            "hcc": ShockGroup(
                shocks={
                    "hcc_transitory": Uniform(start=-np.sqrt(3.0), stop=np.sqrt(3.0))
                },
                nodes=Product(hcc_transitory=EqualSpacedUniform(n=args.shock_nodes)),
            )
        },
    )
    rng = np.random.default_rng(args.seed)
    initial = {
        "regime_id": np.zeros(args.subjects, dtype=np.int32),
        "period": np.zeros(args.subjects, dtype=np.int32),
        "wealth": rng.uniform(1.0, top / 2, args.subjects),
        "good_health": rng.integers(0, 2, args.subjects, dtype=np.int32),
    }
    return model, {}, initial, {"savings_points": args.savings_points}


def _nnbegm(args: argparse.Namespace) -> _Built:
    from lcm import NestedConsumptionSavingsRegime
    from lcm.shocks import ShockGroup, Uniform, UserNodes
    from lcm.solvers import NBEGM, NNBEGM, FiniteOuterGrid
    from tests.egm._nnbegm_toy import _declarations, _params

    base = _declarations()
    child = base.regimes["child"]
    assert isinstance(child, NestedConsumptionSavingsRegime)
    savings = IrregSpacedGrid(points=np.geomspace(0.25, 6.0, args.savings_points))
    housing = IrregSpacedGrid(points=np.linspace(0.0, 2.0, args.housing_points))
    outer_points = np.linspace(0.0, 2.0, args.outer_points)
    outer = IrregSpacedGrid(points=outer_points)
    child = replace(
        child,
        states={
            "wealth": IrregSpacedGrid(points=np.linspace(1.5, 3.0, args.wealth_points)),
            "housing": housing,
        },
        actions={
            "consumption": IrregSpacedGrid(
                points=np.geomspace(0.05, 5.0, args.consumption_points)
            ),
            # Mirrored rather than `linspace(-2, 2)`, whose middle point misses 0 by
            # rounding, so keeping the house would be off the GridSearch arm's grid.
            "investment": IrregSpacedGrid(
                points=np.concatenate([-outer_points[:0:-1], outer_points])
            ),
        },
        solver=NNBEGM(
            inner=NBEGM(savings_grid=savings, jump_read="bridged"),
            outer_search=FiniteOuterGrid(grid=outer),
        ),
    )
    if args.solver == "gridsearch":

        def feasible(*, savings: FloatND) -> BoolND:
            return savings >= 0.25

        # NNBEGM targets only the finite outer grid inside [0, 2].
        def outer_domain(*, new_housing: FloatND) -> BoolND:
            return (new_housing >= 0.0) & (new_housing <= 2.0)

        child = _gridsearch(child, feasible=feasible, outer_domain=outer_domain)
    eps = np.linspace(0.0, 1.0, args.shock_nodes)
    inputs = {f.name: getattr(base, f.name) for f in fields(base) if f.init}
    inputs["regimes"] = {
        "child": child,
        "done": replace(
            base.regimes["done"], states={"wealth": savings, "housing": housing}
        ),
    }
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
    parser.add_argument("--workload", choices=("dc-aca", "nnbegm"), required=True)
    parser.add_argument(
        "--solver", choices=("egm", "nbegm", "gridsearch"), default="egm"
    )
    parser.add_argument("--wealth-points", type=int, default=5)
    parser.add_argument("--wealth-top", type=float, default=5.0)
    parser.add_argument("--housing-points", type=int, default=3)
    parser.add_argument("--outer-points", type=int, default=2)
    parser.add_argument("--savings-points", type=int, default=12)
    parser.add_argument("--consumption-points", type=int, default=12)
    parser.add_argument("--shock-nodes", type=int, default=3)
    parser.add_argument("--periods", type=int, default=3)
    parser.add_argument("--subjects", type=int, default=30)
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.solver == "nbegm" and args.workload != "dc-aca":
        parser.error("--solver nbegm applies to the dc-aca workload only")

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
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "devices": [str(d) for d in jax.devices()],
        "cpu_count": os.cpu_count(),
        "lcm_file": lcm.__file__,
        "_lcm_file": _lcm.__file__,
    }
    build = _dc_aca if args.workload == "dc-aca" else _nnbegm
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
            jax.block_until_ready(result.values)
            calls[label] = {"wall_seconds": time.perf_counter() - start, **_memory()}
        for label in ("simulate_cold", "simulate_warm"):
            start = time.perf_counter()
            data = model.simulate(
                params=params,
                solution=result,
                seed=args.seed,
                log_level="progress",
                initial_conditions=initial,
            ).to_dataframe()
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".log").write_text("\n".join(lines) + "\n")
    args.out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps({"calls": calls, "sizes": receipt["sizes"]}, indent=2))


if __name__ == "__main__":
    main()
