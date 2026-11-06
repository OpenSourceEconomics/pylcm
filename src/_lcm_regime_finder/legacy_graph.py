"""Lift legacy age-only activity/transition declarations into explicit graphs.

This compatibility adapter evaluates each age predicate on the complete finite
time axis. It never probes an economic state/action or a probability law.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from typing import Any


def install_regime_adapter(lcm: Any) -> type:
    original = lcm.Regime

    @dataclass(frozen=True, kw_only=True, init=False)
    class LegacyRegime(original):
        transition: Any = None
        active: Any = None

        def __init__(self, *, transition: Any = None, active: Any = None, **kwargs: Any) -> None:
            object.__setattr__(self, "transition", transition)
            object.__setattr__(self, "active", active)
            original.__init__(self, **kwargs)

    lcm.Regime = LegacyRegime
    return LegacyRegime


def lift_model_kwargs(lcm: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "edges" in kwargs:
        return kwargs, {}
    regimes = kwargs["regimes"]
    if not any(hasattr(regime, "active") or hasattr(regime, "transition") for regime in regimes.values()):
        return kwargs, {}
    ages = kwargs["ages"]
    axis = list(ages.exact_values)
    active: dict[str, list[int]] = {}
    for name, regime in regimes.items():
        predicate = getattr(regime, "active", None)
        if predicate is None:
            active[name] = list(range(len(axis)))
            continue
        signature = inspect.signature(predicate)
        if {name.lstrip("_") for name in signature.parameters} - {"age", "period"}:
            raise ValueError("Legacy activity is not an age/period-only declaration: " + name)
        admitted = []
        for period, age in enumerate(axis):
            args = {key: float(age) if key.lstrip("_") == "age" else period for key in signature.parameters}
            value = predicate(**args)
            if getattr(value, "shape", ()) != ():
                raise ValueError("Legacy age activity must be scalar")
            if bool(value):
                admitted.append(period)
        active[name] = admitted
    edges = {}
    for name, regime in regimes.items():
        law = getattr(regime, "transition", None)
        if law is None:
            continue
        if not isinstance(law, Mapping):
            raise ValueError("Legacy graph lift currently requires explicit per-target laws")
        cases = {}
        for period in active[name]:
            if period + 1 >= len(axis):
                continue
            allowed = {target: law[target] for target in law if period + 1 in active[target]}
            if allowed:
                cases[float(axis[period])] = allowed
        if cases:
            edges[name] = lcm.Transition(law=lcm.ByAge(cases=cases))
    initial = tuple((axis[p], name) for name in regimes for p in active[name])
    return {**kwargs, "edges": edges, "initial_nodes": initial}, {
        "activity_support": active,
        "activity_evaluation": "Every declared finite age evaluated; no state/action or probability sampling",
        "initial_admission": "Every legacy-active age/regime start retained, matching the unconstrained legacy start interface",
        "law_values": "Original probability and state laws unchanged; only declaration location moved",
    }
