"""An exhaustive scalar feasibility oracle independent of pylcm's feasibility code.

For every subject the oracle enumerates the full cartesian product of the regime's
declared action-grid points and calls the user's constraint functions one scalar
combination at a time. No vmap, no jit and no engine helper is involved, so the
oracle can arbitrate the vectorized kernel.
"""

import inspect
import itertools
from collections.abc import Callable, Mapping
from typing import cast

import numpy as np
import pandas as pd

from lcm import DiscreteGrid, LinSpacedGrid, Model


def exhaustive_scalar_feasibility(
    *,
    model: Model,
    initial_conditions: pd.DataFrame,
    params: Mapping[str, object],
) -> np.ndarray:
    """Return the per-subject feasibility mask by brute-force scalar enumeration.

    Args:
        model: The unsolved model whose user regimes declare the constraints.
        initial_conditions: One row per subject with `regime_name`, `age` and the
            regime's states in user vocabulary (categorical labels allowed).
        params: User parameters as passed to `Model.simulate`.

    Returns:
        Boolean array with one entry per row of `initial_conditions`.

    """
    verdicts = []
    for _, row in initial_conditions.iterrows():
        regime_name = str(row["regime_name"])
        regime = model.user_regimes[regime_name]
        subject = _subject_kwargs(model=model, regime_name=regime_name, row=row)
        action_grids = {
            name: _grid_points(grid) for name, grid in regime.actions.items()
        }
        combos: list[dict[str, object]] = [
            dict(zip(action_grids, values, strict=True))
            for values in itertools.product(*action_grids.values())
        ]
        verdicts.append(
            any(
                all(
                    _call_scalar(
                        func=cast("Callable[..., object]", constraint),
                        available={
                            **subject,
                            **combo,
                            **_params_for(
                                params=params,
                                fixed_params=model.fixed_params,
                                regime_name=regime_name,
                                func_name=name,
                            ),
                        },
                    )
                    for name, constraint in regime.constraints.items()
                )
                for combo in combos
            )
        )
    return np.asarray(verdicts, dtype=bool)


def _grid_points(grid: object) -> list[object]:
    if isinstance(grid, LinSpacedGrid):
        return list(np.linspace(grid.start, grid.stop, grid.n_points).tolist())
    if isinstance(grid, DiscreteGrid):
        return list(grid.codes)
    msg = f"The oracle does not enumerate grids of type {type(grid).__name__}."
    raise NotImplementedError(msg)


def _subject_kwargs(
    *, model: Model, regime_name: str, row: pd.Series
) -> dict[str, object]:
    regime = model.user_regimes[regime_name]
    kwargs: dict[str, object] = {}
    for name, grid in regime.states.items():
        value = row[name]
        if isinstance(grid, DiscreteGrid) and isinstance(value, str):
            value = grid.codes[grid.categories.index(value)]
        kwargs[name] = value
    age = float(row["age"])
    kwargs["age"] = age
    kwargs["period"] = model.ages.age_to_period(age)
    return kwargs


def _params_for(
    *,
    params: Mapping[str, object],
    fixed_params: Mapping[str, object],
    regime_name: str,
    func_name: str,
) -> dict[str, object]:
    """Collect scalar parameters visible to one constraint; runtime binds over fixed."""
    found: dict[str, object] = {}
    for source in (fixed_params, params):
        regime_tree = source.get(regime_name, {})
        func_tree = (
            regime_tree.get(func_name, {}) if isinstance(regime_tree, Mapping) else {}
        )
        for tree in (source, regime_tree, func_tree, source.get(func_name, {})):
            if isinstance(tree, Mapping):
                found.update(
                    {k: v for k, v in tree.items() if not isinstance(v, Mapping)}
                )
    return found


def _call_scalar(
    *, func: Callable[..., object], available: Mapping[str, object]
) -> bool:
    names = inspect.signature(func).parameters
    return bool(func(**{name: available[name] for name in names}))
