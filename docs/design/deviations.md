# Deviations from Issue #296 Plan

Changes made during implementation that differ from the proposal in the issue comment.

## 1. Circular import: lazy imports in params leaf modules

**Problem:** Moving `utils.py` to `utils/containers.py` created a circular import:
`grids/__init__` → `categorical` → `containers` → `params` → `params/processing` →
`interfaces` → `grids`.

**Fix:** `params/mapping_leaf.py` and `params/sequence_leaf.py` use lazy (in-function)
imports for `ensure_containers_are_immutable` and `_make_immutable` from
`utils/containers.py`, with `# noqa: PLC0415` to suppress the ruff warning.

**Alternative considered:** Moving the import the other direction (having containers not
import from params). However, `_make_immutable` recursively handles `MappingLeaf` and
`SequenceLeaf` instances, so it genuinely needs to know about them. The lazy import in
the leaf modules (which are rarely instantiated directly) is the least disruptive
solution.

## 2. `create_state_space_info` uses lazy import for `variable_info`

**Problem:** Moving `_create_state_space_info` from `regime_processing.py` to `V.py`
required importing `get_variable_info` and `get_grids` from `variable_info.py`. This
would create a circular dependency since `variable_info` also imports grid types used by
`V.py`.

**Fix:** `create_state_space_info` in `V.py` uses a lazy import for `get_variable_info`
and `get_grids` with `# noqa: PLC0415`.

## 3. Type annotation weakened, then restored

The agent initially changed `create_state_space_info(regime: Regime)` to
`create_state_space_info(regime: object)` to avoid importing `Regime` in `V.py`. This
was caught by `ty` and fixed by adding `from lcm.regime import Regime` — no circular
dependency existed for this import path.

## 4. Ruff per-file-ignores updated

The old config had `per-file-ignores."src/lcm/functools.py"` for ANN401 and
`per-file-ignores."src/lcm/*Q_*.py"` for N999. Updated to:

- `per-file-ignores."src/lcm/utils/functools.py"` for ANN401
- `per-file-ignores."src/lcm/**/*[A-Z]*.py"` for N999 (covers `Q_and_F.py`, `V.py`,
  `max_Q_over_a.py`)

## 5. `process_params` re-export in `params/__init__.py` uses `__getattr__`

**Problem:** Adding `from lcm.params.processing import process_params` eagerly to
`params/__init__.py` creates a circular import: `typing` → `params` →
`params/processing` → `utils/namespace` → `typing`.

**Fix:** `params/__init__.py` uses a module-level `__getattr__` to lazily import
`process_params` on first access. This keeps `from lcm.params import process_params`
working without triggering the cycle at import time.

## 6. Lazy import in `next_state.py` for `_get_vmap_params` is required

The code review flagged `from lcm.regime_building.max_Q_over_a import _get_vmap_params`
as an unnecessary lazy import. Moving it to top-level caused the same `typing` circular
import chain. The lazy import stays.

## 7. `lcm_examples` import updated

`src/lcm_examples/mahler_yum_2024/_model.py` imported from `lcm.dispatchers` — updated
to `lcm.utils.dispatchers`. This file was not mentioned in the plan but is part of the
repo.

## No deviations from the target structure

The final file layout matches the issue comment exactly. All moves, splits, and merges
were executed as proposed.
