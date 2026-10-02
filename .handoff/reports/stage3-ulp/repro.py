"""Solve the life-cycle model blocked and unblocked; list differing cells with ULP."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import jax
import numpy as np

import lcm
import lc_model

print("lcm:", lcm.__file__, "x64:", jax.config.jax_enable_x64, "jax", jax.__version__)


def ulp_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Number of representable values between a and b (same-sign finite)."""
    ia = a.view(np.int64 if a.dtype == np.float64 else np.int32).astype(np.int64)
    ib = b.view(np.int64 if b.dtype == np.float64 else np.int32).astype(np.int64)
    return np.abs(ia - ib)


def solve(typed_dead: bool, blocked: bool, extra=None):
    m = lc_model.model(typed_dead=typed_dead, blocked=blocked, extra=extra)
    v = m.solve(params=lc_model.params(typed_dead=typed_dead), log_level="off").values
    return {p: {r: np.asarray(a) for r, a in by.items()} for p, by in v.items()}


def compare(got, ref, label):
    total = 0
    for p in ref:
        for r in ref[p]:
            a, b = got[p][r], ref[p][r]
            assert a.shape == b.shape and a.dtype == b.dtype, (p, r, a.shape, b.shape)
            assert np.isfinite(b).any()
            diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            n = int(diff.sum())
            total += n
            if n:
                fin = diff & np.isfinite(a) & np.isfinite(b)
                gaps = ulp_gap(a, b)
                print(f"  [{label}] period {p} {r} shape {b.shape}: {n}/{b.size} differ, max ULP {gaps[fin].max()}, max abs {np.abs(a-b)[fin].max():.3e}")
                for idx in zip(*np.nonzero(diff)):
                    print(f"     idx {tuple(int(i) for i in idx)} blocked {a[idx]!r} unblocked {b[idx]!r} ulp {gaps[idx]}")
            else:
                print(f"  [{label}] period {p} {r}: bitwise equal ({b.size} cells, dtype {b.dtype})")
    return total


if __name__ == "__main__":
    extra = eval(os.environ.get("EXTRA", "None"))
    grand = 0
    for typed_dead in (True, False):
        print(f"typed_dead={typed_dead} extra={extra}")
        grand += compare(solve(typed_dead, True, extra), solve(typed_dead, False, extra), f"td={typed_dead}")
    print("TOTAL differing cells:", grand)
