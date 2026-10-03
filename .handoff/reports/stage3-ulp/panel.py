"""Simulate blocked-solve vs unblocked-solve panels; report per-column differences."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import jax
import numpy as np
import pandas as pd
import lc_model
from repro import ulp_gap
print("x64", jax.config.jax_enable_x64)
n = 300
rng = np.random.default_rng(seed=3)
initial = {
    "wealth": rng.uniform(1.0, 10.0, n),
    "pref_type": np.arange(n, dtype=np.int32) % 3,
    "health": (np.arange(n, dtype=np.int32) // 3) % 2,
    "age": np.zeros(n),
    "regime_id": np.zeros(n, dtype=np.int32),
}
assert np.isfinite(initial["wealth"]).all()
for td in (True, False):
    frames = {}
    for blocked in (False, True):
        m = lc_model.model(typed_dead=td, blocked=blocked)
        p = lc_model.params(typed_dead=td)
        sol = m.solve(params=p, log_level="off")
        frames[blocked] = m.simulate(params=p, initial_conditions=initial, solution=sol, seed=7, log_level="off").to_dataframe()
    a, b = frames[True], frames[False]
    assert a.shape == b.shape and list(a.columns) == list(b.columns) and a.index.equals(b.index), (a.shape, b.shape)
    print(f"typed_dead={td} rows={len(a)} columns={list(a.columns)}")
    for col in a.columns:
        x, y = a[col].to_numpy(), b[col].to_numpy()
        if pd.api.types.is_float_dtype(a[col]):
            same = (x == y) | (np.isnan(x) & np.isnan(y))
            nd = int((~same).sum())
            fin = ~same & np.isfinite(x) & np.isfinite(y)
            mx = int(ulp_gap(x[fin], y[fin]).max()) if fin.any() else 0
            print(f"  {col}: float, {nd} differing rows, max ULP {mx}")
        else:
            xa, ya = a[col].astype(object), b[col].astype(object)
            both_missing = xa.isna() & ya.isna()
            nd = int((~((xa == ya) | both_missing)).sum())
            print(f"  {col}: {a[col].dtype}, {nd} differing rows ({int(both_missing.sum())} rows missing in both)")
