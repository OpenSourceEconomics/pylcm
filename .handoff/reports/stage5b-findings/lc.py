import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from tests.solution import test_block_major_lifetime as t
bm, params = t._workload(name="life_cycle", schedule=t._BLOCK_MAJOR)
pm, _ = t._workload(name="life_cycle", schedule=t._PERIOD_MAJOR)
ub, _ = t._workload(name="life_cycle", schedule=None)
vb = bm.solve(params=params, log_level="off").values
vp = pm.solve(params=params, log_level="off").values
vu = ub.solve(params=params, log_level="off").values
for p in sorted(vu):
    for r in vu[p]:
        a, b, c = (np.asarray(v[p][r]) for v in (vb, vp, vu))
        print(p, r, a.shape, "bm==pm", a.tobytes()==b.tobytes(), "pm==ub", b.tobytes()==c.tobytes(), "maxdiff bm-pm", np.max(np.abs(a-b)))
print({p: tuple(vp[p]) for p in vp})
print({p: tuple(vu[p]) for p in vu})
print({p: tuple(vb[p]) for p in vb})
for p in sorted(vu):
    for r in vu[p]:
        b, c = (np.asarray(v[p][r]) for v in (vp, vu))
        d = b != c
        if d.any():
            ulps = np.abs(b.view(np.int64) - c.view(np.int64))
            print(p, r, "n diff", int(d.sum()), "of", d.size, "max ulps", int(ulps.max()), "max abs", float(np.max(np.abs(b-c))))
