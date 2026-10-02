import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from tests.solution import test_block_major_lifetime as t
from tests.simulation import test_type_grouped_simulation as lc
params = lc._params(typed_dead=True)
for codes, name in ((lc._UNBALANCED, "unbalanced"), ((1,)*7, "one_type"), ((0,1,2)*3, "balanced")):
    initial = lc._initial(codes=codes)
    frames = {}
    for label, sched in (("bm", t._BLOCK_MAJOR), ("pm", t._PERIOD_MAJOR), ("ub", None)):
        m = t._life_cycle_model(schedule=sched)
        frames[label] = m.simulate(params=params, initial_conditions=initial, solution=m.solve(params=params, log_level="off"), seed=7, log_level="off").to_dataframe()
    def same(a, b):
        return all(frames[a][c].to_numpy().tobytes() == frames[b][c].to_numpy().tobytes() for c in frames[a].columns if frames[a][c].dtype.kind == "f") and frames[a].equals(frames[b])
    print(name, "bm==pm", same("bm","pm"), "pm==ub", same("pm","ub"), "bm==ub", same("bm","ub"))
