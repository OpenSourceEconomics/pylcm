"""Forward simulation: the sampling loop, and the planning that admits it.

`simulate.py` is the loop itself; `programs.py` and `runtime.py` declare and
dispatch its per-regime work; and the chunk, residency and memory modules admit
that work against the resolved device-memory budget.
"""
