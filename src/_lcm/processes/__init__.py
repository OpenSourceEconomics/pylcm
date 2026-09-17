"""Stochastic-process infrastructure behind the classes `lcm.processes` exposes.

A process bundles a discretization grid with its transition mechanism:
`base.py` holds the shared base and quadrature helpers, `iid.py` and `ar1.py`
the leaf classes, and `grid_resolution.py` and `state_conditioned.py` the
parameter-dependent support and conditioning routes.
"""

from _lcm.processes.ar1 import _AR1Process
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.processes.iid import _IIDProcess

__all__ = [
    "_AR1Process",
    "_ContinuousStochasticProcess",
    "_IIDProcess",
]
