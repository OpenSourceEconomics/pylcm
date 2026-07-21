"""User-facing stochastic process leaf classes.

Each class declares a **state grid AND** its transition mechanism. Unlike
ordinary grids (`LinSpacedGrid` etc.), which are pure outcome-space, a
process class bundles both the discretization nodes and the dynamics on
those nodes.

Place an instance of one of these in `Regime(states={...})` — *not* in
`state_transitions`. pylcm treats the discretization nodes as the realized
values of the state at each grid point; the transition mechanism is
invoked automatically.

Naming convention: `<Distribution><Kind>Process`.

- `*IIDProcess`: independent draws at each period.
- `*AR1Process`: AR(1) process with a chosen discretization scheme.
"""

from _lcm.processes.ar1 import (
    RouwenhorstAR1Process,
    TauchenAR1Process,
    TauchenNormalMixtureAR1Process,
)
from _lcm.processes.base import StateConditioned
from _lcm.processes.iid import (
    LogNormalIIDProcess,
    NormalIIDProcess,
    NormalMixtureIIDProcess,
    UniformIIDProcess,
)

__all__ = [
    "LogNormalIIDProcess",
    "NormalIIDProcess",
    "NormalMixtureIIDProcess",
    "RouwenhorstAR1Process",
    "StateConditioned",
    "TauchenAR1Process",
    "TauchenNormalMixtureAR1Process",
    "UniformIIDProcess",
]
