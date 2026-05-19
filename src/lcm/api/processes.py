"""User-facing stochastic process leaf classes.

Each class declares a **state grid AND** its transition mechanism. Unlike
ordinary grids (`LinSpacedGrid` etc.), which are pure outcome-space, a
process class bundles both the discretization nodes and the dynamics on
those nodes.

Place an instance of one of these in `Regime(states={...})` — *not* in
`state_transitions`. pylcm treats the discretization nodes as the realized
values of the state at each grid point; the transition mechanism is
invoked automatically.

The classes are aliases of the internal `lcm._processes` leaf classes
during the transitional period — the leaf classes themselves still go by
their pre-Phase-2b names (`Uniform`, `Tauchen`, ...) in their definition
files. A subsequent rename will replace those internal names with the
canonical `*Process` ones; this module then becomes a plain re-export.

Naming convention: `<Distribution><Kind>Process`.

- `*IIDProcess`: independent draws at each period.
- `*AR1Process`: AR(1) process with a chosen discretization scheme.
"""

from lcm._processes.ar1 import (
    Rouwenhorst as RouwenhorstAR1Process,
)
from lcm._processes.ar1 import (
    Tauchen as TauchenAR1Process,
)
from lcm._processes.ar1 import (
    TauchenNormalMixture as TauchenNormalMixtureAR1Process,
)
from lcm._processes.iid import (
    LogNormal as LogNormalIIDProcess,
)
from lcm._processes.iid import (
    Normal as NormalIIDProcess,
)
from lcm._processes.iid import (
    NormalMixture as NormalMixtureIIDProcess,
)
from lcm._processes.iid import (
    Uniform as UniformIIDProcess,
)

__all__ = [
    "LogNormalIIDProcess",
    "NormalIIDProcess",
    "NormalMixtureIIDProcess",
    "RouwenhorstAR1Process",
    "TauchenAR1Process",
    "TauchenNormalMixtureAR1Process",
    "UniformIIDProcess",
]
