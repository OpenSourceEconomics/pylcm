"""User-facing stochastic process leaf classes.

Each class declares a **state grid AND** its transition mechanism. Unlike
ordinary grids (`LinSpacedGrid` etc.), which are pure outcome-space, a
process class bundles both the discretization nodes and the dynamics on
those nodes.

Place an instance of one of these in `Regime(states={...})` — *not* in
`state_transitions`. pylcm treats the discretization nodes as the realized
values of the state at each grid point; the transition mechanism is
invoked automatically.

The classes are aliases of the internal `lcm.shocks` leaf classes. The
implementations live there for now; the deeper rename
(`lcm/shocks/` → `lcm/_processes/`, `_ShockGrid` → `_ProcessGrid`,
`is_shock` → `is_process`, `shock_names` → `process_names`) is deferred
to a follow-up. New code should use the names exposed here.

Naming convention: `<Distribution><Kind>Process`.

- `*IIDProcess`: independent draws at each period.
- `*AR1Process`: AR(1) process with a chosen discretization scheme.
"""

from lcm.shocks.ar1 import (
    Rouwenhorst as RouwenhorstAR1Process,
)
from lcm.shocks.ar1 import (
    Tauchen as TauchenAR1Process,
)
from lcm.shocks.ar1 import (
    TauchenNormalMixture as TauchenNormalMixtureAR1Process,
)
from lcm.shocks.iid import (
    LogNormal as LogNormalIIDProcess,
)
from lcm.shocks.iid import (
    Normal as NormalIIDProcess,
)
from lcm.shocks.iid import (
    NormalMixture as NormalMixtureIIDProcess,
)
from lcm.shocks.iid import (
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
