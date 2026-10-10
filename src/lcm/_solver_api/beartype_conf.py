"""The `BeartypeConf` for exact validators at the solver-API boundary.

Solver plugins and the result spine call these validators with values pylcm did
not build. A type violation there raises `SolverAPITypeError`, a `TypeError`, so
the callers that already treat a wrongly typed key or payload as a `TypeError`
keep doing so. The solver API imports nothing from `_lcm`, so this conf lives
beside it rather than with the engine's confs.
"""

from beartype import BeartypeConf

from lcm.exceptions import SolverAPITypeError

# beartype's default strategy checks one sampled entry of each mapping or
# sequence per call; `is_pep484_tower=True` lets an `int` satisfy `float`.
SOLVER_API_CONF = BeartypeConf(
    is_pep484_tower=True,
    violation_door_type=SolverAPITypeError,
    violation_param_type=SolverAPITypeError,
    violation_return_type=SolverAPITypeError,
)
