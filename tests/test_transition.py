"""Tests for the user-facing transition wrappers."""

import lcm
from lcm import LinSpacedGrid
from lcm.transition import SolveSimulateStatePair


def test_solve_simulate_state_pair_carries_solve_function_and_simulate_state() -> None:
    """A `SolveSimulateStatePair` holds a solve-phase function and a simulate-phase
    state (grid + transition).

    In the solve phase the name is a derived function (not a grid dimension); in
    the simulate phase it is a seeded, evolved state.
    """

    def imputed(aime: float) -> float:
        return aime * 0.1

    def evolve(pension_wealth: float) -> float:
        return pension_wealth * 1.03

    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    pair = SolveSimulateStatePair(solve=imputed, grid=grid, transition=evolve)

    assert (pair.solve, pair.grid, pair.transition) == (imputed, grid, evolve)


def test_solve_simulate_state_pair_is_part_of_public_api() -> None:
    """`SolveSimulateStatePair` is importable from the top-level `lcm` package."""
    assert lcm.SolveSimulateStatePair is SolveSimulateStatePair
