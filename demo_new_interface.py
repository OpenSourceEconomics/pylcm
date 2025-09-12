#!/usr/bin/env python3
"""Demo of the new Model interface."""

import jax.numpy as jnp
from pybaum import tree_map

from tests.test_models import get_model_config, get_params


def main():
    print("🎯 PyLCM Model Refactor Demo")
    print("=" * 50)
    
    # Create a model - all functions are pre-compiled during initialization!
    print("\n1. Creating model (functions compile during initialization)...")
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    
    # Model now has pre-computed functions
    print(f"✓ Model created with {len(model.state_action_spaces)} periods")
    print(f"✓ Pre-compiled {len(model.max_Q_over_a_functions)} optimization functions")
    
    # Get parameters from template  
    params = tree_map(lambda _: 0.2, model.params_template)
    print(f"✓ Using parameters: {list(params.keys())}")
    
    # New clean interface - solve directly
    print("\n2. Solving model using model.solve()...")
    solution = model.solve(params)
    print(f"✓ Solution computed for periods: {list(solution.keys())}")
    
    # Setup simulation
    initial_states = {
        "wealth": jnp.array([10.0, 20.0]),
    }
    print(f"✓ Initial states: {len(next(iter(initial_states.values())))} agents")
    
    # New clean interface - solve and simulate in one call
    print("\n3. Solving and simulating using model.solve_and_simulate()...")
    results = model.solve_and_simulate(
        params=params,
        initial_states=initial_states,
    )
    print(f"✓ Simulation results: {results.shape[0]} rows, {results.shape[1]} columns")
    print(f"✓ Columns: {list(results.columns)}")
    
    print("\n🎉 Demo complete! The new interface is much cleaner than get_lcm_function()!")
    
    # Show comparison
    print("\n📊 Old vs New Interface:")
    print("OLD: solve_model, params_template = get_lcm_function(model, targets='solve')")
    print("     solution = solve_model(params)")
    print()
    print("NEW: solution = model.solve(params)")
    print("     # Functions pre-compiled, parameters available as model.params_template")


if __name__ == "__main__":
    main()