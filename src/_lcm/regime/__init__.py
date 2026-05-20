"""Internal helpers backing `lcm.regime`.

The user-facing `Regime`, `MarkovTransition`, and `SolveSimulateFunctionPair`
live in `lcm.regime`. Validation and the default aggregator live here,
behind a leading underscore, so the public surface stays a thin layer of
class definitions.

"""
