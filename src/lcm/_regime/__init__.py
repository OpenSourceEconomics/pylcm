"""Internal helpers backing `lcm.api.regime`.

The user-facing `Regime`, `MarkovTransition`, and `SolveSimulateFunctionPair`
live in `lcm.api.regime`. Validation, the default aggregator, and the
`MarkovTransition` probability-array helpers live here, behind a leading
underscore, so the public surface stays a thin layer of class definitions
plus deprecated public functions.

"""
