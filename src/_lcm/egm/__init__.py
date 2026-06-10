"""Endogenous-grid-method (DC-EGM) solver internals.

Hosts the DC-EGM model-contract validation (`_lcm.egm.validation`), the
upper-envelope refinement of EGM candidate solutions
(`_lcm.egm.upper_envelope`), and interpolation on the NaN-padded grids the
refinement produces (`_lcm.egm.interp`).
"""
