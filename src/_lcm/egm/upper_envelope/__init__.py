"""Upper-envelope refinement of EGM candidate solutions.

Discrete choices make the EGM candidate value correspondence multi-valued in
non-concave regions; the algorithms here select the upper envelope and drop
dominated candidates. Currently implemented: the Fast Upper-Envelope Scan
(`_lcm.egm.upper_envelope.fues`).
"""
