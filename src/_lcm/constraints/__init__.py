"""Everything a regime's declared constraints become before a solver sees them.

`processed.py` normalizes each declaration into one form, `bounds.py` and
`ir.py` recognise what a condition says, `routes.py` assigns one terminal
disposition per constraint per solver route, and `materialize.py` turns the
result into a function the DAG can compose.
"""
