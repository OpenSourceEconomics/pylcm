"""Independent extensional oracle for local opaque-callback preservation.

There are no producer, checker, source parser or graph imports here. A use is
represented by its bound function token and complete ordered argument tuple.
Universal equality for an uninterpreted deterministic function follows from
equality of these applications; changing a tuple is deliberately not certified.
"""
from itertools import product


def unchanged_applications(original, proposed):
    """Sufficient extensional condition, independent of callback interpretation."""
    if set(original) != set(proposed):
        return False
    return all(original[key] == proposed[key] for key in original)


def literal_loop_payoff(x, history):
    """Algebraic scalar reference for the source fixture's two loop iterations."""
    return x + 3 * history


def check_oracle():
    tested = 0
    for x, history in product((-2, 0, 3), repeat=2):
        original = {"opaque": ("source-bound-pure-function", (x, history))}
        assert unchanged_applications(original, dict(original))
        changed = {"opaque": ("source-bound-pure-function", (x, history + 1))}
        assert not unchanged_applications(original, changed)
        assert literal_loop_payoff(x, history) != literal_loop_payoff(x, history + 1)
        tested += 1
    return tested


if __name__ == "__main__":
    print(f"PASS: {check_oracle()} extensional application pairs")
