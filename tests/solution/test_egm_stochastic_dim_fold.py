"""Stochastic expectations are applied only after row interpolation."""

import _lcm.egm.continuation as cont_mod
from tests.test_models import nbegm_stochastic_node_toy as toy


def test_stochastic_nodes_remain_until_after_interpolation(monkeypatch):
    """Every stochastic row is interpolated before its expectation is taken."""
    calls = []
    original = cont_mod._expect_over_stochastic_nodes

    def spy(**kwargs):
        calls.append(
            (
                kwargs["read"].stochastic_state_names,
                kwargs["carry"].breakpoints is not None,
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(cont_mod, "_expect_over_stochastic_nodes", spy)
    toy.build_model(variant="nbegm").solve(params=toy.build_params(), log_level="debug")
    smooth_calls = [names for names, has_jumps in calls if not has_jumps]
    jumped_calls = [names for names, has_jumps in calls if has_jumps]
    assert smooth_calls
    assert all("income" in names for names in smooth_calls)
    assert all("income" in names for names in jumped_calls)
