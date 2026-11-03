"""Bind the exact interval oracle to the live ordinary winner reduction."""

import ast
from typing import Literal


def _function(*, tree: ast.Module, name: str) -> ast.FunctionDef:
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(definitions) == 1, f"ordinary route needs one live {name}"
    return definitions[0]


def _requires(*, tree: ast.AST, fragment: str) -> None:
    node = ast.parse(fragment).body[0]
    expected = node.value if isinstance(node, ast.Expr) else node
    signature = ast.dump(expected)
    assert any(ast.dump(node) == signature for node in ast.walk(tree)), (
        f"ordinary route lost its stable-index operation: {fragment}"
    )


def ordinary_identity(tree: ast.Module) -> Literal["stable"]:
    """Require live global-index comparisons before choosing the oracle's identity.

    A missing or renamed source seam fails here; it never becomes a simulated
    position-based production algorithm. Both the batched rank reader and the
    streaming block reducer must preserve the same global identity and payload.
    """
    ranks = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_OrdinaryRank"
    ]
    assert len(ranks) == 1, "ordinary route needs one live _OrdinaryRank"
    fields = [
        node.target.id
        for node in ranks[0].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert fields == [
        "value",
        "right_available",
        "slope_high",
        "slope_low",
        "stable_index",
    ], "ordinary rank must end with the global stable index"

    obligations = {
        "_ordinary_rank_argmax": (
            "rank[:-1]",
            (
                "earliest = jnp.min(jnp.where(tied, rank.stable_index, "
                "jnp.iinfo(jnp.int32).max), axis=1, keepdims=True)"
            ),
            "tied = tied & (rank.stable_index == earliest)",
        ),
        "_ordinary_block_best": (
            "offered[:-1]",
            "jnp.broadcast_to(block_stable_index[None, :], terms.brackets.shape)",
            (
                "earliest = jnp.min(jnp.where(still_tied, offered.stable_index, "
                "jnp.iinfo(offered.stable_index.dtype).max), axis=1, keepdims=True)"
            ),
            "still_tied = still_tied & (offered.stable_index == earliest)",
        ),
        "_ordinary_block_step": (
            (
                "offered, channels = _ordinary_block_best("
                "block=block_live_and_stable[0], block_live=block_live_and_stable[1], "
                "block_stable_index=block_live_and_stable[2], flat=flat, dtype=dtype)"
            ),
            "zip(offered[:-1], held[:-1], strict=True)",
            "take = take | (~decided & (offered.stable_index < held.stable_index))",
            "jnp.where(take, challenging, holding)",
            "jnp.where(take, offered_channel, standing_channel)",
        ),
        "_merge_link_rows_ordinary": (
            "index = _ordinary_rank_argmax(rank=rank)",
            "_pick_column(column=identities, index=index)",
            "_pick_column(column=column, index=index)",
        ),
        "_envelope_blocked_ordinary": (
            "functools.partial(_ordinary_block_step, flat=flat, dtype=dtype)",
            "(blocks, live_blocks, stable_blocks)",
        ),
        "_reduce_envelope_ordinary": (
            (
                "reduction = _envelope_blocked_ordinary(links=links, query=query, "
                "block_size=_ordinary_block_size(requested=segment_block_size, "
                "n_segment=n_segment), savings_endpoints=savings_endpoints)"
            ),
        ),
    }
    for name, fragments in obligations.items():
        function = _function(tree=tree, name=name)
        for fragment in fragments:
            _requires(tree=function, fragment=fragment)
    return "stable"
