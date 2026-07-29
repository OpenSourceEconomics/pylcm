"""Candidate blocks own disjoint segment-id ranges at every grid size.

The branch-aware envelope fuses candidates that share a segment id into one
link. Two candidate families sharing an id would let the envelope bracket across
them, so each block's range has to start where the previous block's ends.
"""

from _lcm.egm.nbegm_step import _point_candidate_segment_bases


def test_the_save_to_cliff_block_starts_past_the_savings_node_block() -> None:
    """The cliff base clears every id the dense node family occupies."""
    n_liquid, n_nodes = 200, 200
    node_base, cliff_base = _point_candidate_segment_bases(
        interval_block_end=1_000, n_liquid=n_liquid, n_nodes=n_nodes
    )
    assert cliff_base == node_base + n_liquid * n_nodes


def test_the_savings_node_block_starts_past_the_interval_blocks() -> None:
    """The node base is the first id no per-interval block owns."""
    node_base, _ = _point_candidate_segment_bases(
        interval_block_end=1_000, n_liquid=5, n_nodes=21
    )
    assert node_base == 1_000
