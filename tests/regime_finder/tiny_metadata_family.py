"""Literal complete six-layout oracle for the source-bound example.

This module reads the ORIGINAL declaration fixture and has no producer/checker
imports. The pure original children callback is interpreted separately. It
enumerates every coherent categorical choice and every life-stage cut directly;
costs follow cardinality products and explicit original dependency incidences.
"""
from __future__ import annotations

import hashlib
from itertools import product
import json
import math
from pathlib import Path

from .metadata_oracle import ScalarSourceOracle, nondominated


def original():
    return json.loads(Path(__file__).with_name("fixtures_metadata").joinpath("tiny_original.json").read_text())


def full_family():
    declaration = original()
    assets = declaration["domains"]["assets_large"]["size"]
    detailed = declaration["domains"]["family4"]["values"]
    original_callback = declaration["callbacks"]["children"]["original"]
    oracle = ScalarSourceOracle()
    image = [oracle.call(original_callback, {"family": value}) for value in detailed]
    coarse_size = len(set(image))
    candidates = []
    # The early utility consumes detail. If cell1 is coarse, the identity law
    # into cell2 requires cell2 coarse too; a detailed source can feed either.
    for coarse1, coarse2 in product((False, True), repeat=2):
        if coarse1 and not coarse2:
            continue
        projection = {0: False, 1: coarse1, 2: coarse2}
        for cut in (False, True):
            blocks = [[0], [1], [2]] if cut else [[0, 1], [2]]
            work = storage = clarity = 0
            for block in blocks:
                axes = {"assets": assets}
                for cell in block:
                    axes["children" if projection[cell] else "family"] = coarse_size if projection[cell] else len(detailed)
                rows = math.prod(axes.values())
                outgoing = sum(source in block and target not in block for source, target in [(0, 1), (1, 2)])
                work += (len(block) + outgoing) * rows
                storage += len(block) * rows
                clarity += 4 + (2 if any(projection[cell] for cell in block) else 0)
            candidates.append({"id": "C" + str(len(candidates)), "coarse_cells": [cell for cell in projection if projection[cell]],
                               "blocks": blocks, "cost": {"work": work, "storage": storage, "clarity": clarity}})
    kept = nondominated(candidates, lambda item: tuple(item["cost"][key] for key in ("work", "storage", "clarity")))
    return {"schema": "independent-tiny-family/v1", "complete": True,
            "input_sha256": hashlib.sha256(json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "source": "tests.regime_finder/fixtures_metadata/tiny_original.json",
            "method": "all two Boolean late-cell projections subject to exact identity-law closure; both life-stage cut masks; literal allocated products",
            "trusted_boundary": "declared pure scalar children AST and explicit original identity transitions",
            "characteristic_map": image, "candidate_count": len(candidates), "candidates": candidates,
            "frontier_ids": [item["id"] for item in kept],
            "numerical_bellman_rows": 0, "cartesian_tensor_allocated": False}


if __name__ == "__main__":
    print(json.dumps(full_family(), indent=2))
