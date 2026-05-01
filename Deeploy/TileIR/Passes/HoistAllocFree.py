# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""HoistAllocFreePass — hoist alloc/free bindings out of the outermost loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from Deeploy.TileIR.IR.TileBinding import TileBinding
from Deeploy.TileIR.Passes.Base import TileBindingPass


@dataclass
class HoistAllocFreePass(TileBindingPass):
    """Move alloc/free bindings outside the outermost for-loop nest.

    Without this pass every (bx, by) tile iteration calls flex_l1_malloc /
    flex_l1_free for the same fixed-size L1 scratch buffers.  Because buffer
    sizes are loop-invariant, a single alloc before the loop and a single free
    after it are correct and eliminate the per-iteration allocator overhead.

    Must run after SoftwarePipelinePass (so double-buffer _0/_1 splits exist)
    and before GlobalClusterBarrierPass.
    """

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        # Find the first for_open (outermost loop)
        outer_open_idx = None
        for i, b in enumerate(bindings):
            if b.op_kind in ("for_open", "pipelined_for_open"):
                outer_open_idx = i
                break
        if outer_open_idx is None:
            return bindings

        # Find the matching for_close
        depth = 0
        outer_close_idx = None
        for i in range(outer_open_idx, len(bindings)):
            if bindings[i].op_kind in ("for_open", "pipelined_for_open"):
                depth += 1
            elif bindings[i].op_kind in ("for_close", "pipelined_for_close"):
                depth -= 1
                if depth == 0:
                    outer_close_idx = i
                    break
        if outer_close_idx is None:
            return bindings

        hoisted_allocs: List[TileBinding] = []
        hoisted_frees: List[TileBinding] = []
        loop_body: List[TileBinding] = []

        for b in bindings[outer_open_idx:outer_close_idx + 1]:
            if b.op_kind == "alloc":
                hoisted_allocs.append(b)
            elif b.op_kind == "free":
                hoisted_frees.append(b)
            else:
                loop_body.append(b)

        return (bindings[:outer_open_idx]
                + hoisted_allocs
                + loop_body
                + hoisted_frees
                + bindings[outer_close_idx + 1:])
