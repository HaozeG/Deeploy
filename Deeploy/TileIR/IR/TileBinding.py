# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR core data types — TileBinding and operation kind classification."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from Deeploy.DeeployTypes import (
    CodeGenVerbosity,
    CodeSnippet,
    CodeTransformation,
    ExecutionBlock,
    NetworkContext,
    NodeTemplate,
    _NoVerbosity,
)
from Deeploy.TileIR.Backend.Transformations import get_tile_op_transformer


class TileOpCategory(enum.Enum):
    """Semantic category for TileIR operation kinds."""
    MEMORY = "memory"
    COMPUTE = "compute"
    CONTROL_FLOW = "control_flow"
    SYNC = "sync"
    COLLECTIVE = "collective"
    MISC = "misc"


class TileOpKind(str, enum.Enum):
    """Semantic operation kinds for TileIR bindings.

    Each member's value is the string used in operator_representation and
    template dispatch.  ``str`` base means ``b.op_kind == "load"`` works.
    The ``.category`` property returns a :class:`TileOpCategory`.
    """
    # --- memory ---
    alloc = "alloc"
    free = "free"
    load = "load"
    store = "store"
    copy = "copy"
    fill = "fill"
    # --- compute ---
    reduce = "reduce"
    gemm = "gemm"
    eltwise = "eltwise"
    # --- control flow ---
    for_open = "for_open"
    for_close = "for_close"
    pipelined_for_open = "pipelined_for_open"
    pipelined_for_close = "pipelined_for_close"
    if_open = "if_open"
    if_close = "if_close"
    else_open = "else_open"
    # --- sync ---
    sync = "sync"
    global_barrier = "global_barrier"
    # --- collective ---
    group_collective = "group_collective"
    group_barrier = "group_barrier"
    alloc_reducer = "alloc_reducer"
    group_preamble = "group_preamble"
    intra_cluster_reduce = "intra_cluster_reduce"
    # --- misc ---
    comment = "comment"
    block_preamble = "block_preamble"
    math_preamble = "math_preamble"
    runtime_assert = "runtime_assert"
    runtime_assume = "runtime_assume"
    cumsum = "cumsum"

    @property
    def category(self) -> TileOpCategory:
        return _OP_CATEGORY_MAP[self]

    def __str__(self) -> str:
        return self.value


_OP_CATEGORY_MAP: Dict[TileOpKind, TileOpCategory] = {
    TileOpKind.alloc: TileOpCategory.MEMORY,
    TileOpKind.free: TileOpCategory.MEMORY,
    TileOpKind.load: TileOpCategory.MEMORY,
    TileOpKind.store: TileOpCategory.MEMORY,
    TileOpKind.copy: TileOpCategory.MEMORY,
    TileOpKind.fill: TileOpCategory.MEMORY,
    TileOpKind.reduce: TileOpCategory.COMPUTE,
    TileOpKind.gemm: TileOpCategory.COMPUTE,
    TileOpKind.eltwise: TileOpCategory.COMPUTE,
    TileOpKind.for_open: TileOpCategory.CONTROL_FLOW,
    TileOpKind.for_close: TileOpCategory.CONTROL_FLOW,
    TileOpKind.pipelined_for_open: TileOpCategory.CONTROL_FLOW,
    TileOpKind.pipelined_for_close: TileOpCategory.CONTROL_FLOW,
    TileOpKind.if_open: TileOpCategory.CONTROL_FLOW,
    TileOpKind.if_close: TileOpCategory.CONTROL_FLOW,
    TileOpKind.else_open: TileOpCategory.CONTROL_FLOW,
    TileOpKind.sync: TileOpCategory.SYNC,
    TileOpKind.global_barrier: TileOpCategory.SYNC,
    TileOpKind.group_collective: TileOpCategory.COLLECTIVE,
    TileOpKind.group_barrier: TileOpCategory.COLLECTIVE,
    TileOpKind.alloc_reducer: TileOpCategory.COLLECTIVE,
    TileOpKind.intra_cluster_reduce: TileOpCategory.COMPUTE,
    TileOpKind.group_preamble: TileOpCategory.CONTROL_FLOW,
    TileOpKind.comment: TileOpCategory.MISC,
    TileOpKind.block_preamble: TileOpCategory.MISC,
    TileOpKind.math_preamble: TileOpCategory.MISC,
    TileOpKind.runtime_assert: TileOpCategory.MISC,
    TileOpKind.runtime_assume: TileOpCategory.MISC,
    TileOpKind.cumsum: TileOpCategory.COMPUTE,
}

# Op kinds that barrier passes should skip (transparent to barrier insertion).
# Includes if_open/if_close/else_open (used by GlobalClusterBarrierPass).
_BARRIER_TRANSPARENT_OP_KINDS: frozenset = frozenset({
    TileOpKind.for_open,
    TileOpKind.for_close,
    TileOpKind.comment,
    TileOpKind.if_open,
    TileOpKind.if_close,
    TileOpKind.else_open,
})

_GLOBAL_CLUSTER_SWITCH_BARRIER_TEMPLATE = NodeTemplate("""\
// Global barrier on cluster_id switch (${from_cluster_id} -> ${to_cluster_id})
flex_global_barrier_xy();
""")


@dataclass
class TileBinding:
    """Represents a single TileIR operation with per-op transformation."""

    op_kind: TileOpKind
    template: NodeTemplate
    operator_representation: Dict
    op_name: Optional[str] = None
    code_transformer: Optional[CodeTransformation] = None

    def __post_init__(self) -> None:
        # Coerce string values to TileOpKind enum members
        if isinstance(self.op_kind, str) and not isinstance(self.op_kind, TileOpKind):
            object.__setattr__(self, "op_kind", TileOpKind(self.op_kind))
        if self.op_name is None:
            self.op_name = f"tile_{self.op_kind}"
        if self.code_transformer is None:
            self.code_transformer = get_tile_op_transformer(self.op_kind)

    def bind(self, execution_block: ExecutionBlock) -> None:
        """Append this operation to an ExecutionBlock."""
        execution_block.addRight(self.template, dict(self.operator_representation))

    def codeTransform(
        self,
        ctxt: NetworkContext,
        verbose: CodeGenVerbosity = _NoVerbosity,
    ) -> Tuple[NetworkContext, ExecutionBlock]:
        """Apply this binding's per-op CodeTransformation pipeline."""
        execution_block = ExecutionBlock(
            CodeSnippet(self.template, dict(self.operator_representation))
        )
        ctxt, execution_block = self.code_transformer.transform(
            ctxt,
            execution_block,
            self.op_name,
            verbose,
        )
        return ctxt, execution_block
