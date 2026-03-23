# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR binding abstractions.

Main classes
------------
TileBinding
    Represents one TileIR operation.  Owns the ``NodeTemplate`` used to emit
    code, the operator representation dict, and an optional per-op
    ``CodeTransformation`` pipeline (looked up by op-kind from
    ``Transformations.py``).

TileBindingPass
    Base class for structural passes that rewrite a list of ``TileBinding``
    objects before per-op code transforms run.  Subclass and override
    ``apply(bindings) -> bindings``.

GlobalClusterBarrierPass
    Built-in ``TileBindingPass``.  Walks the binding list and inserts an
    unguarded ``flex_global_barrier_xy();`` binding whenever adjacent
    operations target different ``cluster_id`` values.  Applied by default.

TileBindingPipeline
    Holds an ordered list of ``TileBinding`` objects (one per emitted op) and
    an ordered list of ``TileBindingPass`` objects.  Canonical usage::

        pipeline = visitor.visit_bindings(primfunc, ctxt)
        pipeline.add_binding_pass(MyPass())   # optional extra passes
        ctxt, eb = pipeline.codeTransform(ctxt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

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

_GLOBAL_CLUSTER_SWITCH_BARRIER_TEMPLATE = NodeTemplate("""\
// Global barrier on cluster_id switch (${from_cluster_id} -> ${to_cluster_id})
flex_global_barrier_xy();
""")

# TODO: consider defining as compute, memory, sync, etc op kinds
TileOpKind = Literal[
    "alloc",
    "free",
    "load",
    "store",
    "copy",
    "fill",
    "reduce",
    "gemm",
    "eltwise",
    "for_open",
    "for_close",
    "sync",
    "comment",
]


_BARRIER_TRANSPARENT_OP_KINDS = {
    "for_open",
    "for_close",
    "comment",
}


@dataclass
class TileBinding:
    """Represents a single TileIR operation with per-op transformation."""

    op_kind: TileOpKind
    template: NodeTemplate
    operator_representation: Dict
    op_name: Optional[str] = None
    code_transformer: Optional[CodeTransformation] = None

    def __post_init__(self) -> None:
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


@dataclass
class TileBindingPass:
    """Base class for structural passes over ordered TileBindings."""

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        return bindings


@dataclass
class GlobalClusterBarrierPass(TileBindingPass):
    """Insert an unguarded global barrier between cluster_id transitions."""

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        transformed: List[TileBinding] = []
        prev_cluster_id = None
        prev_seen = False

        for binding in bindings:
            if binding.op_kind in _BARRIER_TRANSPARENT_OP_KINDS:
                transformed.append(binding)
                continue

            current_cluster_id = binding.operator_representation.get("cluster_id", None)
            if prev_seen and current_cluster_id != prev_cluster_id:
                transformed.append(
                    TileBinding(
                        op_kind="sync",
                        template=_GLOBAL_CLUSTER_SWITCH_BARRIER_TEMPLATE,
                        operator_representation={
                            "from_cluster_id": prev_cluster_id,
                            "to_cluster_id": current_cluster_id,
                        },
                        op_name="tile_global_cluster_switch_barrier",
                    ))

            transformed.append(binding)
            prev_cluster_id = current_cluster_id
            prev_seen = True

        return transformed


@dataclass
class TileBindingPipeline:
    """Ordered list of TileBindings for one TileLang PrimFunc."""

    bindings: List[TileBinding] = field(default_factory=list)
    binding_passes: List[TileBindingPass] = field(
        default_factory=lambda: [GlobalClusterBarrierPass()])

    def add(self, binding: TileBinding) -> None:
        self.bindings.append(binding)

    def bind(self) -> ExecutionBlock:
        """Materialize a raw ExecutionBlock without code transformations."""
        execution_block = ExecutionBlock()
        for binding in self.bindings:
            binding.bind(execution_block)
        return execution_block

    def add_binding_pass(self, binding_pass: TileBindingPass) -> None:
        """Append a structural pass to be applied before per-op transforms."""
        self.binding_passes.append(binding_pass)

    def _apply_binding_passes(self, bindings: List[TileBinding]) -> List[TileBinding]:
        transformed = bindings
        for binding_pass in self.binding_passes:
            transformed = binding_pass.apply(transformed)
        return transformed

    def codeTransform(
        self,
        ctxt: NetworkContext,
        execution_block: Optional[ExecutionBlock] = None,
        verbose: CodeGenVerbosity = _NoVerbosity,
    ) -> Tuple[NetworkContext, ExecutionBlock]:
        """Apply per-op transformations and assemble the transformed block."""
        transformed_block = ExecutionBlock()
        transformed_bindings = self._apply_binding_passes(self.bindings)

        for binding in transformed_bindings:
            ctxt, op_block = binding.codeTransform(ctxt, verbose)
            for snippet in op_block.codeSnippets:
                transformed_block.addRight(
                    snippet.template,
                    dict(snippet.operatorRepresentation),
                )

        # TODO: check with hoisting results
        ctxt, _ = transformed_block.hoisting(ctxt)
        return ctxt, transformed_block
