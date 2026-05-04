# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileBindingPipeline — ordered binding list with configurable pass pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from Deeploy.DeeployTypes import (
    CodeGenVerbosity,
    ExecutionBlock,
    NetworkContext,
    _NoVerbosity,
)
from Deeploy.TileIR.IR.TileBinding import TileBinding
from Deeploy.TileIR.Passes.Base import TileBindingPass
from Deeploy.TileIR.Passes.HoistAllocFree import HoistAllocFreePass
from Deeploy.TileIR.Passes.SoftwarePipeline import SoftwarePipelinePass
from Deeploy.TileIR.Passes.SpatzVectorization import SpatzVectorizationPass
from Deeploy.TileIR.Passes.Sync import DedupSyncPass, GlobalClusterBarrierPass

if TYPE_CHECKING:
    from Deeploy.TileIR.IR.CollectivePrimitives import ClusterGroupRegistry


@dataclass
class TileBindingPipeline:
    """Ordered list of TileBindings for one TileLang PrimFunc."""

    bindings: List[TileBinding] = field(default_factory=list)
    group_registry: Optional["ClusterGroupRegistry"] = field(default=None, repr=False)
    binding_passes: Optional[List[TileBindingPass]] = field(default=None)

    def __post_init__(self) -> None:
        if self.binding_passes is None:
            self.binding_passes = [
                SoftwarePipelinePass(group_registry=self.group_registry),
                HoistAllocFreePass(),
                SpatzVectorizationPass(),
                GlobalClusterBarrierPass(),
                DedupSyncPass(),
            ]

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

        ctxt, _ = transformed_block.hoisting(ctxt)
        return ctxt, transformed_block
