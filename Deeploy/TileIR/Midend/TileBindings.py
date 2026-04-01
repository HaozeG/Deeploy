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

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

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
    "pipelined_for_open",
    "pipelined_for_close",
    "sync",
    "comment",
    "block_preamble",
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


_INTRA_CLUSTER_SYNC_TEMPLATE = NodeTemplate("flex_intra_cluster_sync();\n")


@dataclass
class SoftwarePipelinePass(TileBindingPass):
    """Transform ``pipelined_for_open`` scopes into double-buffer software pipeline.

    For each ``pipelined_for_open`` / ``pipelined_for_close`` scope with
    ``num_stages >= 2``:

    1. Duplicate every L1 load-target buffer (A_local → A_local_0, A_local_1).
    2. Emit a **prologue** that loads stage 0 before the loop (ONE sync).
    3. Rewrite the loop body so the DM core prefetches stage ``(bk+1)%2`` while
       the compute core runs GEMM on stage ``bk%2`` — both run concurrently
       between the single ``flex_intra_cluster_sync()`` per iteration.

    The DM core and RedMule / compute cores are distinct hardware cores inside a
    SoftHier cluster and execute concurrently between ``flex_intra_cluster_sync``
    calls.  No async DMA API is required.
    """

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        # ---------------------------------------------------------------
        # Pre-scan: find all pipelined scopes and their load buffer names
        # ---------------------------------------------------------------
        scope_info: Dict[int, dict] = {}  # for_open index -> metadata
        for idx, b in enumerate(bindings):
            if b.op_kind != "pipelined_for_open":
                continue
            num_stages = b.operator_representation.get("num_stages", 2)
            if num_stages < 2:
                continue
            loop_var = b.operator_representation["loop_var"]
            extent = b.operator_representation["extent"]

            depth = 1
            j = idx + 1
            load_buf_names: Set[str] = set()
            close_idx = idx + 1
            while j < len(bindings) and depth > 0:
                if bindings[j].op_kind == "pipelined_for_open":
                    depth += 1
                elif bindings[j].op_kind == "pipelined_for_close":
                    depth -= 1
                    if depth == 0:
                        close_idx = j
                if depth > 0 and bindings[j].op_kind == "load":
                    dst = bindings[j].operator_representation.get("dst", "")
                    if dst:
                        load_buf_names.add(dst)
                j += 1

            scope_info[idx] = {
                "num_stages": num_stages,
                "loop_var": loop_var,
                "extent": extent,
                "load_buf_names": load_buf_names,
                "close_idx": close_idx,
            }

        if not scope_info:
            return bindings  # nothing to do

        all_dup_names: Set[str] = set()
        for info in scope_info.values():
            all_dup_names.update(info["load_buf_names"])

        # ---------------------------------------------------------------
        # Main rewrite pass
        # ---------------------------------------------------------------
        buf_dtypes: Dict[str, str] = {}  # buffer name -> C type (from alloc bindings)
        result: List[TileBinding] = []
        i = 0
        while i < len(bindings):
            b = bindings[i]

            # -- Duplicate alloc for load buffers --
            if b.op_kind == "alloc":
                name = b.operator_representation.get("name", "")
                if name in all_dup_names:
                    dtype = b.operator_representation.get("dtype", "fp16")
                    buf_dtypes[name] = dtype
                    num_stages = self._stages_for(name, scope_info)
                    for s in range(num_stages):
                        rep_s = dict(b.operator_representation)
                        rep_s["name"] = f"{name}_{s}"
                        result.append(TileBinding(op_kind="alloc", template=b.template,
                                                  operator_representation=rep_s))
                    i += 1
                    continue

            # -- Duplicate free for load buffers --
            if b.op_kind == "free":
                name = b.operator_representation.get("name", "")
                if name in all_dup_names:
                    num_stages = self._stages_for(name, scope_info)
                    for s in range(num_stages - 1, -1, -1):
                        rep_s = dict(b.operator_representation)
                        rep_s["name"] = f"{name}_{s}"
                        result.append(TileBinding(op_kind="free", template=b.template,
                                                  operator_representation=rep_s))
                    i += 1
                    continue

            # -- Rewrite pipelined for scope --
            if i in scope_info:
                info = scope_info[i]
                result, i = self._rewrite_scope(bindings, i, info, buf_dtypes, result)
                continue

            result.append(b)
            i += 1

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stages_for(buf_name: str, scope_info: Dict[int, dict]) -> int:
        for info in scope_info.values():
            if buf_name in info["load_buf_names"]:
                return info["num_stages"]
        return 2

    @staticmethod
    def _subst_loop_var(expr, loop_var: str, replacement: str):
        """Replace bare occurrences of *loop_var* in a C-expression string."""
        if isinstance(expr, int):
            return expr
        return re.sub(r'\b' + re.escape(loop_var) + r'\b', f'({replacement})', str(expr))

    def _rewrite_scope(
        self,
        bindings: List[TileBinding],
        i: int,
        info: dict,
        buf_dtypes: Dict[str, str],
        result: List[TileBinding],
    ) -> Tuple[List[TileBinding], int]:
        from Deeploy.TileIR.Backend.Templates.SoftHierTileTemplates import (
            ForLoopCloseTemplate,
            ForLoopOpenTemplate,
        )

        num_stages = info["num_stages"]
        loop_var = info["loop_var"]
        extent = info["extent"]
        load_buf_names = info["load_buf_names"]
        close_idx = info["close_idx"]

        inner = bindings[i + 1:close_idx]
        load_bindings = [b for b in inner if b.op_kind == "load"
                         and b.operator_representation.get("dst") in load_buf_names]
        compute_bindings = [b for b in inner if b.op_kind in ("gemm", "eltwise", "reduce")]

        if not load_bindings or not compute_bindings:
            # Fall back to plain for-loop
            rep_open = dict(bindings[i].operator_representation)
            rep_open.pop("num_stages", None)
            result.append(TileBinding(op_kind="for_open", template=ForLoopOpenTemplate,
                                      operator_representation=rep_open))
            for b in inner:
                result.append(b)
            result.append(TileBinding(op_kind="for_close", template=ForLoopCloseTemplate,
                                      operator_representation={"loop_var": loop_var}))
            return result, close_idx + 1

        # ---- Prologue: load stage 0 before the loop ----
        for lb in load_bindings:
            orig_dst = lb.operator_representation["dst"]
            rep = dict(lb.operator_representation)
            rep["dst"] = f"{orig_dst}_0"
            rep["src_offset"] = self._subst_loop_var(rep.get("src_offset", 0), loop_var, "0")
            result.append(TileBinding(op_kind="load", template=lb.template,
                                      operator_representation=rep))
        result.append(TileBinding(op_kind="sync", template=_INTRA_CLUSTER_SYNC_TEMPLATE,
                                  operator_representation={"cluster_id": None}))

        # ---- Main loop open ----
        rep_open = {
            "loop_var": loop_var,
            "min_val": 0,
            "extent": extent,
            "cluster_id": None,
        }
        result.append(TileBinding(op_kind="for_open", template=ForLoopOpenTemplate,
                                  operator_representation=rep_open))

        # ---- Stage pointer declarations (_cur / _nxt) ----
        stage_lines = []
        for lb in load_bindings:
            orig_dst = lb.operator_representation["dst"]
            dtype = buf_dtypes.get(orig_dst, "fp16")
            # cur: buffer for the current iteration (already loaded in prologue or prev iter)
            stage_lines.append(
                f"{dtype}* {orig_dst}_cur = ({loop_var} % {num_stages} == 0) "
                f"? {orig_dst}_0 : {orig_dst}_1;")
            # nxt: buffer to prefetch into for the next iteration
            stage_lines.append(
                f"{dtype}* {orig_dst}_nxt = ({loop_var} % {num_stages} == 0) "
                f"? {orig_dst}_1 : {orig_dst}_0;")
        stage_select_source = "\n".join(stage_lines) + "\n"
        result.append(TileBinding(
            op_kind="comment",
            template=NodeTemplate(stage_select_source),
            operator_representation={"cluster_id": None},
        ))

        # ---- Prefetch loads (DM core): load next stage, guarded ----
        guard_expr = f"{loop_var} + 1 < {extent}"
        for lb in load_bindings:
            orig_dst = lb.operator_representation["dst"]
            rep = dict(lb.operator_representation)
            rep["dst"] = f"{orig_dst}_nxt"
            rep["src_offset"] = self._subst_loop_var(
                rep.get("src_offset", 0), loop_var, f"{loop_var} + 1")
            # Wrap load body in a guard
            inner_src = lb.template.template._source
            guarded_src = f"if ({guard_expr}) {{\n{inner_src}}}\n"
            result.append(TileBinding(
                op_kind="load",
                template=NodeTemplate(guarded_src),
                operator_representation=rep,
            ))

        # ---- Compute bindings (compute core): use _cur buffers ----
        for cb in compute_bindings:
            rep = dict(cb.operator_representation)
            for k in ("A", "B"):
                if rep.get(k) in load_buf_names:
                    rep[k] = f"{rep[k]}_cur"
            result.append(TileBinding(op_kind=cb.op_kind, template=cb.template,
                                      operator_representation=rep))

        # ---- Single sync per iteration ----
        result.append(TileBinding(op_kind="sync", template=_INTRA_CLUSTER_SYNC_TEMPLATE,
                                  operator_representation={"cluster_id": None}))

        # ---- Main loop close ----
        result.append(TileBinding(op_kind="for_close", template=ForLoopCloseTemplate,
                                  operator_representation={"loop_var": loop_var}))

        return result, close_idx + 1


@dataclass
class TileBindingPipeline:
    """Ordered list of TileBindings for one TileLang PrimFunc."""

    bindings: List[TileBinding] = field(default_factory=list)
    binding_passes: List[TileBindingPass] = field(
        default_factory=lambda: [SoftwarePipelinePass(), GlobalClusterBarrierPass()])

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
