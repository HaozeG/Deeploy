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
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Set, Tuple

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

if TYPE_CHECKING:
    from Deeploy.TileIR.IR.CollectivePrimitives import ClusterGroupRegistry

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
    "if_open",
    "if_close",
    "else_open",
]


_BARRIER_TRANSPARENT_OP_KINDS = {
    "for_open",
    "for_close",
    "comment",
    "if_open",
    "if_close",
    "else_open",
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

    When *group_registry* is provided and a pipelined scope belongs to a
    multi-cluster group (``group_x > 1``), a **K-split** strided loop is emitted
    instead of a plain loop:

    * Prologue loads from ``bk = cluster_in_group_id_x_<gid>`` (per-cluster
      K-offset) into stage 0.
    * Main loop runs ``for (bk = cluster_in_group_id_x_<gid>; bk < extent;
      bk += group_x)``.  Each cluster handles every ``group_x``-th K-block.
    * Prefetch guard and offset use the strided step.

    This fixes the 2× numerical error in TP-GEMM where without K-split both
    clusters accumulate the full K sum, causing the allreduce to double the
    result.
    """

    group_registry: Optional["ClusterGroupRegistry"] = None

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

        # Pre-scan: collect group_ids that have *allreduce* collective ops.
        # K-split strided loops are only applied to TP allreduce groups, NOT to
        # directional collectives (bcast_axis, shift) used in SUMMA / Cannon kernels.
        collective_group_ids: Set[str] = set()
        for b in bindings:
            if b.op_kind == "group_collective":
                gid = b.operator_representation.get("group_id")
                if not gid:
                    continue
                from Deeploy.TileIR.IR.ParallelPasses import CollectiveBinding as _CB
                if isinstance(b, _CB) and b.spec is not None:
                    if b.spec.op == "allreduce":
                        collective_group_ids.add(gid)
                else:
                    collective_group_ids.add(gid)  # legacy non-CollectiveBinding path

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
                result, i = self._rewrite_scope(bindings, i, info, buf_dtypes, result,
                                                collective_group_ids)
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

    def _tp_group_info(self, inner_bindings: List["TileBinding"],
                       collective_group_ids: Optional[Set[str]] = None):
        """Detect a TP K-split group (group_x > 1) for a pipelined scope.

        Detection strategy (most to least specific):

        1. Scan *inner_bindings* for a ``shard_metadata.group_id`` that maps to
           a group with ``group_x > 1`` in the registry.  This path fires when
           the cluster_group AttrStmt wraps the For loop body.

        2. Fall back to scanning the registry directly for any group with
           ``group_x > 1`` AND a collective op in *collective_group_ids*.  This is
           needed because TileLang's lowering only attaches the cluster_group
           AttrStmt to the ``T.allreduce`` call, not to the surrounding For loop,
           so inner bindings carry ``shard_metadata=None``.
           Restricting to *collective_group_ids* prevents DP groups (group_x > 1
           but no collectives) from incorrectly triggering K-split.

        Returns ``(group_id, group_x)`` when a TP group is found,
        or ``(None, 1)`` otherwise.
        """
        if self.group_registry is None:
            return None, 1
        # Only allreduce groups should trigger K-split.  Build the effective
        # set once so both strategies share the same restriction.
        _allreduce_groups = collective_group_ids or set()

        # Strategy 1: shard_metadata on inner bindings (most specific).
        # The GEMM and other compute bindings inside T.cluster_group carry
        # shard_metadata.group_id, so we see the group here even without an
        # explicit allreduce AttrStmt wrapping the loop.  Guard with
        # _allreduce_groups to prevent SUMMA / Cannon (bcast_axis / shift)
        # groups from being mistaken for TP K-split groups.
        # TP K-split requires group_y == 1 (1-D group along X). 2-D groups
        # (SUMMA) partition K by instance (gid), not by cluster_in_group_id_x.
        for b in inner_bindings:
            sm = b.operator_representation.get("shard_metadata")
            if sm is None or not hasattr(sm, "group_id") or not sm.group_id:
                continue
            if sm.group_id not in _allreduce_groups:
                continue
            try:
                group = self.group_registry.get(sm.group_id)
                if group.group_x > 1 and int(group.group_y) == 1:
                    return sm.group_id, int(group.group_x)
            except (KeyError, AttributeError):
                pass
        # Strategy 2: registry fallback — groups with allreduce AND group_x > 1.
        # Only match when _allreduce_groups is non-empty (i.e. we have real
        # allreduce info); an empty set means no allreduce → no K-split.
        # Guard: 2-D groups (group_y > 1) are never TP K-split.
        for group in self.group_registry.groups:
            if int(group.group_x) > 1 and int(group.group_y) == 1 and group.group_id in _allreduce_groups:
                return group.group_id, int(group.group_x)
        return None, 1

    def _rewrite_scope(
        self,
        bindings: List[TileBinding],
        i: int,
        info: dict,
        buf_dtypes: Dict[str, str],
        result: List[TileBinding],
        collective_group_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[TileBinding], int]:
        from Deeploy.TileIR.Backend.Templates.SoftHierTileTemplates import (
            ForLoopCloseTemplate,
            ForLoopOpenTemplate,
            ForLoopOpenStridedTemplate,
            IfCloseTemplate,
            IfOpenTemplate,
        )

        num_stages = info["num_stages"]
        loop_var = info["loop_var"]
        extent = info["extent"]
        load_buf_names = info["load_buf_names"]
        close_idx = info["close_idx"]

        inner = bindings[i + 1:close_idx]
        load_bindings = [b for b in inner if b.op_kind == "load"
                         and b.operator_representation.get("dst") in load_buf_names]

        # Track which loads are inside an if_open/if_close conditional block so
        # the extracted prologue and prefetch loads can be wrapped in the same
        # condition.  Without this, loads would become unconditional even though
        # the original TileLang kernel scoped them with e.g. "if gid_x == gid_y".
        _cond_stack: List[str] = []
        _load_condition: Dict[int, Optional[str]] = {}  # id(binding) → condition
        for b in inner:
            if b.op_kind == "if_open":
                _cond_stack.append(b.operator_representation.get("condition", ""))
            elif b.op_kind == "if_close":
                if _cond_stack:
                    _cond_stack.pop()
            elif b.op_kind == "load" and b.operator_representation.get("dst") in load_buf_names:
                _load_condition[id(b)] = _cond_stack[-1] if _cond_stack else None
        compute_bindings = [b for b in inner if b.op_kind in ("gemm", "eltwise", "reduce")]

        # Detect TP K-split: multi-cluster group along the K dimension.
        tp_group_id, tp_stride = self._tp_group_info(inner, collective_group_ids or set())

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

        # For TP K-split (block distribution): rank r handles bk in
        # [r*tiles_per_rank, (r+1)*tiles_per_rank).  Prologue loads from the
        # start of that rank's contiguous block into _0.
        # For non-TP: prologue loads from bk=0 into _0 (existing behaviour).
        if tp_group_id:
            tiles_per_rank = extent // tp_stride  # exact integer division
            rank_var = f"cluster_in_group_id_x_{tp_group_id}"
            tp_start_expr = f"{rank_var} * {tiles_per_rank}"
            tp_end_expr   = f"({rank_var} + 1) * {tiles_per_rank}"
            prologue_subst = tp_start_expr
        else:
            tiles_per_rank = None
            tp_start_expr  = None
            tp_end_expr    = None
            prologue_subst = "0"

        # ---- Prologue: load stage 0 (cluster-specific K offset) before loop ----
        for lb in load_bindings:
            orig_dst = lb.operator_representation["dst"]
            rep = dict(lb.operator_representation)
            rep["dst"] = f"{orig_dst}_0"
            rep["src_offset"] = self._subst_loop_var(
                rep.get("src_offset", 0), loop_var, prologue_subst)
            cond = _load_condition.get(id(lb))
            if cond:
                result.append(TileBinding(op_kind="if_open", template=IfOpenTemplate,
                                          operator_representation={"condition": cond,
                                                                    "cluster_id": None}))
            result.append(TileBinding(op_kind="load", template=lb.template,
                                      operator_representation=rep))
            if cond:
                result.append(TileBinding(op_kind="if_close", template=IfCloseTemplate,
                                          operator_representation={"cluster_id": None}))
        result.append(TileBinding(op_kind="sync", template=_INTRA_CLUSTER_SYNC_TEMPLATE,
                                  operator_representation={"cluster_id": None}))

        # ---- Main loop open ----
        if tp_group_id:
            # TP block distribution: each rank iterates over its contiguous
            # slice [rank*tiles_per_rank, (rank+1)*tiles_per_rank).
            rep_open = {
                "loop_var":  loop_var,
                "min_val":   tp_start_expr,
                "extent":    tp_end_expr,
                "cluster_id": None,
            }
            result.append(TileBinding(op_kind="for_open", template=ForLoopOpenTemplate,
                                      operator_representation=rep_open))
        else:
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
            if tp_group_id:
                # Block distribution: stage alternates relative to rank's block
                # start so _0 always corresponds to the prologue regardless of rank.
                stage_lines.append(
                    f"{dtype}* {orig_dst}_cur = (({loop_var} - {tp_start_expr}) % {num_stages} == 0) "
                    f"? {orig_dst}_0 : {orig_dst}_1;")
                stage_lines.append(
                    f"{dtype}* {orig_dst}_nxt = (({loop_var} - {tp_start_expr}) % {num_stages} == 0) "
                    f"? {orig_dst}_1 : {orig_dst}_0;")
            else:
                # Non-TP: runtime selection between double-buffer stages.
                stage_lines.append(
                    f"{dtype}* {orig_dst}_cur = ({loop_var} % {num_stages} == 0) "
                    f"? {orig_dst}_0 : {orig_dst}_1;")
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
        # For TP block: next iteration is bk+1, guard against rank's block end.
        next_bk_expr = f"{loop_var} + 1"
        guard_expr = f"{next_bk_expr} < {tp_end_expr}" if tp_group_id else f"{next_bk_expr} < {extent}"
        for lb in load_bindings:
            orig_dst = lb.operator_representation["dst"]
            rep = dict(lb.operator_representation)
            rep["dst"] = f"{orig_dst}_nxt"
            rep["src_offset"] = self._subst_loop_var(
                rep.get("src_offset", 0), loop_var, next_bk_expr)
            cond = _load_condition.get(id(lb))
            if cond:
                # Load was originally inside a conditional: emit
                #   if (guard) { if (cond) { load } }
                # using proper if_open/if_close bindings so downstream passes
                # (e.g. DedupSyncPass) see a consistent binding sequence.
                result.append(TileBinding(op_kind="if_open", template=IfOpenTemplate,
                                          operator_representation={"condition": guard_expr,
                                                                    "cluster_id": None}))
                result.append(TileBinding(op_kind="if_open", template=IfOpenTemplate,
                                          operator_representation={"condition": cond,
                                                                    "cluster_id": None}))
                result.append(TileBinding(op_kind="load", template=lb.template,
                                          operator_representation=rep))
                result.append(TileBinding(op_kind="if_close", template=IfCloseTemplate,
                                          operator_representation={"cluster_id": None}))
                result.append(TileBinding(op_kind="if_close", template=IfCloseTemplate,
                                          operator_representation={"cluster_id": None}))
            else:
                # Unconditional load: wrap body in guard only.
                inner_src = lb.template.template._source
                guarded_src = f"if ({guard_expr}) {{\n{inner_src}}}\n"
                result.append(TileBinding(
                    op_kind="load",
                    template=NodeTemplate(guarded_src),
                    operator_representation=rep,
                ))

        # ---- Main loop body: non-load bindings in original order ----
        # Includes gemm/eltwise/reduce AND group_collective ops (group_shift,
        # group_bcast_axis, etc.) so that directional collectives inside a
        # T.Pipelined loop survive the double-buffering rewrite.
        _pipelined_scope_kinds = {"pipelined_for_open", "pipelined_for_close"}
        for b in inner:
            if b.op_kind == "load" and b.operator_representation.get("dst") in load_buf_names:
                continue  # promoted to prefetch stage above
            if b.op_kind in _pipelined_scope_kinds:
                continue  # handled by the outer scope logic
            rep = dict(b.operator_representation)
            # Remap staged load-buffer names to their _cur versions.
            # Also handles prefixed variants (e.g. "DeeployNetwork_A_local" → "A_local_cur")
            # that arise when the TIR buffer name includes the network prefix.
            for key in ("A", "B", "src_name", "dst_name", "src_buffer", "dst_buffer"):
                val = rep.get(key)
                if not isinstance(val, str):
                    continue
                if val in load_buf_names:
                    rep[key] = f"{val}_cur"
                else:
                    for _short in load_buf_names:
                        if val.endswith("_" + _short):
                            rep[key] = f"{_short}_cur"
                            break
            # Preserve CollectiveBinding subclass so CollectiveLoweringPass can dispatch.
            from Deeploy.TileIR.IR.ParallelPasses import CollectiveBinding as _CollBind
            if isinstance(b, _CollBind):
                result.append(_CollBind(
                    op_kind=b.op_kind, template=b.template,
                    operator_representation=rep, spec=b.spec, op_name=b.op_name,
                ))
            else:
                result.append(TileBinding(op_kind=b.op_kind, template=b.template,
                                          operator_representation=rep))

        # ---- Single sync per iteration ----
        result.append(TileBinding(op_kind="sync", template=_INTRA_CLUSTER_SYNC_TEMPLATE,
                                  operator_representation={"cluster_id": None}))

        # ---- Main loop close ----
        result.append(TileBinding(op_kind="for_close", template=ForLoopCloseTemplate,
                                  operator_representation={"loop_var": loop_var}))

        return result, close_idx + 1


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


@dataclass
class DedupSyncPass(TileBindingPass):
    """Collapse runs of consecutive ``flex_intra_cluster_sync()`` bindings.

    Multiple passes emit intra-cluster syncs independently — the visitor
    appends one after every load/copy/gemm/reduce/fill/eltwise, and
    ``SoftwarePipelinePass`` adds one per pipelined-loop iteration.  After
    promoting prefetch loads out of the loop body, several visitor-emitted
    syncs can end up adjacent.  This pass keeps only the first of each run.

    Only intra-cluster syncs are collapsed; global barriers and group
    barriers use distinct templates and are left untouched.
    """

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        from Deeploy.TileIR.Backend.Templates.SoftHierTileTemplates import (
            TileSyncTemplate,
        )
        intra_sync_templates = {TileSyncTemplate, _INTRA_CLUSTER_SYNC_TEMPLATE}

        def _is_intra_sync(b: TileBinding) -> bool:
            return b.op_kind == "sync" and b.template in intra_sync_templates

        result: List[TileBinding] = []
        prev_was_intra_sync = False
        for b in bindings:
            cur_is_intra_sync = _is_intra_sync(b)
            if cur_is_intra_sync and prev_was_intra_sync:
                continue
            result.append(b)
            prev_was_intra_sync = cur_is_intra_sync
        return result


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

        # TODO: check with hoisting results
        ctxt, _ = transformed_block.hoisting(ctxt)
        return ctxt, transformed_block
