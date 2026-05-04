# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""SoftwarePipelinePass — double-buffer software pipelining for T.Pipelined loops."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import TileGroupBarrierTemplate
from Deeploy.TileIR.Backend.Templates.SoftHierTileTemplates import TileSyncTemplate
from Deeploy.TileIR.IR.CollectiveBinding import CollectiveBinding
from Deeploy.TileIR.IR.TileBinding import TileBinding, TileOpKind
from Deeploy.TileIR.Passes.Base import (
    _INTRA_CLUSTER_SYNC_TEMPLATE,
    TileBindingPass,
)

if TYPE_CHECKING:
    from Deeploy.TileIR.IR.CollectivePrimitives import ClusterGroupRegistry


@dataclass
class ScopePhases:
    """Classification of bindings inside a T.Pipelined scope.

    Separates the scope body into three schedulable phases so the pipeline
    rewriter can map each phase to the correct pipeline slot (prologue,
    prefetch, compute body).

    Attributes
    ----------
    loads : List[TileBinding]
        ``load`` bindings that write to staged (double-buffered) buffers.
    comms : List[TileBinding]
        ``CollectiveBinding`` ops (bcast_axis, shift, …) whose ``src_buffer``
        is a staged buffer.  These are moved into the prefetch slot alongside
        the loads so the DM core can disseminate the next tile while the
        compute core runs GEMM on the current tile.
    follow_syncs : List[TileBinding]
        ``sync`` bindings that immediately follow a promoted load or COMM.
        These guarded the old "load → sync → GEMM" pattern; after pipeline
        rewriting the compute core reads ``_cur`` so they are redundant and
        must be excluded to allow DM/compute overlap.
    empty_conditionals : List[TileBinding]
        ``if_open`` / ``if_close`` bindings whose entire contents were promoted
        (loads + comms + follow_syncs).  Keeping these would produce empty
        ``if (...) {}`` ghost blocks in the output, so they are excluded.
    conditional_map : Dict[int, Optional[str]]
        Maps ``id(binding)`` → the innermost ``if_open`` condition that lexically
        wraps the binding (or ``None`` if unconditional).
    staged_buffers : Set[str]
        The set of L1 buffer names that are double-buffered.
    """

    loads: List["TileBinding"]
    comms: List["TileBinding"]
    follow_syncs: List["TileBinding"]
    empty_conditionals: List["TileBinding"]
    conditional_map: Dict[int, Optional[str]]
    staged_buffers: Set[str]


@dataclass
class SoftwarePipelinePass(TileBindingPass):
    """Transform ``pipelined_for_open`` scopes into double-buffer software pipeline.

    For each ``pipelined_for_open`` / ``pipelined_for_close`` scope with
    ``num_stages >= 2``:

    1. Duplicate every L1 load-target buffer (A_local → A_local_0, A_local_1).
    2. Classify the scope body via ``_classify_phases`` into three phases:

       * **LOAD** — ``load`` ops writing to staged buffers.
       * **COMM** — ``CollectiveBinding`` ops (``bcast_axis``, ``shift``, …)
         whose ``src_buffer`` is a staged buffer.
       * **COMPUTE** — ``gemm`` / ``eltwise`` / ``reduce`` ops plus any
         remaining pass-through bindings.

    3. Emit a **prologue** that runs ``[LOAD(_0) + COMM(_0)]`` before the loop
       then a single ``flex_intra_cluster_sync()``.  This ensures the first
       tile is fully loaded *and* disseminated before compute begins.
    4. Rewrite the loop body as::

           [LOAD_nxt + COMM_nxt]  (DM core, guarded by bk+1 < extent)
           COMPUTE_cur            (compute core, overlaps with DM above)
           flex_intra_cluster_sync()

       The DM core prefetches *and* broadcasts the next tile concurrently with
       the compute core running GEMM on the current tile.  For SUMMA / Cannon
       this eliminates the serialized bcast-then-GEMM pattern and overlaps the
       inter-cluster communication with arithmetic.

    When *group_registry* is provided and a pipelined scope belongs to a
    multi-cluster group (``group_x > 1``) with an allreduce collective, a
    **K-split** strided loop is emitted instead of a plain loop.
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
                if isinstance(b, CollectiveBinding) and b.spec is not None:
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
        _allreduce_groups = collective_group_ids or set()

        # Strategy 1: shard_metadata on inner bindings (most specific).
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
        for group in self.group_registry.groups:
            if int(group.group_x) > 1 and int(group.group_y) == 1 and group.group_id in _allreduce_groups:
                return group.group_id, int(group.group_x)
        return None, 1

    @staticmethod
    def _classify_phases(inner: List[TileBinding], load_buf_names: Set[str]) -> "ScopePhases":
        """Classify bindings inside a pipelined scope into pipeline phases.

        Returns a ``ScopePhases`` with:

        * ``loads`` — ``load`` bindings writing to staged buffers.
        * ``comms`` — ``CollectiveBinding`` ops whose ``spec.src_buffer`` is a
          staged buffer.  These are moved into the prefetch slot alongside loads
          so the DM core can disseminate the next tile while the compute core
          runs GEMM on the current tile concurrently.
        * ``conditional_map`` — maps ``id(binding)`` to the innermost
          ``if_open`` condition lexically enclosing it, or ``None``.
        """
        loads = [b for b in inner
                 if b.op_kind == "load" and b.operator_representation.get("dst") in load_buf_names]
        comms = [b for b in inner
                 if isinstance(b, CollectiveBinding) and b.spec is not None
                 and b.spec.src_buffer in load_buf_names]

        staged_ids = {id(b) for b in loads + comms}
        cond_stack: List[str] = []
        conditional_map: Dict[int, Optional[str]] = {}
        follow_syncs: List[TileBinding] = []
        prev_was_staged = False
        for b in inner:
            if b.op_kind == "if_open":
                cond_stack.append(b.operator_representation.get("condition", ""))
            elif b.op_kind == "if_close":
                if cond_stack:
                    cond_stack.pop()

            if id(b) in staged_ids:
                conditional_map[id(b)] = cond_stack[-1] if cond_stack else None
                prev_was_staged = True
            elif b.op_kind == "sync" and prev_was_staged:
                follow_syncs.append(b)
            else:
                prev_was_staged = False

        # Detect if_open/if_close pairs whose contents are entirely promoted.
        all_promoted_ids = {id(b) for b in loads + comms + follow_syncs}
        empty_conditionals: List[TileBinding] = []
        idx = 0
        while idx < len(inner):
            b = inner[idx]
            if b.op_kind == "if_open":
                depth = 1
                j = idx + 1
                while j < len(inner) and depth > 0:
                    if inner[j].op_kind == "if_open":
                        depth += 1
                    elif inner[j].op_kind == "if_close":
                        depth -= 1
                    j += 1
                close_j = j - 1  # index of the matching if_close
                contents = inner[idx + 1:close_j]
                non_structural = [c for c in contents if c.op_kind not in ("if_open", "if_close")]
                if non_structural and all(id(c) in all_promoted_ids for c in non_structural):
                    empty_conditionals.append(b)
                    empty_conditionals.append(inner[close_j])
            idx += 1

        return ScopePhases(
            loads=loads,
            comms=comms,
            follow_syncs=follow_syncs,
            empty_conditionals=empty_conditionals,
            conditional_map=conditional_map,
            staged_buffers=load_buf_names,
        )

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
            IfCloseTemplate,
            IfOpenTemplate,
        )

        num_stages = info["num_stages"]
        loop_var = info["loop_var"]
        extent = info["extent"]
        load_buf_names = info["load_buf_names"]
        close_idx = info["close_idx"]

        inner = bindings[i + 1:close_idx]
        phases = self._classify_phases(inner, load_buf_names)
        load_bindings = phases.loads
        _load_condition = phases.conditional_map
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
        else:
            tiles_per_rank = None
            tp_start_expr  = None
            tp_end_expr    = None

        # ---- Prologue: prime N-1 stages (bk=0 through bk=N-2) ----
        # For N=2 (double-buffering) this loads exactly stage _0 — identical
        # to the old single-stage prologue.  For N>=3 it loads enough stages
        # so the loop's prefetch target is valid from iteration 0.
        comm_group_id: Optional[str] = phases.comms[0].spec.group_id if phases.comms else None
        prologue_num_stages = num_stages - 1
        for s in range(prologue_num_stages):
            if s >= extent:
                break  # extent smaller than prologue size (pathological)
            if tp_group_id:
                stage_subst = str(tp_start_expr) if s == 0 else f"({tp_start_expr} + {s})"
            else:
                stage_subst = str(s)
            for lb in load_bindings:
                orig_dst = lb.operator_representation["dst"]
                rep = dict(lb.operator_representation)
                rep["dst"] = f"{orig_dst}_{s}"
                rep["src_offset"] = self._subst_loop_var(
                    rep.get("src_offset", 0), loop_var, stage_subst)
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
            for cb in phases.comms:
                orig_buf = cb.spec.src_buffer
                rep = dict(cb.operator_representation)
                rep["src_name"] = f"{orig_buf}_{s}"
                rep["dst_name"] = f"{orig_buf}_{s}"
                result.append(CollectiveBinding(
                    op_kind="group_collective",
                    template=cb.template,
                    operator_representation=rep,
                    spec=cb.spec,
                    op_name=cb.op_name,
                ))
        # After prologue loads and bcasts, align DM and compute cores.
        result.append(TileBinding(op_kind="sync", template=_INTRA_CLUSTER_SYNC_TEMPLATE,
                                  operator_representation={"cluster_id": None}))

        # ---- Main loop open ----
        if tp_group_id:
            rep_open = {
                "loop_var":  loop_var,
                "min_val":   tp_start_expr,
                "extent":    tp_end_expr,
                "cluster_id": None,
                "group_id": tp_group_id,
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

        # ---- Sync segment at loop start (SUMMA-like pattern only) ----
        if comm_group_id:
            result.append(TileBinding(
                op_kind="sync",
                template=TileGroupBarrierTemplate,
                operator_representation={"group_id": comm_group_id, "cluster_id": None},
            ))
            result.append(TileBinding(op_kind="sync", template=_INTRA_CLUSTER_SYNC_TEMPLATE,
                                      operator_representation={"cluster_id": None}))

        # ---- Stage pointer declarations (_cur / _nxt) ----
        stage_lines = []
        for lb in load_bindings:
            orig_dst = lb.operator_representation["dst"]
            dtype = buf_dtypes.get(orig_dst, "fp16")

            if num_stages == 2:
                # Optimized path for double-buffering (preserves identical C output)
                if tp_group_id:
                    mod_cond = f"(({loop_var} - {tp_start_expr}) % 2 == 0)"
                else:
                    mod_cond = f"({loop_var} % 2 == 0)"
                stage_lines.append(
                    f"{dtype}* {orig_dst}_cur = {mod_cond} ? {orig_dst}_0 : {orig_dst}_1;")
                stage_lines.append(
                    f"{dtype}* {orig_dst}_nxt = {mod_cond} ? {orig_dst}_1 : {orig_dst}_0;")
            else:
                # General N-stage pipeline: pointer array with modulo indexing
                stage_names = ", ".join(f"{orig_dst}_{s}" for s in range(num_stages))
                stage_lines.append(
                    f"{dtype}* {orig_dst}_stages[{num_stages}] = {{{stage_names}}};")
                if tp_group_id:
                    stage_lines.append(
                        f"{dtype}* {orig_dst}_cur = {orig_dst}_stages"
                        f"[({loop_var} - {tp_start_expr}) % {num_stages}];")
                    stage_lines.append(
                        f"{dtype}* {orig_dst}_nxt = {orig_dst}_stages"
                        f"[({loop_var} - {tp_start_expr} + {num_stages - 1}) % {num_stages}];")
                else:
                    stage_lines.append(
                        f"{dtype}* {orig_dst}_cur = {orig_dst}_stages"
                        f"[{loop_var} % {num_stages}];")
                    stage_lines.append(
                        f"{dtype}* {orig_dst}_nxt = {orig_dst}_stages"
                        f"[({loop_var} + {num_stages - 1}) % {num_stages}];")
        stage_select_source = "\n".join(stage_lines) + "\n"
        result.append(TileBinding(
            op_kind="comment",
            template=NodeTemplate(stage_select_source),
            operator_representation={"cluster_id": None},
        ))

        # ---- Prefetch loads (DM core): load stage N-1 steps ahead, guarded ----
        next_bk_expr = f"{loop_var} + {num_stages - 1}"
        guard_expr = f"{next_bk_expr} < {tp_end_expr}" if tp_group_id else f"{next_bk_expr} < {extent}"
        for lb in load_bindings:
            orig_dst = lb.operator_representation["dst"]
            rep = dict(lb.operator_representation)
            rep["dst"] = f"{orig_dst}_nxt"
            rep["src_offset"] = self._subst_loop_var(
                rep.get("src_offset", 0), loop_var, next_bk_expr)
            cond = _load_condition.get(id(lb))
            if cond:
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
                inner_src = lb.template.template._source
                guarded_src = f"if ({guard_expr}) {{\n{inner_src}}}\n"
                result.append(TileBinding(
                    op_kind="load",
                    template=NodeTemplate(guarded_src),
                    operator_representation=rep,
                ))

        # ---- Prefetch COMM (DM core): disseminate next staged tile, guarded ----
        for cb in phases.comms:
            orig_buf = cb.spec.src_buffer
            rep = dict(cb.operator_representation)
            rep["src_name"] = f"{orig_buf}_nxt"
            rep["dst_name"] = f"{orig_buf}_nxt"
            result.append(TileBinding(op_kind="if_open", template=IfOpenTemplate,
                                      operator_representation={"condition": guard_expr,
                                                                "cluster_id": None}))
            result.append(CollectiveBinding(
                op_kind="group_collective",
                template=cb.template,
                operator_representation=rep,
                spec=cb.spec,
                op_name=cb.op_name,
            ))
            result.append(TileBinding(op_kind="if_close", template=IfCloseTemplate,
                                      operator_representation={"cluster_id": None}))

        # ---- Main loop body: compute bindings in original order ----
        _pipelined_scope_kinds = {"pipelined_for_open", "pipelined_for_close"}
        _skip_ids = {id(b) for b in phases.comms + phases.follow_syncs + phases.empty_conditionals}
        _compute_start = len(result)
        for b in inner:
            if b.op_kind == "load" and b.operator_representation.get("dst") in load_buf_names:
                continue  # promoted to prefetch stage above
            if id(b) in _skip_ids:
                continue  # promoted comm, follow-sync, or empty ghost conditional
            if b.op_kind in _pipelined_scope_kinds:
                continue  # handled by the outer scope logic
            rep = dict(b.operator_representation)
            # Remap staged load-buffer names to their _cur versions.
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
            if isinstance(b, CollectiveBinding):
                result.append(CollectiveBinding(
                    op_kind=b.op_kind, template=b.template,
                    operator_representation=rep, spec=b.spec, op_name=b.op_name,
                ))
            else:
                result.append(TileBinding(op_kind=b.op_kind, template=b.template,
                                          operator_representation=rep))

        # Strip trailing intra-cluster syncs from the compute body.
        # For COMM pipelines the loop-top group-barrier + sync provides
        # inter-iteration synchronization; for non-COMM pipelines the
        # trailing sync emitted below serves the same purpose.
        _intra_sync_templates = {_INTRA_CLUSTER_SYNC_TEMPLATE, TileSyncTemplate}
        while len(result) > _compute_start:
            _last = result[-1]
            if _last.op_kind == "sync" and _last.template in _intra_sync_templates:
                result.pop()
            else:
                break

        # ---- Trailing sync (non-COMM pipelines only) ----
        if not comm_group_id:
            result.append(TileBinding(op_kind="sync", template=_INTRA_CLUSTER_SYNC_TEMPLATE,
                                      operator_representation={"cluster_id": None}))

        # ---- Main loop close ----
        result.append(TileBinding(op_kind="for_close", template=ForLoopCloseTemplate,
                                  operator_representation={"loop_var": loop_var}))

        # ---- Post-loop sync (COMM pipelines) ----
        # For COMM pipelines the loop-top barrier synchronizes iterations
        # but the *last* iteration has no next barrier.  Without this sync
        # the DM core's TileStore can read the GEMM output buffer before
        # the compute core (flex_is_first_core) has finished writing.
        if comm_group_id:
            result.append(TileBinding(op_kind="sync", template=_INTRA_CLUSTER_SYNC_TEMPLATE,
                                      operator_representation={"cluster_id": None}))

        return result, close_idx + 1
