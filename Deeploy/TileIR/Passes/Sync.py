# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Sync and barrier passes — cluster/group barrier insertion and sync deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.TileIR.Backend.Templates.SoftHierTileTemplates import TileSyncTemplate
from Deeploy.TileIR.IR.TileBinding import (
    _BARRIER_TRANSPARENT_OP_KINDS,
    _GLOBAL_CLUSTER_SWITCH_BARRIER_TEMPLATE,
    TileBinding,
    TileOpKind,
)
from Deeploy.TileIR.Passes.Base import (
    _GLOBAL_BARRIER_TEMPLATE,
    _INTRA_CLUSTER_SYNC_TEMPLATE,
    TileBindingPass,
)

# Barrier-transparent op kinds for GroupAwareBarrierPass (narrower set:
# excludes if_open/if_close/else_open since group transitions inside
# conditionals still need barriers).
_GROUP_AWARE_BARRIER_TRANSPARENT: frozenset = frozenset({
    TileOpKind.for_open,
    TileOpKind.for_close,
    TileOpKind.comment,
})

_GROUP_BARRIER_TEMPLATE = NodeTemplate("""\
// Group barrier: all ${group_id} clusters sync
grid_sync_group_barrier_xy(&group_info_${group_id});
""")


@dataclass
class GlobalClusterBarrierPass(TileBindingPass):
    """Insert a global barrier between cluster_id transitions.

    Barriers are skipped when either the source or destination cluster_id is
    ``None`` (meaning "all clusters"): transitioning between a guarded block
    and an unguarded block does not require cross-cluster synchronization
    because each cluster handles the transition independently.
    """

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        transformed: List[TileBinding] = []
        prev_cluster_id = None
        prev_seen = False

        for binding in bindings:
            if binding.op_kind in _BARRIER_TRANSPARENT_OP_KINDS:
                transformed.append(binding)
                continue

            current_cluster_id = binding.operator_representation.get("cluster_id", None)
            if (prev_seen
                    and current_cluster_id != prev_cluster_id
                    and prev_cluster_id is not None
                    and current_cluster_id is not None):
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
class GroupAwareBarrierPass(TileBindingPass):
    """Insert barriers between operations, respecting cluster group boundaries.

    Rules
    -----
    * Adjacent ops in the **same group** → no automatic group barrier inserted
      (CollectiveLoweringPass handles that precisely around collectives).
    * Adjacent ops with **different (or no) group_id** but the **same
      cluster_id** → no barrier needed.
    * Adjacent ops with a **cluster_id change** regardless of group →
      ``flex_global_barrier_xy()``.
    * Barrier-transparent op kinds (for_open/close, comment) are skipped.
    """

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        transformed: List[TileBinding] = []
        prev_cluster_id = None
        prev_group_id: Optional[str] = None
        prev_seen = False

        for binding in bindings:
            if binding.op_kind in _GROUP_AWARE_BARRIER_TRANSPARENT:
                transformed.append(binding)
                continue

            rep = binding.operator_representation
            current_cluster_id = rep.get("cluster_id", None)
            shard_meta = rep.get("shard_metadata", None)
            current_group_id: Optional[str] = (
                shard_meta.group_id if shard_meta is not None else None
            )

            if prev_seen:
                if current_cluster_id != prev_cluster_id:
                    transformed.append(
                        TileBinding(
                            op_kind="sync",
                            template=_GLOBAL_BARRIER_TEMPLATE,
                            operator_representation={
                                "from_cluster_id": prev_cluster_id,
                                "to_cluster_id": current_cluster_id,
                            },
                            op_name="tile_global_barrier",
                        ))

            transformed.append(binding)
            prev_cluster_id = current_cluster_id
            prev_group_id = current_group_id
            prev_seen = True

        return transformed


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
