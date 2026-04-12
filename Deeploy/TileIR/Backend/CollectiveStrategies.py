# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Strategy-dispatched collective lowering for SoftHier.

Each ``CollectiveStrategy`` subclass handles a subset of collective ops
and emits ``TileBinding`` objects using the correct SoftHier DMA primitives.

The critical correctness fix over the previous implementation:
    flex_dma_async_reduction / flex_dma_async_broadcast take a
    (row_mask, col_mask) pair, NOT a flat bitmask.  The correct values
    are runtime fields of ``GridSyncGroupInfo``:
        row_mask = group_info_<gid>.wakeup_row_mask
        col_mask = group_info_<gid>.wakeup_col_mask
    These are computed by grid_sync_group_init at runtime, not at
    compile time.  Passing compile-time hex literals (the old approach)
    accidentally works only when group_y == 1.

Edge-cluster gating (copied from SummaGEMM.h pattern):
    Only the cluster at the edge of each row/column issues the DMA.
    The runtime identifies it via cluster_for_rowwise / cluster_for_colwise
    fields that TileGroupContextTemplate declares near kernel entry.

Priority order — ``STRATEGIES`` list — first match wins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Type

if TYPE_CHECKING:
    from Deeploy.TileIR.IR.CollectivePrimitives import (
        ClusterGroup,
        CollectiveOpSpec,
    )
    from Deeploy.TileIR.IR.HardwareBinding import HardwareBinding, HwTopology
    from Deeploy.TileIR.Midend.TileBindings import TileBinding


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CollectiveStrategy(ABC):
    """Abstract base for a single collective lowering strategy."""

    @abstractmethod
    def matches(
        self,
        spec: "CollectiveOpSpec",
        group: "ClusterGroup",
        topology: Optional["HwTopology"],
    ) -> bool:
        """Return True if this strategy handles *spec* for the given group."""
        ...

    @abstractmethod
    def emit(
        self,
        spec: "CollectiveOpSpec",
        binding: "HardwareBinding",
        registry,
    ) -> List["TileBinding"]:
        """Emit hardware TileBindings implementing *spec*."""
        ...


# ---------------------------------------------------------------------------
# Helper: C expression for row/col masks based on reduce_axis
# ---------------------------------------------------------------------------

def _row_col_masks(
    gid: str,
    reduce_axis: Optional[str],
    group: "ClusterGroup",
) -> tuple:
    """Return (row_mask_expr, col_mask_expr, edge_flag_expr) C strings.

    For an axis-scoped collective:
      - row-axis reduce: row_mask = wakeup_row_mask, col = wakeup_col_mask
      - col-axis reduce: col_mask = wakeup_col_mask, row = wakeup_row_mask
      - edge cluster identified by cluster_for_rowwise / cluster_for_colwise

    For a full-group collective (reduce_axis is None):
      - use wakeup_row_mask and wakeup_col_mask, edge = cluster_for_rowwise.
    """
    if reduce_axis is not None and reduce_axis == group.axis_names[1]:
        # Reduce along the y (column) axis
        row_mask = f"group_info_{gid}.wakeup_row_mask"
        col_mask = f"group_info_{gid}.wakeup_col_mask"
        edge_flag = f"cluster_for_colwise_{gid}"
    else:
        # Default: reduce along x (row) axis or full-group
        row_mask = f"group_info_{gid}.wakeup_row_mask"
        col_mask = f"group_info_{gid}.wakeup_col_mask"
        edge_flag = f"cluster_for_rowwise_{gid}"
    return row_mask, col_mask, edge_flag


def _collective_op_kind(reduce_op: str) -> str:
    """Map reduce_op string to SoftHier C enum constant."""
    mapping = {
        "sum": "COLLECTIVE_REDADD_FP_16",
        "max": "COLLECTIVE_REDMAX_FP_16",
    }
    if reduce_op not in mapping:
        raise NotImplementedError(
            f"reduce_op={reduce_op!r} not supported. Supported: {list(mapping.keys())}"
        )
    return mapping[reduce_op]


# ---------------------------------------------------------------------------
# Strategy 1: AxisReduceBroadcast
#   Fires when: op == "allreduce" AND (reduce_axis is set OR src_layout.partial)
#   Pattern: single axis-scoped reduce + broadcast using edge-cluster gating
#   and runtime wakeup_*_mask.  This is the SUMMA/FlatAttention pattern.
# ---------------------------------------------------------------------------

class AxisReduceBroadcast(CollectiveStrategy):
    """Axis-scoped allreduce via row/col DMA primitives (SUMMA pattern).

    Matches when:
      - op == "allreduce", AND
      - reduce_axis is set OR src_layout.partial is set

    Emits:
      1. TileCollectiveReduceTemplate with runtime wakeup_row/col_mask
      2. TileCollectiveBroadcastTemplate with the same masks
    """

    def matches(self, spec, group, topology):
        if spec.op != "allreduce":
            return False
        if spec.reduce_axis is not None:
            return True
        if spec.src_layout is not None and spec.src_layout.is_partial:
            return True
        return False

    def emit(self, spec, binding, registry) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveBroadcastTemplate,
            TileCollectiveReduceTemplate,
        )
        from Deeploy.TileIR.Midend.TileBindings import TileBinding

        group = registry.get(spec.group_id)
        gid = spec.group_id
        root_cluster_id = binding.root_cluster_for(gid, registry)

        # Determine reduce axis
        reduce_axis = spec.reduce_axis
        if reduce_axis is None and spec.src_layout is not None:
            reduce_axis = spec.src_layout.reduce_axis()

        row_mask, col_mask, edge_flag = _row_col_masks(gid, reduce_axis, group)
        op_kind = _collective_op_kind(spec.reduce_op)

        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"

        reduce_rep = {
            "src_name":           spec.src_buffer,
            "dst_name":           spec.src_buffer,  # in-place
            "root_cluster_id":    root_cluster_id,
            "group_id":           gid,
            "collective_op_kind": op_kind,
            "row_mask":           row_mask,
            "col_mask":           col_mask,
            "edge_flag":          edge_flag,
            "nbytes":             nbytes_placeholder,
            "cluster_id":         None,
            "shard_metadata":     None,
        }
        bcast_rep = {
            "src_name":        spec.src_buffer,
            "dst_name":        spec.src_buffer,
            "root_cluster_id": root_cluster_id,
            "group_id":        gid,
            "row_mask":        row_mask,
            "col_mask":        col_mask,
            "edge_flag":       edge_flag,
            "nbytes":          nbytes_placeholder,
            "cluster_id":      None,
            "shard_metadata":  None,
        }

        return [
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveReduceTemplate,
                operator_representation=reduce_rep,
                op_name=f"tile_collective_reduce_{gid}",
            ),
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveBroadcastTemplate,
                operator_representation=bcast_rep,
                op_name=f"tile_collective_broadcast_{gid}",
            ),
        ]


# ---------------------------------------------------------------------------
# Strategy 2: AxisBroadcast
#   Fires when: op == "broadcast"
# ---------------------------------------------------------------------------

class AxisBroadcast(CollectiveStrategy):
    """Axis-scoped broadcast using runtime wakeup_*_mask."""

    def matches(self, spec, group, topology):
        return spec.op == "broadcast"

    def emit(self, spec, binding, registry) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveBroadcastTemplate,
        )
        from Deeploy.TileIR.Midend.TileBindings import TileBinding

        group = registry.get(spec.group_id)
        gid = spec.group_id
        root_cluster_id = binding.root_cluster_for(gid, registry)

        reduce_axis = spec.reduce_axis
        row_mask, col_mask, edge_flag = _row_col_masks(gid, reduce_axis, group)
        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"

        rep = {
            "src_name":        spec.src_buffer,
            "dst_name":        spec.dst_buffer,
            "root_cluster_id": root_cluster_id,
            "group_id":        gid,
            "row_mask":        row_mask,
            "col_mask":        col_mask,
            "edge_flag":       edge_flag,
            "nbytes":          nbytes_placeholder,
            "cluster_id":      None,
            "shard_metadata":  None,
        }

        return [
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveBroadcastTemplate,
                operator_representation=rep,
                op_name=f"tile_collective_broadcast_{gid}",
            )
        ]


# ---------------------------------------------------------------------------
# Strategy 3: FullGroupReduceBroadcast  (legacy / fallback)
#   Fires when: op == "allreduce" and no axis annotation
#   Same as old code but now correctly uses runtime masks.
# ---------------------------------------------------------------------------

class FullGroupReduceBroadcast(CollectiveStrategy):
    """Full-group allreduce fallback — reduce to root then broadcast back.

    Uses runtime ``wakeup_row_mask`` / ``wakeup_col_mask`` (not compile-time
    hex literals), making it correct for both 1-D and 2-D groups.
    """

    def matches(self, spec, group, topology):
        return spec.op == "allreduce"

    def emit(self, spec, binding, registry) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveBroadcastTemplate,
            TileCollectiveReduceTemplate,
        )
        from Deeploy.TileIR.Midend.TileBindings import TileBinding

        group = registry.get(spec.group_id)
        gid = spec.group_id
        root_cluster_id = binding.root_cluster_for(gid, registry)
        # Use row-wise pattern as default for full-group
        row_mask, col_mask, edge_flag = _row_col_masks(gid, None, group)
        op_kind = _collective_op_kind(spec.reduce_op)
        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"

        reduce_rep = {
            "src_name":           spec.src_buffer,
            "dst_name":           spec.src_buffer,
            "root_cluster_id":    root_cluster_id,
            "group_id":           gid,
            "collective_op_kind": op_kind,
            "row_mask":           row_mask,
            "col_mask":           col_mask,
            "edge_flag":          edge_flag,
            "nbytes":             nbytes_placeholder,
            "cluster_id":         None,
            "shard_metadata":     None,
        }
        bcast_rep = {
            "src_name":        spec.src_buffer,
            "dst_name":        spec.src_buffer,
            "root_cluster_id": root_cluster_id,
            "group_id":        gid,
            "row_mask":        row_mask,
            "col_mask":        col_mask,
            "edge_flag":       edge_flag,
            "nbytes":          nbytes_placeholder,
            "cluster_id":      None,
            "shard_metadata":  None,
        }

        return [
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveReduceTemplate,
                operator_representation=reduce_rep,
                op_name=f"tile_collective_reduce_{gid}",
            ),
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveBroadcastTemplate,
                operator_representation=bcast_rep,
                op_name=f"tile_collective_broadcast_{gid}",
            ),
        ]


# ---------------------------------------------------------------------------
# Strategy 4: ScatterStrategy
# ---------------------------------------------------------------------------

class ScatterStrategy(CollectiveStrategy):
    """Root distributes chunks to all group members."""

    def matches(self, spec, group, topology):
        return spec.op == "scatter"

    def emit(self, spec, binding, registry) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveScatterTemplate,
        )
        from Deeploy.TileIR.Midend.TileBindings import TileBinding

        gid = spec.group_id
        root_cluster_id = binding.root_cluster_for(gid, registry)
        cluster_ids = binding.clusters_for(gid)
        num_members = len(cluster_ids)
        cluster_ids_c = "{" + ", ".join(str(c) for c in cluster_ids) + "}"
        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"

        rep = {
            "src_name":        spec.src_buffer,
            "dst_name":        spec.dst_buffer,
            "root_cluster_id": root_cluster_id,
            "group_id":        gid,
            "cluster_ids_c":   cluster_ids_c,
            "num_members":     num_members,
            "chunk_nbytes":    f"(({nbytes_placeholder}) / {num_members})",
            "cluster_id":      None,
            "shard_metadata":  None,
        }

        return [
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveScatterTemplate,
                operator_representation=rep,
                op_name=f"tile_collective_scatter_{gid}",
            )
        ]


# ---------------------------------------------------------------------------
# Strategy 5: GatherStrategy
# ---------------------------------------------------------------------------

class GatherStrategy(CollectiveStrategy):
    """Root pulls one chunk from each group member."""

    def matches(self, spec, group, topology):
        return spec.op == "gather"

    def emit(self, spec, binding, registry) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveGatherTemplate,
        )
        from Deeploy.TileIR.Midend.TileBindings import TileBinding

        gid = spec.group_id
        root_cluster_id = binding.root_cluster_for(gid, registry)
        cluster_ids = binding.clusters_for(gid)
        num_members = len(cluster_ids)
        cluster_ids_c = "{" + ", ".join(str(c) for c in cluster_ids) + "}"
        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"

        rep = {
            "src_name":        spec.src_buffer,
            "dst_name":        spec.dst_buffer,
            "root_cluster_id": root_cluster_id,
            "group_id":        gid,
            "cluster_ids_c":   cluster_ids_c,
            "num_members":     num_members,
            "chunk_nbytes":    f"(({nbytes_placeholder}) / {num_members})",
            "cluster_id":      None,
            "shard_metadata":  None,
        }

        return [
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveGatherTemplate,
                operator_representation=rep,
                op_name=f"tile_collective_gather_{gid}",
            )
        ]


# ---------------------------------------------------------------------------
# Priority list — first match wins
# ---------------------------------------------------------------------------

STRATEGIES: List[Type[CollectiveStrategy]] = [
    AxisReduceBroadcast,      # allreduce with axis annotation (SUMMA/FlatAttn pattern)
    AxisBroadcast,            # broadcast
    FullGroupReduceBroadcast, # allreduce without axis annotation (legacy fallback)
    ScatterStrategy,          # scatter
    GatherStrategy,           # gather
]
