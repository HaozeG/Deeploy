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
    from Deeploy.TileIR.IR.TileBinding import TileBinding


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

    Mirrors the reference MLA_decode_MHA.h idiom exactly:

      axis = axis_names[0] (x) — row-wise (broadcast/reduce across a row):
        row_mask = wakeup_row_mask,  col_mask = (ARCH_NUM_CLUSTER_Y - 1)
        edge    = (cluster_in_group_id_x_<gid> == 0)   [west-edge cluster]

      axis = axis_names[1] (y) — col-wise (broadcast/reduce along a column):
        row_mask = (ARCH_NUM_CLUSTER_X - 1), col_mask = wakeup_col_mask
        edge    = (cluster_in_group_id_y_<gid> == 0)   [south-edge cluster]

      reduce_axis is None — full-group 2-D tree (fallback):
        row_mask = wakeup_row_mask, col_mask = wakeup_col_mask
        edge    = cluster_for_rowwise_<gid>             [diagonal cluster]

    Inter-group / split-K (cross-instance) is NOT handled here; callers
    override masks via spec.global_barrier_before (AxisReduceBroadcast).
    """
    x_axis, y_axis = group.axis_names

    if reduce_axis == x_axis:
        # Row-wise: reduce/broadcast across a row (along x within each row).
        return (
            f"group_info_{gid}.wakeup_row_mask",
            "(ARCH_NUM_CLUSTER_Y - 1)",
            f"(cluster_in_group_id_x_{gid} == 0)",
        )

    if reduce_axis == y_axis:
        # Col-wise: reduce/broadcast along a column (along y within each col).
        return (
            "(ARCH_NUM_CLUSTER_X - 1)",
            f"group_info_{gid}.wakeup_col_mask",
            f"(cluster_in_group_id_y_{gid} == 0)",
        )

    # Full-group: 2-D reduction tree (no axis annotation).
    return (
        f"group_info_{gid}.wakeup_row_mask",
        f"group_info_{gid}.wakeup_col_mask",
        f"cluster_for_rowwise_{gid}",
    )


def _inter_group_masks(gid: str, axis: str = None, group=None) -> tuple:
    """Return (row_mask, col_mask, edge_flag) for cross-instance collectives.

    Inverted masks span corresponding ranks across all group instances.
    Matches SummaGEMM.h:396-397,480-481 (~wakeup_row_mask / ~wakeup_col_mask).

    Axis-based selection (mirrors _row_col_masks for intra-group) applies when:
      (a) split is 2-D (len(split_axes) >= 2): each split axis maps cleanly to
          one physical direction regardless of intra-group shape.
            split_axes[0] → y-direction → (~wakeup_row_mask, ARCH_NUM_CLUSTER_Y-1)
            split_axes[1] → x-direction → (ARCH_NUM_CLUSTER_X-1, ~wakeup_col_mask)
      (b) intra-group is 1-D (group_y == 1): split_axes[0] → row reduction.

    Falls back to full 2-D (both masks inverted) when:
      - 2-D intra-group + 1-D split (instances span both physical dims; a
        single axis cannot distinguish direction) — preserves existing behaviour.
      - axis is None or axis not in split_axes.
    """
    _is_2d_group = group is not None and group.group_y > 1
    _is_2d_split = group is not None and len(group.split_axes) >= 2
    use_axis_based = _is_2d_split or not _is_2d_group

    if use_axis_based and axis is not None and group is not None \
            and group.split_axes and axis in group.split_axes:
        idx = group.split_axes.index(axis)
        if idx == 0:
            return (
                f"(~group_info_{gid}.wakeup_row_mask)",
                "(ARCH_NUM_CLUSTER_Y - 1)",
                f"(cluster_in_group_id_x_{gid} == 0)",
            )
        else:
            return (
                "(ARCH_NUM_CLUSTER_X - 1)",
                f"(~group_info_{gid}.wakeup_col_mask)",
                f"(cluster_in_group_id_y_{gid} == 0)",
            )

    # Full 2-D inter-group reduction
    return (
        f"(~group_info_{gid}.wakeup_row_mask)",
        f"(~group_info_{gid}.wakeup_col_mask)",
        f"cluster_for_rowwise_{gid}",
    )


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
        from Deeploy.TileIR.IR.TileBinding import TileBinding

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

        # Split-K cross-instance reduction: use inverted masks so the DMA reduction
        # tree spans corresponding ranks across all group instances, and use a global
        # barrier (flex_global_barrier_xy) instead of the per-group barrier.
        # Matches SummaGEMM.h:396-397,480-481 (~wakeup_row_mask / ~wakeup_col_mask).
        global_barrier = spec.global_barrier_before
        if global_barrier:
            row_mask = f"(~group_info_{gid}.wakeup_row_mask)"
            col_mask = f"(~group_info_{gid}.wakeup_col_mask)"
            # All clusters in every instance participate; edge_flag = rowwise (same formula)
            edge_flag = f"cluster_for_rowwise_{gid}"

        reduce_rep = {
            "src_name":           spec.src_buffer,
            "dst_name":           spec.src_buffer,  # in-place
            "root_cluster_id":    root_cluster_id,
            "group_id":           gid,
            "collective_op_kind": op_kind,
            "row_mask":           row_mask,
            "col_mask":           col_mask,
            "edge_flag":          edge_flag,
            "global_barrier":     global_barrier,
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
            "global_barrier":  global_barrier,
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
        from Deeploy.TileIR.IR.TileBinding import TileBinding

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
        from Deeploy.TileIR.IR.TileBinding import TileBinding

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
        from Deeploy.TileIR.IR.TileBinding import TileBinding

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
        from Deeploy.TileIR.IR.TileBinding import TileBinding

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
# Strategy 6: GroupShift  — Cannon / Systolic ring-rotation along one axis.
# ---------------------------------------------------------------------------

class GroupShift(CollectiveStrategy):
    """Ring-shift a tile by ``spec.shift_by`` along ``spec.reduce_axis``.

    Uses ``flex_dma_async_pattern_round_shift_{direction}`` DMA primitives.
    Axis "x" → left/right, axis "y" → up (down not yet available, falls back
    to up with barrier for ring semantics).
    """

    # Map (axis, direction_sign) → DMA function name
    _SHIFT_FN_MAP = {
        ("x",  1): "flex_dma_async_pattern_round_shift_right",
        ("x", -1): "flex_dma_async_pattern_round_shift_left",
        ("y", -1): "flex_dma_async_pattern_round_shift_up",
        ("y",  1): None,  # "down" not available in runtime; barrier-only fallback
    }

    def matches(self, spec, group, topology):
        return spec.op == "shift"

    def emit(self, spec, binding, registry) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveGroupShiftTemplate,
        )
        from Deeploy.TileIR.IR.TileBinding import TileBinding

        gid = spec.group_id
        group = registry.get(gid)
        axis_idx = group.axis_index(spec.reduce_axis) if spec.reduce_axis else 0
        axis_name = spec.reduce_axis or group.axis_names[0]
        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"

        # Determine DMA shift function
        shift_dir = 1 if spec.shift_by > 0 else -1
        shift_fn = self._SHIFT_FN_MAP.get((axis_name, shift_dir))

        rep = {
            "src_name":       spec.src_buffer,
            "dst_name":       spec.dst_buffer,
            "group_id":       gid,
            "axis_index":     axis_idx,
            "axis_name":      axis_name,
            "shift_by":       spec.shift_by,
            "shift_fn":       shift_fn,
            "nbytes":         nbytes_placeholder,
            "cluster_id":     None,
            "shard_metadata": None,
        }
        return [
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveGroupShiftTemplate,
                operator_representation=rep,
                op_name=f"tile_collective_group_shift_{gid}",
            )
        ]


# ---------------------------------------------------------------------------
# Strategy 7: GroupBcastAxis — axis-scoped broadcast from a named source rank.
# ---------------------------------------------------------------------------

class GroupBcastAxis(CollectiveStrategy):
    """Axis-scoped broadcast originating from ``spec.from_coord``.

    Differs from :class:`AxisBroadcast` in that the source rank is
    user-specified (not the group root) — this is SUMMA's A-broadcast
    phase, where every column originates from a different rank.
    """

    def matches(self, spec, group, topology):
        return spec.op == "bcast_axis"

    def emit(self, spec, binding, registry) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveGroupBcastAxisTemplate,
        )
        from Deeploy.TileIR.IR.TileBinding import TileBinding

        gid = spec.group_id
        group = registry.get(gid)
        axis_idx = group.axis_index(spec.reduce_axis) if spec.reduce_axis else 0
        row_mask, col_mask, edge_flag = _row_col_masks(gid, spec.reduce_axis, group)
        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"

        # Dynamic from_coord: derive per-cluster edge_flag from the C expression
        # rather than the static group root (e.g. "gid_y" → diagonal cluster).
        if spec.from_coord_expr is not None:
            along = spec.reduce_axis or group.axis_names[0]
            if along == group.axis_names[0]:
                edge_flag = f"(cluster_in_group_id_x_{gid} == ({spec.from_coord_expr}))"
            else:
                edge_flag = f"(cluster_in_group_id_y_{gid} == ({spec.from_coord_expr}))"

        rep = {
            "src_name":       spec.src_buffer,
            "dst_name":       spec.dst_buffer,
            "group_id":       gid,
            "axis_index":     axis_idx,
            "axis_name":      spec.reduce_axis or group.axis_names[0],
            "from_coord":     spec.from_coord_expr if spec.from_coord_expr is not None else spec.from_coord,
            "row_mask":       row_mask,
            "col_mask":       col_mask,
            "edge_flag":      edge_flag,
            "nbytes":         nbytes_placeholder,
            "cluster_id":     None,
            "shard_metadata": None,
        }
        return [
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveGroupBcastAxisTemplate,
                operator_representation=rep,
                op_name=f"tile_collective_group_bcast_axis_{gid}",
            )
        ]


# ---------------------------------------------------------------------------
# Strategy 8: DpReduceStrategy — D.reduce (tl.deeploy.reduce)
#
#   Dispatches on spec.level:
#     "intra_group" — axis-scoped or full-group reduce+broadcast within one
#                     instance, using row/col masks from the group info.
#     "inter_group" — cross-instance reduce+broadcast, using inverted masks
#                     and a global barrier (split-K pattern).
# ---------------------------------------------------------------------------

class DpReduceStrategy(CollectiveStrategy):
    """Lower D.reduce() to hardware reduce + broadcast.

    Handles both intra-group (axis/full-group) and inter-group (inverted-mask)
    cases based on ``spec.level`` and ``spec.axis``.
    """

    def matches(self, spec, group, topology):
        return spec.op == "dp_reduce"

    def emit(self, spec, binding, registry) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveBroadcastTemplate,
            TileCollectiveReduceTemplate,
        )
        from Deeploy.TileIR.IR.TileBinding import TileBinding

        gid = spec.group_id
        root_cluster_id = binding.root_cluster_for(gid, registry)
        op_kind = _collective_op_kind(spec.reduce_op)
        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"

        group = registry.get(gid)
        if spec.level == "inter_group":
            row_mask, col_mask, edge_flag = _inter_group_masks(gid, spec.axis, group)
            global_barrier = True
        else:
            row_mask, col_mask, edge_flag = _row_col_masks(gid, spec.axis, group)
            global_barrier = False

        reduce_rep = {
            "src_name":           spec.src_buffer,
            "dst_name":           spec.src_buffer,
            "root_cluster_id":    root_cluster_id,
            "group_id":           gid,
            "collective_op_kind": op_kind,
            "row_mask":           row_mask,
            "col_mask":           col_mask,
            "edge_flag":          edge_flag,
            "global_barrier":     global_barrier,
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
            "global_barrier":  global_barrier,
            "nbytes":          nbytes_placeholder,
            "cluster_id":      None,
            "shard_metadata":  None,
        }

        return [
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveReduceTemplate,
                operator_representation=reduce_rep,
                op_name=f"tile_dp_reduce_{gid}",
            ),
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveBroadcastTemplate,
                operator_representation=bcast_rep,
                op_name=f"tile_dp_reduce_bcast_{gid}",
            ),
        ]


# ---------------------------------------------------------------------------
# Strategy 9: DpBroadcastStrategy — D.broadcast (tl.deeploy.broadcast)
#
#   Dispatches on spec.level:
#     "intra_group" — axis-scoped broadcast within one instance.
#                     Uses the same GroupBcastAxis pattern: edge cluster is
#                     the one whose rank along *axis* equals *root_expr*.
#     "inter_group" — cross-instance broadcast using inverted masks.
# ---------------------------------------------------------------------------

class DpBroadcastStrategy(CollectiveStrategy):
    """Lower D.broadcast() to a hardware broadcast.

    Handles both intra-group (axis-scoped, dynamic root) and inter-group
    (inverted-mask) cases based on ``spec.level`` and ``spec.axis``.
    """

    def matches(self, spec, group, topology):
        return spec.op == "dp_broadcast"

    def emit(self, spec, binding, registry) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveBroadcastTemplate,
            TileCollectiveGroupBcastAxisTemplate,
        )
        from Deeploy.TileIR.IR.TileBinding import TileBinding

        gid = spec.group_id
        root_cluster_id = binding.root_cluster_for(gid, registry)
        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"

        group = registry.get(gid)
        if spec.level == "inter_group":
            row_mask, col_mask, edge_flag = _inter_group_masks(gid, spec.axis, group)
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
                    op_name=f"tile_dp_bcast_inter_{gid}",
                )
            ]

        # Intra-group: axis-scoped broadcast.  The root is spec.root_expr
        # (C-expression string, e.g. "gid_y"), which varies per cluster.
        # This is identical to the GroupBcastAxis pattern.
        row_mask, col_mask, edge_flag = _row_col_masks(gid, spec.axis, group)

        if spec.root_expr:
            along = spec.axis or group.axis_names[0]
            if along == group.axis_names[0]:
                edge_flag = f"(cluster_in_group_id_x_{gid} == ({spec.root_expr}))"
            else:
                edge_flag = f"(cluster_in_group_id_y_{gid} == ({spec.root_expr}))"

        axis_idx = group.axis_index(spec.axis) if spec.axis else 0
        rep = {
            "src_name":   spec.src_buffer,
            "dst_name":   spec.dst_buffer,
            "group_id":   gid,
            "axis_index": axis_idx,
            "axis_name":  spec.axis or group.axis_names[0],
            "from_coord": spec.root_expr if spec.root_expr else "0",
            "row_mask":   row_mask,
            "col_mask":   col_mask,
            "edge_flag":  edge_flag,
            "nbytes":     nbytes_placeholder,
            "cluster_id": None,
            "shard_metadata": None,
        }
        return [
            TileBinding(
                op_kind="group_collective",
                template=TileCollectiveGroupBcastAxisTemplate,
                operator_representation=rep,
                op_name=f"tile_dp_bcast_intra_{gid}",
            )
        ]


# ---------------------------------------------------------------------------
# Priority list — first match wins
# ---------------------------------------------------------------------------

STRATEGIES: List[Type[CollectiveStrategy]] = [
    DpReduceStrategy,         # D.reduce  (tl.deeploy.reduce)  — explicit level/axis
    DpBroadcastStrategy,      # D.broadcast (tl.deeploy.broadcast) — explicit level/axis
    AxisReduceBroadcast,      # allreduce with axis annotation (SUMMA/FlatAttn pattern)
    AxisBroadcast,            # broadcast
    FullGroupReduceBroadcast, # allreduce without axis annotation (legacy fallback)
    ScatterStrategy,          # scatter
    GatherStrategy,           # gather
    GroupShift,               # ring-shift along a group axis (Cannon / Systolic)
    GroupBcastAxis,           # axis-scoped broadcast from a chosen rank (SUMMA A-phase)
]
