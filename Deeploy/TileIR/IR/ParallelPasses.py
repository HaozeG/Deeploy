# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR cluster-group structural passes and CollectiveBinding.

This module provides:

CollectiveBinding
    A TileBinding that carries a CollectiveOpSpec alongside the standard
    NodeTemplate / operator_representation.

GroupAwareBarrierPass
    A TileBindingPass that replaces the generic GlobalClusterBarrierPass when
    cluster groups are in use.  Adjacent ops belonging to the same group get a
    group-scoped barrier (``grid_sync_group_barrier_xy``); ops crossing group
    boundaries or exiting all groups get a global barrier.

CollectiveLoweringPass
    A TileBindingPass that:
      1. Resolves root cluster for every ``alloc_reducer`` binding.
      2. Patches ``cluster_id`` and ``cluster_guard_type`` on alloc_fragment
         bindings inside a group.
      3. Calls ``backend.lower(spec, hw_binding, registry)`` for each
         CollectiveBinding and replaces it with the hardware binding sequence.
      4. Prepends one TileGroupInitTemplate per unique group_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.TileIR.Midend.TileBindings import TileBinding, TileBindingPass

if TYPE_CHECKING:
    from Deeploy.TileIR.IR.CollectivePrimitives import (
        ClusterGroupRegistry,
        CollectiveOpSpec,
        ShardMetadata,
    )
    from Deeploy.TileIR.IR.HardwareBinding import CollectiveBackend, HardwareBinding

# ---------------------------------------------------------------------------
# Barrier templates (used internally by the passes)
# ---------------------------------------------------------------------------

_GLOBAL_BARRIER_TEMPLATE = NodeTemplate("""\
// Global barrier (cross-group or ungrouped cluster switch)
flex_global_barrier_xy();
""")

_GROUP_BARRIER_TEMPLATE = NodeTemplate("""\
// Group barrier: all ${group_id} clusters sync
grid_sync_group_barrier_xy(&group_info_${group_id});
""")

# ---------------------------------------------------------------------------
# CollectiveBinding
# ---------------------------------------------------------------------------

_BARRIER_TRANSPARENT_OP_KINDS = {
    "for_open",
    "for_close",
    "comment",
}


@dataclass
class CollectiveBinding(TileBinding):
    """A TileBinding that carries a CollectiveOpSpec for backend lowering.

    Parameters
    ----------
    spec : CollectiveOpSpec
        Logical description of the collective operation.  Used by
        ``CollectiveLoweringPass`` to call the hardware backend.
    """

    spec: Optional["CollectiveOpSpec"] = None

    def __post_init__(self) -> None:
        # op_kind must be set before calling super().__post_init__
        if self.op_kind != "group_collective":
            self.op_kind = "group_collective"
        super().__post_init__()


# ---------------------------------------------------------------------------
# GroupAwareBarrierPass
# ---------------------------------------------------------------------------


@dataclass
class GroupAwareBarrierPass(TileBindingPass):
    """Insert barriers between operations, respecting cluster group boundaries.

    Rules
    -----
    * Adjacent ops in the **same group** → ``TileGroupBarrierTemplate``
      (group-scoped ``grid_sync_group_barrier_xy``).
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
            if binding.op_kind in _BARRIER_TRANSPARENT_OP_KINDS:
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
                    # Cluster switch — always a global barrier
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
                elif (
                    current_group_id is not None
                    and current_group_id == prev_group_id
                ):
                    # Same group, possibly different op (collective barrier)
                    # GroupAwareBarrierPass intentionally does NOT auto-insert
                    # group barriers here — CollectiveLoweringPass handles that
                    # precisely around collectives.
                    pass

            transformed.append(binding)
            prev_cluster_id = current_cluster_id
            prev_group_id = current_group_id
            prev_seen = True

        return transformed


# ---------------------------------------------------------------------------
# CollectiveLoweringPass
# ---------------------------------------------------------------------------


@dataclass
class CollectiveLoweringPass(TileBindingPass):
    """Lower CollectiveBindings to hardware TileBindings.

    Steps
    -----
    1. Collect all unique ``group_id`` values from the binding list.
    2. Prepend a ``TileGroupInitTemplate`` per group (all clusters execute;
       sandwiched by global barriers).
    3. For every ``alloc_reducer`` binding:
       a. Patch ``cluster_id`` to the root cluster (from hw_binding + registry).
    4. For every ``CollectiveBinding``:
       a. Call ``backend.lower(spec, hw_binding, registry)`` to expand it into
          hardware-level bindings.
    5. Patch ``alloc`` bindings inside a group to use a group-membership guard
       (``cluster_guard_type = "group_membership"``).

    Parameters
    ----------
    registry : ClusterGroupRegistry
        Declared cluster groups.
    hw_binding : HardwareBinding
        Physical cluster-ID mapping.
    backend : CollectiveBackend
        Hardware backend for collective lowering.
    """

    registry: "ClusterGroupRegistry" = field(default=None)
    hw_binding: "HardwareBinding" = field(default=None)
    backend: "CollectiveBackend" = field(default=None)

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        if self.registry is None or self.hw_binding is None or self.backend is None:
            return bindings

        # Merge physical cluster IDs from hw_binding into each ClusterGroup so
        # registry.root_cluster_for() / registry.clusters_for() work below.
        self.hw_binding.populate_registry(self.registry)

        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileGroupContextTemplate,
            TileGroupInitTemplate,
        )

        # Collect unique group_ids (preserve insertion order)
        seen_groups: Set[str] = set()
        ordered_groups: List[str] = []
        for b in bindings:
            gid = self._group_id_of(b)
            if gid is not None and gid not in seen_groups:
                seen_groups.add(gid)
                ordered_groups.append(gid)

        # Build group-init prefix (one per group)
        prefix: List[TileBinding] = []
        if ordered_groups:
            prefix.append(
                TileBinding(
                    op_kind="sync",
                    template=_GLOBAL_BARRIER_TEMPLATE,
                    operator_representation={},
                    op_name="tile_global_barrier_before_group_init",
                ))
            for gid in ordered_groups:
                group = self.registry.get(gid)
                x_dim, y_dim = group.group_x, group.group_y
                prefix.append(
                    TileBinding(
                        op_kind="group_barrier",
                        template=TileGroupInitTemplate,
                        operator_representation={
                            "group_id": gid,
                            "x_dim": x_dim,
                            "y_dim": y_dim,
                            "cluster_id": None,
                        },
                        op_name=f"tile_group_init_{gid}",
                    ))
            prefix.append(
                TileBinding(
                    op_kind="sync",
                    template=_GLOBAL_BARRIER_TEMPLATE,
                    operator_representation={},
                    op_name="tile_global_barrier_after_group_init",
                ))
            # Emit TileGroupContextTemplate after the post-init barrier.
            # num_groups is passed so cluster_active_* uses this_grid_id < num_groups
            # (SummaGEMM pattern) to restrict execution to active group instances.
            for gid in ordered_groups:
                group = self.registry.get(gid)
                prefix.append(
                    TileBinding(
                        op_kind="group_barrier",
                        template=TileGroupContextTemplate,
                        operator_representation={
                            "group_id":   gid,
                            "num_groups": group.num_groups,
                            "cluster_id": None,
                        },
                        op_name=f"tile_group_context_{gid}",
                    ))

        # Process each binding
        transformed: List[TileBinding] = []
        for binding in bindings:
            if binding.op_kind == "alloc_reducer":
                # Patch cluster_id to root cluster
                gid = binding.operator_representation.get("group_id")
                if gid is not None:
                    root_cid = self.registry.root_cluster_for(gid)
                    binding.operator_representation["cluster_id"] = root_cid
                transformed.append(binding)

            elif isinstance(binding, CollectiveBinding) and binding.spec is not None:
                # Lower to hardware bindings
                hw_bindings = self.backend.lower(
                    binding.spec,
                    self.hw_binding,  # still passed for strategy compat
                    self.registry,
                )
                # Resolve nbytes for the placeholder
                nbytes = binding.operator_representation.get("nbytes", 0)
                for hb in hw_bindings:
                    if "nbytes" in hb.operator_representation:
                        if isinstance(hb.operator_representation["nbytes"], str) and \
                                hb.operator_representation["nbytes"].startswith("sizeof_buffer_"):
                            hb.operator_representation["nbytes"] = nbytes
                    transformed.append(hb)

            elif binding.op_kind == "alloc" and self._group_id_of(binding) is not None:
                # Tag group-member allocs for guard generation
                binding.operator_representation["cluster_guard_type"] = "group_membership"
                gid = self._group_id_of(binding)
                cluster_ids = self.registry.clusters_for(gid)
                binding.operator_representation["group_cluster_ids"] = cluster_ids
                transformed.append(binding)

            else:
                transformed.append(binding)

        return prefix + transformed

    @staticmethod
    def _group_id_of(binding: TileBinding) -> Optional[str]:
        """Return the group_id from a binding's shard_metadata, or None."""
        shard_meta = binding.operator_representation.get("shard_metadata", None)
        if shard_meta is None:
            return None
        if hasattr(shard_meta, "group_id"):
            return shard_meta.group_id
        return None
