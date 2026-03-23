# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR hardware binding for collective operations.

Separates the logical cluster-group description (``ClusterGroupRegistry``)
from the hardware-level mapping of group names to physical cluster IDs and
the lowering of ``CollectiveOpSpec`` objects to ``TileBinding`` sequences.

Classes
-------
HardwareBinding
    Maps group_id strings to ordered physical cluster ID lists.
    Provides helper methods for bitmask computation, grid dimensions, and
    root-cluster resolution.

CollectiveBackend (ABC)
    Abstract base for hardware-specific collective lowering.

SoftHierCollectiveBackend
    SoftHier-specific implementation using ``flex_dma_async_reduction`` and
    ``flex_dma_async_broadcast`` together with ``GridSyncGroupInfo``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from Deeploy.TileIR.IR.CollectivePrimitives import (
        ClusterGroupRegistry,
        CollectiveOpSpec,
    )
    from Deeploy.TileIR.Midend.TileBindings import TileBinding


@dataclass
class HardwareBinding:
    """Maps logical cluster-group names to physical cluster IDs.

    Parameters
    ----------
    cluster_ids : Dict[str, List[int]]
        ``{group_id: [cluster_id, ...]}`` mapping.  The list order determines
        which cluster is the root (index from ``ClusterGroup.root_instance``).
    """

    cluster_ids: Dict[str, List[int]] = field(default_factory=dict)

    def clusters_for(self, group_id: str) -> List[int]:
        """Return the ordered cluster list for *group_id*."""
        if group_id not in self.cluster_ids:
            raise KeyError(f"HardwareBinding: unknown group_id={group_id!r}")
        return self.cluster_ids[group_id]

    def root_cluster_for(
        self,
        group_id: str,
        registry: "ClusterGroupRegistry",
    ) -> int:
        """Return the physical cluster ID of the root cluster for *group_id*.

        The root is determined by ``ClusterGroup.root_instance`` (an index
        into the cluster_ids list).
        """
        group = registry.get(group_id)
        clusters = self.clusters_for(group_id)
        return clusters[group.root_instance]

    def bitmask_for(self, group_id: str) -> int:
        """Return a bitmask with bits set for each cluster in the group.

        Bit *i* is set when cluster *i* belongs to this group.  Used by
        ``flex_dma_async_reduction`` and ``flex_dma_async_broadcast``.
        """
        mask = 0
        for cid in self.clusters_for(group_id):
            mask |= (1 << cid)
        return mask

    def grid_dims_for(self, group_id: str) -> Tuple[int, int]:
        """Return (x_dim, y_dim) grid dimensions for ``grid_sync_group_init``.

        Currently always returns (num_clusters, 1) — a 1-D grid of clusters.
        Future work may expose a 2-D cluster grid.
        """
        n = len(self.clusters_for(group_id))
        return (n, 1)


class CollectiveBackend(ABC):
    """Abstract base class for hardware-specific collective lowering."""

    @abstractmethod
    def lower(
        self,
        spec: "CollectiveOpSpec",
        binding: "HardwareBinding",
        registry: "ClusterGroupRegistry",
    ) -> List["TileBinding"]:
        """Lower *spec* to a list of hardware TileBindings.

        Parameters
        ----------
        spec :
            Logical description of the collective.
        binding :
            Physical cluster-ID mapping.
        registry :
            Group declarations (for root_instance, strategy, etc.).

        Returns
        -------
        List[TileBinding]
            Ordered sequence of TileBindings that implement the collective.
        """
        ...


class SoftHierCollectiveBackend(CollectiveBackend):
    """SoftHier collective backend.

    Lowers ``CollectiveOpSpec`` objects to SoftHier DMA-based collectives:

    * ``"allreduce"`` → ``TileCollectiveReduceTemplate`` (root performs DMA
      reduction into ``dst_buffer``) followed by
      ``TileCollectiveBroadcastTemplate`` (root broadcasts result back).
    * ``"broadcast"`` → ``TileCollectiveBroadcastTemplate`` only.
    * ``"scatter"`` / ``"gather"`` → not yet implemented (raises
      ``NotImplementedError``).

    Hardware APIs used:
        ``flex_dma_async_reduction``, ``flex_dma_async_broadcast``,
        ``grid_sync_group_barrier_xy``, ``GridSyncGroupInfo``.
    """

    def lower(
        self,
        spec: "CollectiveOpSpec",
        binding: HardwareBinding,
        registry: "ClusterGroupRegistry",
    ) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
            TileCollectiveBroadcastTemplate,
            TileCollectiveReduceTemplate,
        )
        from Deeploy.TileIR.Midend.TileBindings import TileBinding

        group = registry.get(spec.group_id)
        root_cluster_id = binding.root_cluster_for(spec.group_id, registry)
        src_bitmask = binding.bitmask_for(spec.group_id)
        dst_bitmask = binding.bitmask_for(spec.group_id)
        nbytes_placeholder = f"sizeof_buffer_{spec.src_buffer}"  # resolved by caller

        results: List[TileBinding] = []

        if spec.op in ("allreduce",):
            # Step 1: reduce src_buffer from all members → root dst_buffer
            reduce_rep = {
                "src_name": spec.src_buffer,
                "dst_name": spec.dst_buffer,
                "root_cluster_id": root_cluster_id,
                "group_id": spec.group_id,
                "collective_op_kind": _reduce_op_to_c(spec.reduce_op),
                "src_bitmask": hex(src_bitmask),
                "dst_bitmask": hex(dst_bitmask),
                "nbytes": nbytes_placeholder,
                "cluster_id": None,  # no cluster guard — guard is inside template
                "shard_metadata": None,
            }
            results.append(
                TileBinding(
                    op_kind="group_collective",
                    template=TileCollectiveReduceTemplate,
                    operator_representation=reduce_rep,
                    op_name=f"tile_collective_reduce_{spec.group_id}",
                ))

            # Step 2: broadcast root dst_buffer → all members
            bcast_rep = {
                "src_name": spec.dst_buffer,
                "dst_name": spec.src_buffer,
                "root_cluster_id": root_cluster_id,
                "group_id": spec.group_id,
                "src_bitmask": hex(src_bitmask),
                "dst_bitmask": hex(dst_bitmask),
                "nbytes": nbytes_placeholder,
                "cluster_id": None,
                "shard_metadata": None,
            }
            results.append(
                TileBinding(
                    op_kind="group_collective",
                    template=TileCollectiveBroadcastTemplate,
                    operator_representation=bcast_rep,
                    op_name=f"tile_collective_broadcast_{spec.group_id}",
                ))

        elif spec.op == "broadcast":
            bcast_rep = {
                "src_name": spec.src_buffer,
                "dst_name": spec.dst_buffer,
                "root_cluster_id": root_cluster_id,
                "group_id": spec.group_id,
                "src_bitmask": hex(src_bitmask),
                "dst_bitmask": hex(dst_bitmask),
                "nbytes": nbytes_placeholder,
                "cluster_id": None,
                "shard_metadata": None,
            }
            results.append(
                TileBinding(
                    op_kind="group_collective",
                    template=TileCollectiveBroadcastTemplate,
                    operator_representation=bcast_rep,
                    op_name=f"tile_collective_broadcast_{spec.group_id}",
                ))

        else:
            raise NotImplementedError(
                f"SoftHierCollectiveBackend: op={spec.op!r} not yet implemented. "
                "Supported: allreduce, broadcast.")

        return results


def _reduce_op_to_c(reduce_op: str) -> str:
    """Map a reduce_op string to the SoftHier C enum constant."""
    mapping = {
        "sum": "FLEX_DMA_REDUCTION_SUM",
        "max": "FLEX_DMA_REDUCTION_MAX",
        "min": "FLEX_DMA_REDUCTION_MIN",
    }
    return mapping.get(reduce_op, "FLEX_DMA_REDUCTION_SUM")
