# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR hardware binding for collective operations."""

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
class HwTopology:
    """Physical cluster grid description.

    Parameters
    ----------
    Px : int  ARCH_NUM_CLUSTER_X
    Py : int  ARCH_NUM_CLUSTER_Y
    """

    Px: int
    Py: int = 1

    def total(self) -> int:
        return self.Px * self.Py

    def pos_of(self, cluster_id: int) -> Tuple[int, int]:
        return (cluster_id % self.Px, cluster_id // self.Px)

    def cluster_of(self, x: int, y: int) -> int:
        return y * self.Px + x


@dataclass
class HardwareBinding:
    """Maps logical cluster-group names to physical cluster IDs.

    Parameters
    ----------
    cluster_ids : Dict[str, List[int]]
        ``{group_id: [physical_cluster_id, ...]}`` for all clusters across
        all group instances.
    topology : HwTopology or None
        Physical grid description.
    """

    cluster_ids: Dict[str, List[int]] = field(default_factory=dict)
    topology: Optional[HwTopology] = None

    def clusters_for(self, group_id: str) -> List[int]:
        if group_id not in self.cluster_ids:
            raise KeyError(f"HardwareBinding: unknown group_id={group_id!r}")
        return self.cluster_ids[group_id]

    def root_cluster_for(
        self,
        group_id: str,
        registry: "ClusterGroupRegistry",
    ) -> int:
        group = registry.get(group_id)
        clusters = self.clusters_for(group_id)
        root_rank = group.root_instance
        if root_rank >= len(clusters):
            raise IndexError(
                f"HardwareBinding: root_rank={root_rank} out of range for "
                f"group_id={group_id!r} with {len(clusters)} clusters"
            )
        return clusters[root_rank]

    def grid_dims_for(self, group_id: str, registry: Optional["ClusterGroupRegistry"] = None) -> Tuple[int, int]:
        """Return (x_dim, y_dim) for grid_sync_group_init."""
        if registry is not None and registry.contains(group_id):
            group = registry.get(group_id)
            return (group.group_x, group.group_y)
        n = len(self.clusters_for(group_id))
        return (n, 1)

    def group_instances(
        self,
        group_id: str,
        registry: "ClusterGroupRegistry",
    ) -> List[List[int]]:
        group = registry.get(group_id)
        k = group.num_ranks_per_instance
        all_clusters = self.clusters_for(group_id)
        return [all_clusters[i * k:(i + 1) * k] for i in range(group.num_groups)]

    def instance_of(
        self,
        group_id: str,
        registry: "ClusterGroupRegistry",
        cluster_id: int,
    ) -> int:
        for idx, instance in enumerate(self.group_instances(group_id, registry)):
            if cluster_id in instance:
                return idx
        return -1

    def populate_registry(self, registry: "ClusterGroupRegistry") -> None:
        """Embed physical_cluster_ids into each ClusterGroup in the registry.

        After this call, ``registry.root_cluster_for(gid)`` and
        ``registry.clusters_for(gid)`` return the physical IDs stored here,
        so ``CollectiveLoweringPass`` can read from the registry directly
        instead of holding a reference to this ``HardwareBinding``.
        """
        for gid, ids in self.cluster_ids.items():
            try:
                group = registry.get(gid)
                object.__setattr__(group, "physical_cluster_ids", list(ids))
            except KeyError:
                pass


class CollectiveBackend(ABC):
    """Abstract base class for hardware-specific collective lowering."""

    @abstractmethod
    def lower(
        self,
        spec: "CollectiveOpSpec",
        binding: "HardwareBinding",
        registry: "ClusterGroupRegistry",
    ) -> List["TileBinding"]:
        ...


class SoftHierCollectiveBackend(CollectiveBackend):
    """SoftHier collective backend — dispatches to CollectiveStrategies."""

    def lower(
        self,
        spec: "CollectiveOpSpec",
        binding: HardwareBinding,
        registry: "ClusterGroupRegistry",
    ) -> List["TileBinding"]:
        from Deeploy.TileIR.Backend.CollectiveStrategies import STRATEGIES

        group = registry.get(spec.group_id)
        topo = binding.topology

        for strategy_cls in STRATEGIES:
            strategy = strategy_cls()
            if strategy.matches(spec, group, topo):
                return strategy.emit(spec, binding, registry)

        raise NotImplementedError(
            f"SoftHierCollectiveBackend: no strategy matched for "
            f"op={spec.op!r}, group={spec.group_id!r}."
        )


def _reduce_op_to_c(reduce_op: str) -> str:
    """Map a reduce_op string to the SoftHier C enum constant."""
    mapping_op = {"sum": "ADD_", "max": "MAX_"}
    if reduce_op not in mapping_op:
        raise NotImplementedError(f"Unsupported reduce_op={reduce_op!r}.")
    return "COLLECTIVE_RED" + mapping_op[reduce_op] + "FP_16"
