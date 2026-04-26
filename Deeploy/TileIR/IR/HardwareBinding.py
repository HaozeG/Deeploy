# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR hardware binding for collective operations.

``HardwareBinding`` maps logical cluster-group names to physical cluster
IDs on a ``Px × Py`` SoC grid.  The mapping is described by a
``Placement`` policy per group; the flat integer list (``cluster_ids``)
is *derived* from ``(topology, group.shape, group.num_groups, placement)``.
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


@dataclass(frozen=True)
class Placement:
    """Policy for deriving physical cluster IDs for one group.

    Either supply ``explicit_ids`` (flat override) OR a derivation rule
    ``(origin, tile_order, stride)``.  When ``explicit_ids`` is set, the
    derivation fields are ignored.

    Parameters
    ----------
    origin : (int, int)
        ``(x0, y0)`` position of instance-0 on the ``Px × Py`` grid.
    tile_order : str
        How the ``num_groups`` instances are tiled across the grid:
        ``"row_major"`` (default) or ``"col_major"``.
    stride : (int, int)
        Per-rank stride inside one instance.  ``(1, 1)`` is contiguous
        row-major; non-unit values express Systolic-style strided
        placements.
    explicit_ids : list[int] or None
        Flat, pre-computed cluster ID list covering every rank of every
        instance (length = ``group_x * group_y * num_groups``).
    """

    origin: Tuple[int, int] = (0, 0)
    tile_order: str = "row_major"
    stride: Tuple[int, int] = (1, 1)
    explicit_ids: Optional[Tuple[int, ...]] = None


def _derive_cluster_ids(
    topology: HwTopology,
    group_x: int,
    group_y: int,
    num_groups: int,
    placement: Placement,
) -> List[int]:
    """Compute the flat list of physical cluster IDs for one group."""
    if placement.explicit_ids is not None:
        return list(placement.explicit_ids)

    sx, sy = placement.stride
    ox, oy = placement.origin

    # Instances tile across the physical grid.  Default: how many full
    # (group_x*sx) tiles fit horizontally.
    tiles_x = max(topology.Px // (group_x * sx), 1)

    ids: List[int] = []
    for g in range(num_groups):
        if placement.tile_order == "col_major":
            tx, ty = g // tiles_x, g % tiles_x
        else:
            tx, ty = g % tiles_x, g // tiles_x
        base_x = ox + tx * group_x * sx
        base_y = oy + ty * group_y * sy
        for ry in range(group_y):
            for rx in range(group_x):
                x = base_x + rx * sx
                y = base_y + ry * sy
                ids.append(topology.cluster_of(x, y))
    return ids


@dataclass
class HardwareBinding:
    """Maps logical cluster-group names to physical cluster IDs.

    Two construction modes:

    * **Explicit** — ``HardwareBinding({"g": [0, 1, 2, 3]})``: backward-
      compatible adapter; the list is stored as a ``Placement(explicit_ids=...)``.
    * **Derived** — ``HardwareBinding.from_placements(topology, {"g": Placement(...)})``:
      IDs are computed from topology and the group's 2-D shape at lookup time.
    """

    cluster_ids: Dict[str, List[int]] = field(default_factory=dict)
    topology: Optional[HwTopology] = None
    placements: Dict[str, Placement] = field(default_factory=dict)

    def __post_init__(self):
        # Adapter: explicit dict positional arg → synthesize Placements.
        for gid, ids in self.cluster_ids.items():
            self.placements.setdefault(gid, Placement(explicit_ids=tuple(ids)))

    # -- alternate constructors ---------------------------------------------

    @classmethod
    def from_placements(
        cls,
        topology: HwTopology,
        placements: Dict[str, Placement],
        registry: "ClusterGroupRegistry",
    ) -> "HardwareBinding":
        """Build a binding where IDs are derived from placement policies."""
        binding = cls(cluster_ids={}, topology=topology, placements=dict(placements))
        # Materialize once so clusters_for() / downstream code has a list view.
        for gid, placement in placements.items():
            group = registry.get(gid)
            binding.cluster_ids[gid] = _derive_cluster_ids(
                topology, group.group_x, group.group_y, group.num_groups, placement,
            )
        return binding

    @classmethod
    def from_flat(
        cls,
        cluster_ids: Dict[str, List[int]],
        topology: Optional[HwTopology] = None,
    ) -> "HardwareBinding":
        """Adapter for pre-existing tests: explicit ID lists."""
        return cls(cluster_ids=dict(cluster_ids), topology=topology)

    # -- queries ------------------------------------------------------------

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

    def grid_dims_for(self, group_id: str, registry: "ClusterGroupRegistry") -> Tuple[int, int]:
        """Return (x_dim, y_dim) for grid_sync_group_init.

        The group must be registered.  Group geometry is the single source
        of truth for dimensions.
        """
        group = registry.get(group_id)
        return (group.group_x, group.group_y)

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

    def compute_hw_bitmask(self, group_id: str) -> int:
        """Return a bitmask with a bit set for every physical cluster in this group."""
        bitmask = 0
        for cid in self.clusters_for(group_id):
            bitmask |= (1 << cid)
        return bitmask

    def compute_this_grid_id_table(
        self,
        group_id: str,
        registry: "ClusterGroupRegistry",
    ) -> List[int]:
        """Return this_grid_id for the lead cluster of each group instance."""
        group = registry.get(group_id)
        k = group.num_ranks_per_instance
        all_clusters = self.clusters_for(group_id)
        num_instances = self.active_instances(group_id, registry)
        instances = [all_clusters[i * k:(i + 1) * k] for i in range(num_instances)]

        if self.topology is None:
            return list(range(num_instances))

        Px = self.topology.Px
        grid_x_num = Px // group.group_x

        table: List[int] = []
        for instance in instances:
            cid = instance[0]
            x, y = self.topology.pos_of(cid)
            tgid = (y // group.group_y) * grid_x_num + (x // group.group_x)
            table.append(tgid)

        return table

    def active_instances(self, group_id: str, registry: "ClusterGroupRegistry") -> int:
        """Return the number of active group instances for *group_id*.

        Authoritative count; overrides the ``num_groups`` hint on
        ``ClusterGroup`` which may come from the DSL and differ from the
        actual deployment.
        """
        group = registry.get(group_id)
        k = group.num_ranks_per_instance
        clusters = self.clusters_for(group_id)
        return len(clusters) // k if k > 0 else 1

    def populate_registry(self, registry: "ClusterGroupRegistry") -> None:
        """Embed physical_cluster_ids into each ClusterGroup in the registry."""
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
