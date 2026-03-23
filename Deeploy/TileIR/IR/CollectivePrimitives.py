# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR collective operation primitives.

Defines the declarative data model for cluster groups and inter-cluster
communication.  No hardware-specific lowering lives here; that belongs in
``HardwareBinding.py`` and the backend templates.

Classes
-------
ParallelismStrategy
    Enumeration of parallelism strategies (DP, TP, SP, EP, PP).

ClusterGroup
    A named, ordered set of cluster IDs that communicate together, annotated
    with a parallelism strategy label.

ClusterGroupRegistry
    Container of ClusterGroup objects; lookup by group_id.

CollectiveOpSpec
    Math-level specification of an inter-cluster collective operation.

ShardMetadata
    Lightweight per-binding metadata tag for tile-reuse analysis passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ParallelismStrategy(Enum):
    """High-level parallelism strategy label.

    These labels carry semantic intent for downstream analysis passes
    (e.g. tile-reuse detection).  They do NOT alter code generation on
    their own — that is driven by the explicit collective primitives.
    """

    DP = "data_parallel"
    TP = "tensor_parallel"
    SP = "sequence_parallel"
    EP = "expert_parallel"
    PP = "pipeline_parallel"


@dataclass
class ClusterGroup:
    """A named set of clusters that communicate together.

    Parameters
    ----------
    group_id : str
        Unique identifier for this group.  Matches the string used in
        ``T.attr("anno", "cluster_group", group_id)`` annotations.
    cluster_ids : List[int]
        Ordered list of cluster IDs in the group.  The index of the root
        cluster is given by ``root_instance``.
    strategy : Optional[ParallelismStrategy]
        Parallelism strategy label (metadata only — no automatic inference).
        ``None`` when no strategy classification is needed.
    root_instance : int
        Index within ``cluster_ids`` of the root (coordinator) cluster.
        Defaults to 0 (first cluster in the list).
    """

    group_id: str
    cluster_ids: List[int]
    strategy: Optional[ParallelismStrategy] = None
    root_instance: int = 0

    @property
    def root_cluster_id(self) -> int:
        """Return the actual cluster ID of the root cluster."""
        return self.cluster_ids[self.root_instance]

    @property
    def num_clusters(self) -> int:
        """Number of clusters in this group."""
        return len(self.cluster_ids)


@dataclass
class ClusterGroupRegistry:
    """Container for a collection of ClusterGroup objects.

    Parameters
    ----------
    groups : List[ClusterGroup]
        All declared cluster groups.
    """

    groups: List[ClusterGroup] = field(default_factory=list)

    def get(self, group_id: str) -> ClusterGroup:
        """Look up a group by its string ID.

        Raises
        ------
        KeyError
            When no group with ``group_id`` exists.
        """
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(f"ClusterGroupRegistry: unknown group_id={group_id!r}")

    def root_cluster(self, group_id: str) -> int:
        """Return the root cluster ID for *group_id*."""
        return self.get(group_id).root_cluster_id

    def all_group_ids(self) -> List[str]:
        """Return all registered group IDs."""
        return [g.group_id for g in self.groups]

    def contains(self, group_id: str) -> bool:
        """Return True when *group_id* is registered."""
        for group in self.groups:
            if group.group_id == group_id:
                return True
        return False


@dataclass
class CollectiveOpSpec:
    """Math-level specification of an inter-cluster collective.

    Hardware lowering is performed by a ``CollectiveBackend``; this class
    carries only the logical description.

    Parameters
    ----------
    op : str
        Collective operation kind: ``"allreduce"``, ``"broadcast"``,
        ``"scatter"``, or ``"gather"``.
    group_id : str
        The cluster group over which the collective is performed.
    src_buffer : str
        Name of the source (partial-result) L1 buffer.
    dst_buffer : str
        Name of the destination (accumulated-result) L1 buffer.
    reduce_op : str
        Reduction operator for ``"allreduce"`` — ``"sum"``, ``"max"``,
        or ``"min"``.  Ignored for non-reduce collectives.
    """

    op: str
    group_id: str
    src_buffer: str
    dst_buffer: str
    reduce_op: str = "sum"


@dataclass
class ShardMetadata:
    """Lightweight per-TileBinding metadata for tile-reuse analysis.

    This dataclass is stored under the ``"shard_metadata"`` key of the
    ``operator_representation`` dict of every TileBinding that was emitted
    within a ``cluster_group`` annotation block.

    Fields
    ------
    group_id : Optional[str]
        Which cluster group this operation belongs to.
    parallelism_strategy : Optional[str]
        Strategy label string (e.g. ``"tensor_parallel"``).  Derived from
        the ``ParallelismStrategy`` enum value of the owning ClusterGroup.
    """

    group_id: Optional[str] = None
    parallelism_strategy: Optional[str] = None
