# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR collective operation primitives.

Defines the declarative data model for cluster groups and inter-cluster
communication.  No hardware-specific lowering lives here; that belongs in
``HardwareBinding.py`` and the backend templates.

Classes
-------
ClusterGroup
    A strict 2-D group of clusters matching SoftHier's
    ``grid_sync_group_init(group_x, group_y)`` primitive.  One call
    creates ``num_groups`` parallel instances tiled across the chip.

ClusterGroupRegistry
    Container of ClusterGroup objects; lookup by group_id.

TensorLayout
    Sharding annotation for a tile buffer relative to a ClusterGroup.

CollectiveOpSpec
    Math-level specification of an inter-cluster collective operation.

ShardMetadata
    Lightweight per-binding metadata tag for tile-reuse analysis passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ClusterGroup:
    """A strict 2-D group of clusters matching SoftHier's grid_sync_group_init.

    One ``grid_sync_group_init(group_x, group_y)`` call produces
    ``num_groups`` parallel group instances tiled across the physical
    ``Px × Py`` cluster grid.  Runtime fields ``this_grid_id``,
    ``wakeup_row_mask`` / ``wakeup_col_mask`` differentiate instances and
    identify which clusters participate in row/column collectives.

    Parameters
    ----------
    group_id : str
        Unique identifier.  Matches the string in ``T.attr("anno",
        "cluster_group", group_id)`` annotations.
    group_x : int
        Group width  (``grid_x_dim`` in ``GridSyncGroupInfo``).
    group_y : int
        Group height (``grid_y_dim`` in ``GridSyncGroupInfo``).
    num_groups : int
        Number of group instances tiled across the physical grid.
    axis_names : (str, str)
        Names for the two group axes, default ``("x", "y")``.
        Used in ``TensorLayout.axis_map`` values.
    root_coord : (int, int)
        ``(rx, ry)`` root rank within one group instance, default ``(0, 0)``.
    """

    group_id: str
    group_x: int
    group_y: int
    num_groups: int = 1
    axis_names: Tuple[str, str] = ("x", "y")
    root_coord: Tuple[int, int] = (0, 0)
    physical_cluster_ids: Optional[List[int]] = None

    def __post_init__(self):
        if self.group_x < 1 or self.group_y < 1:
            raise ValueError(
                f"ClusterGroup '{self.group_id}': group_x and group_y must be >= 1, "
                f"got group_x={self.group_x}, group_y={self.group_y}"
            )
        if self.num_groups < 1:
            raise ValueError(
                f"ClusterGroup '{self.group_id}': num_groups must be >= 1, "
                f"got num_groups={self.num_groups}"
            )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.group_x, self.group_y)

    @property
    def num_ranks_per_instance(self) -> int:
        return self.group_x * self.group_y

    @property
    def total_clusters(self) -> int:
        return self.num_groups * self.group_x * self.group_y

    def axis_index(self, name: str) -> int:
        if name == self.axis_names[0]:
            return 0
        if name == self.axis_names[1]:
            return 1
        raise ValueError(
            f"ClusterGroup '{self.group_id}': axis '{name}' not found in "
            f"axis_names={self.axis_names!r}"
        )

    def rank_of(self, rx: int, ry: int) -> int:
        return ry * self.group_x + rx

    def coord_of(self, rank: int) -> Tuple[int, int]:
        return (rank % self.group_x, rank // self.group_x)

    @property
    def root_cluster_id(self) -> int:
        """Row-major rank of the root within one instance."""
        return self.rank_of(self.root_coord[0], self.root_coord[1])

    @property
    def root_physical_cluster_id(self) -> int:
        """Physical SoC cluster ID of the root within instance 0."""
        if self.physical_cluster_ids is not None:
            return self.physical_cluster_ids[self.root_cluster_id]
        return self.root_cluster_id  # logical rank as fallback

    @property
    def num_clusters(self) -> int:
        """Alias for num_ranks_per_instance (legacy compat)."""
        return self.num_ranks_per_instance

    @property
    def root_instance(self) -> int:
        """Row-major rank of root within one instance."""
        return self.rank_of(self.root_coord[0], self.root_coord[1])


@dataclass
class ClusterGroupRegistry:
    """Container for a collection of ClusterGroup objects."""

    groups: List[ClusterGroup] = field(default_factory=list)
    neighbor_groups: Dict[str, List[str]] = field(default_factory=dict)

    def get(self, group_id: str) -> ClusterGroup:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(f"ClusterGroupRegistry: unknown group_id={group_id!r}")

    def register(self, group: ClusterGroup) -> None:
        """Register a new group. Silently replaces if group_id already exists."""
        for i, g in enumerate(self.groups):
            if g.group_id == group.group_id:
                self.groups[i] = group
                return
        self.groups.append(group)

    def root_cluster(self, group_id: str) -> int:
        return self.get(group_id).root_cluster_id

    def root_cluster_for(self, group_id: str) -> int:
        """Physical root cluster ID for *group_id* (uses physical_cluster_ids if set)."""
        return self.get(group_id).root_physical_cluster_id

    def clusters_for(self, group_id: str) -> Optional[List[int]]:
        """Physical cluster IDs for *group_id*, or None if not set."""
        return self.get(group_id).physical_cluster_ids

    def all_group_ids(self) -> List[str]:
        return [g.group_id for g in self.groups]

    def contains(self, group_id: str) -> bool:
        for group in self.groups:
            if group.group_id == group_id:
                return True
        return False


@dataclass(frozen=True)
class TensorLayout:
    """How a tile buffer is sharded across the two axes of a ClusterGroup.

    Parameters
    ----------
    group_id : str
        Which cluster group this layout belongs to.
    axis_map : Dict[int, str]
        Maps tensor axis indices to group axis names.
    partial : (str, str) or None
        ``(reduce_op, axis_name)`` — marks this buffer as holding a
        partial result awaiting reduction along ``axis_name``.
    """

    group_id: str
    axis_map: Dict[int, str] = field(default_factory=dict)
    partial: Optional[Tuple[str, str]] = None

    @property
    def is_sharded(self) -> bool:
        return bool(self.axis_map)

    @property
    def is_partial(self) -> bool:
        return self.partial is not None

    def sharded_axes(self) -> List[int]:
        return sorted(self.axis_map.keys())

    def reduce_axis(self) -> Optional[str]:
        if self.partial is not None:
            return self.partial[1]
        return None

    def reduce_op(self) -> Optional[str]:
        if self.partial is not None:
            return self.partial[0]
        return None


@dataclass
class CollectiveOpSpec:
    """Math-level specification of an inter-cluster collective.

    Parameters
    ----------
    op : str
        Collective operation: ``"allreduce"``, ``"broadcast"``,
        ``"scatter"``, or ``"gather"``.
    group_id : str
        The cluster group over which the collective is performed.
    src_buffer : str
        Name of the source (partial-result) L1 buffer.
    dst_buffer : str
        Name of the destination (accumulated-result) L1 buffer.
    reduce_op : str
        Reduction operator for ``"allreduce"``: ``"sum"``, ``"max"``.
    reduce_axis : str or None
        Group axis name along which to reduce (``"x"`` or ``"y"``).
        When ``None``, the backend uses full-group allreduce.
    src_layout : TensorLayout or None
        Layout annotation of the source buffer.
    dst_layout : TensorLayout or None
        Layout annotation of the destination buffer.
    """

    op: str
    group_id: str
    src_buffer: str
    dst_buffer: str
    reduce_op: str = "sum"
    reduce_axis: Optional[str] = None
    src_layout: Optional[TensorLayout] = None
    dst_layout: Optional[TensorLayout] = None
    # Extra parameters for directional / point-to-point ops:
    #   ``shift_by``      — ring rotation step (ops: "shift")
    #   ``from_coord``    — source rank along an axis (ops: "bcast_axis"), static int
    #   ``from_coord_expr`` — C expression for source rank when dynamic (e.g. "gid_y")
    #   ``global_barrier_before`` — emit flex_global_barrier_xy() before the collective
    shift_by: int = 0
    from_coord: int = 0
    from_coord_expr: Optional[str] = None
    global_barrier_before: bool = False


@dataclass
class ShardMetadata:
    """Lightweight per-TileBinding metadata for tile-reuse analysis."""

    group_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Spec-string parsers (grammar owned here, matching cluster_group.py)
# ---------------------------------------------------------------------------

def parse_cluster_group_spec(spec: str, registry: "ClusterGroupRegistry") -> str:
    """Parse a cluster_group AttrStmt value and register the group.

    Requires the full spec form ``"name;x=<X>;y=<Y>;num_groups=<N>;
    axes=<a0>,<a1>;root=<rx>,<ry>"``.  ``x`` and ``y`` are mandatory —
    for a 1-D group use ``y=1``.  ``num_groups``, ``axes``, ``root`` are
    optional.
    """
    spec = spec.strip('"').strip("'").strip()
    parts = spec.split(";")
    group_id = parts[0]

    kv: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()

    # Bare-name form: group must already be in the registry (pre-registered
    # by the test harness or a prior pass).  No kv params → nothing to update.
    if not kv:
        if registry.contains(group_id):
            return group_id
        raise ValueError(
            f"parse_cluster_group_spec: bare spec {spec!r} refers to a group "
            f"that is not pre-registered.  Use T.cluster_group(x=N, y=1, ...) "
            f"to declare it inline."
        )

    if "x" not in kv or "y" not in kv:
        raise ValueError(
            f"parse_cluster_group_spec: spec {spec!r} missing required 'x' "
            f"and/or 'y'.  Use T.cluster_group(x=N, y=1, ...) for 1-D groups."
        )

    group_x = int(kv["x"])
    group_y = int(kv["y"])
    num_groups = int(kv.get("num_groups", 1))

    axes_raw = kv.get("axes", "x,y").split(",")
    axis_names: Tuple[str, str] = (
        axes_raw[0].strip() if len(axes_raw) > 0 else "x",
        axes_raw[1].strip() if len(axes_raw) > 1 else "y",
    )

    root_coord: Tuple[int, int] = (0, 0)
    if "root" in kv:
        rc = kv["root"].split(",")
        root_coord = (int(rc[0]), int(rc[1]) if len(rc) > 1 else 0)

    # Always register/update from the DSL spec so group shape (x, y, axis_names)
    # reflects the kernel annotation.  The actual num_active_instances used for
    # cluster_active is derived from HardwareBinding.active_instances() at
    # CollectiveLoweringPass time, so keeping num_groups here as a fallback is safe.
    group = ClusterGroup(
        group_id=group_id,
        group_x=group_x,
        group_y=group_y,
        num_groups=num_groups,
        axis_names=axis_names,
        root_coord=root_coord,
    )
    registry.register(group)

    return group_id


def parse_layout_spec(spec: str, group_id: str):
    """Parse a layout AttrStmt value into a (buf_name, TensorLayout) tuple.

    Grammar::

        "<buf_name>=<clause>[;<clause>...]"
        clause := axis<N>:<axis_name>  |  partial:<op>@<axis_name>
    """
    spec = spec.strip('"').strip("'").strip()
    if "=" not in spec:
        raise ValueError(f"parse_layout_spec: expected '<name>=<clauses>', got {spec!r}")

    buf_name, clauses_str = spec.split("=", 1)
    buf_name = buf_name.strip()

    axis_map: Dict[int, str] = {}
    partial: Optional[Tuple[str, str]] = None

    for clause in clauses_str.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        if clause.startswith("axis") and ":" in clause:
            ax_part, axis_name = clause.split(":", 1)
            ax_idx = int(ax_part[4:])
            axis_map[ax_idx] = axis_name.strip()
        elif clause.startswith("partial:") and "@" in clause:
            rest = clause[len("partial:"):]
            op_str, axis_name = rest.split("@", 1)
            partial = (op_str.strip(), axis_name.strip())
        else:
            raise ValueError(f"parse_layout_spec: unrecognised clause {clause!r} in {spec!r}")

    return buf_name, TensorLayout(group_id=group_id, axis_map=axis_map, partial=partial)
