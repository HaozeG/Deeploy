# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""CollectiveLoweringPass — lower CollectiveBindings to hardware TileBindings."""

from __future__ import annotations

import dataclasses as _dc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Set

from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
    TileGroupContextTemplate,
    TileGroupInitTemplate,
)
from Deeploy.TileIR.IR.CollectiveBinding import CollectiveBinding
from Deeploy.TileIR.IR.TileBinding import TileBinding, TileOpKind
from Deeploy.TileIR.Passes.Base import _GLOBAL_BARRIER_TEMPLATE, TileBindingPass

if TYPE_CHECKING:
    from Deeploy.TileIR.IR.CollectivePrimitives import ClusterGroupRegistry
    from Deeploy.TileIR.IR.HardwareBinding import CollectiveBackend, HardwareBinding


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
            for gid in ordered_groups:
                group = self.registry.get(gid)
                num_active = self.hw_binding.active_instances(gid, self.registry)
                hw_bitmask = self.hw_binding.compute_hw_bitmask(gid)
                prefix.append(
                    TileBinding(
                        op_kind="group_barrier",
                        template=TileGroupContextTemplate,
                        operator_representation={
                            "group_id":   gid,
                            "num_groups": num_active,
                            "hw_bitmask": hw_bitmask,
                            "split_axes":  group.split_axes,
                            "split_shape": group.split_shape,
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
                # Honor buffer-name remapping done by SoftwarePipelinePass:
                # operator_representation["src_name"] / ["dst_name"] may have been
                # renamed to the double-buffered _cur variant; propagate that into
                # the spec so strategies emit the correct C variable name.
                spec = binding.spec
                src_override = binding.operator_representation.get("src_name")
                dst_override = binding.operator_representation.get("dst_name")
                if (src_override is not None and src_override != spec.src_buffer) or \
                        (dst_override is not None and dst_override != spec.dst_buffer):
                    spec = _dc.replace(
                        spec,
                        src_buffer=src_override if src_override else spec.src_buffer,
                        dst_buffer=dst_override if dst_override else spec.dst_buffer,
                    )
                # Lower to hardware bindings
                hw_bindings = self.backend.lower(
                    spec,
                    self.hw_binding,
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

            elif binding.op_kind == "alloc" and (gid := self._group_id_of(binding)) is not None:
                # Tag group-member allocs for guard generation
                binding.operator_representation["cluster_guard_type"] = "group_membership"
                cluster_ids = self.registry.clusters_for(gid)
                binding.operator_representation["group_cluster_ids"] = cluster_ids
                transformed.append(binding)

            else:
                transformed.append(binding)

        return prefix + transformed

    @staticmethod
    def _group_id_of(binding: TileBinding) -> Optional[str]:
        """Return the shard_group_id from a binding's operator_representation, or None."""
        return binding.operator_representation.get("shard_group_id")
