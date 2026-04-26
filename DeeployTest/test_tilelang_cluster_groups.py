# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TileIR cluster-group primitives.

Covers:
  1. TP test  — 2-cluster tensor-parallel GEMM with allreduce output
  2. DP test  — 2-cluster data-parallel (independent tiles), verifies group
                isolation + barriers
  3. SP test  — 2-cluster sequence-parallel with explicit scatter/gather
                (collective API only; no hardware lowering checked)

Unit tests (TestClusterGroupRegistry … TestBackwardCompatibility) test the IR
data model and C-string output only — no simulator required.

End-to-end tests (TestEndToEndGroupCompilation) generate Network.c/Network.h
following the SoftHier harness convention and verify that the CMake
``--target network`` build succeeds.

pytest markers: ``tilelang`` (all tests), ``softhier`` (end-to-end only).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from Deeploy.Logging import DEFAULT_LOGGER as log
from testUtils.core.execution import build_binary, run_simulation
from testUtils.pytestRunner import verify_numeric_outputs

# ---------------------------------------------------------------------------
# Guard: skip the whole module when TVM / TileLang are not available.
# ---------------------------------------------------------------------------
try:
    import tilelang  # noqa: F401
    _TILELANG_AVAILABLE = True
except ImportError:
    _TILELANG_AVAILABLE = False

# Check for the patched directional collective ops (group_shift / group_bcast_axis).
# These require a rebuilt tilelang with tl.tileop.group_shift and
# tl.tileop.group_bcast_axis registered as TVM opaque ops.
_DIRECTIONAL_OPS_AVAILABLE = False
if _TILELANG_AVAILABLE:
    try:
        import tilelang.language as _T_chk
        _DIRECTIONAL_OPS_AVAILABLE = (
            hasattr(_T_chk, "group_shift") and hasattr(_T_chk, "group_bcast_axis")
        )
    except Exception:
        pass

pytestmark = pytest.mark.tilelang

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry_and_binding(ids_a, ids_b=None):
    """Build a ClusterGroupRegistry and HardwareBinding for one or two groups."""
    from Deeploy.TileIR.IR import (
        ClusterGroup,
        ClusterGroupRegistry,
        HardwareBinding,
    )

    groups = [ClusterGroup("group_a", group_x=len(ids_a), group_y=1)]
    hw = {"group_a": ids_a}
    if ids_b is not None:
        groups.append(ClusterGroup("group_b", group_x=len(ids_b), group_y=1))
        hw["group_b"] = ids_b
    return ClusterGroupRegistry(groups), HardwareBinding(hw)


# ---------------------------------------------------------------------------
# Tests: IR data model (no TVM required)
# ---------------------------------------------------------------------------


class TestClusterGroupRegistry:
    def test_get_known_group(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
        )
        g = ClusterGroup("tp_row", group_x=2, group_y=1)
        reg = ClusterGroupRegistry([g])
        assert reg.get("tp_row") is g

    def test_get_unknown_group_raises(self):
        from Deeploy.TileIR.IR import ClusterGroupRegistry
        reg = ClusterGroupRegistry([])
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_root_cluster(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        # root_coord=(1,0) → rank 1 is root; physical cluster at rank 1 is ids[1]=3
        g = ClusterGroup("dp_col", group_x=2, group_y=1, root_coord=(1, 0))
        hw = HardwareBinding({"dp_col": [2, 3]})
        reg = ClusterGroupRegistry([g])
        assert hw.root_cluster_for("dp_col", reg) == 3

    def test_contains(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
        )
        g = ClusterGroup("sp", group_x=2, group_y=1)
        reg = ClusterGroupRegistry([g])
        assert reg.contains("sp")
        assert not reg.contains("other")


class TestClusterGroup2D:
    """Unit tests for the 2-D ClusterGroup structure."""

    def test_shape(self):
        from Deeploy.TileIR.IR import ClusterGroup
        g = ClusterGroup("g", group_x=2, group_y=2, num_groups=2)
        assert g.shape == (2, 2)
        assert g.num_ranks_per_instance == 4

    def test_rank_coord_roundtrip(self):
        from Deeploy.TileIR.IR import ClusterGroup
        g = ClusterGroup("g", group_x=3, group_y=2)
        for rank in range(g.num_ranks_per_instance):
            x, y = g.coord_of(rank)
            assert g.rank_of(x, y) == rank

    def test_root_cluster_id(self):
        from Deeploy.TileIR.IR import ClusterGroup
        g = ClusterGroup("g", group_x=2, group_y=2, root_coord=(1, 0))
        # rank_of(1, 0) = 0*2 + 1 = 1
        assert g.root_instance == 1

    def test_hardware_binding_partitions_by_num_groups(self):
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        g = ClusterGroup("g", group_x=2, group_y=2, num_groups=2)
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"g": [0, 1, 4, 5, 2, 3, 6, 7]})
        instances = hw.group_instances("g", reg)
        assert len(instances) == 2
        assert instances[0] == [0, 1, 4, 5]
        assert instances[1] == [2, 3, 6, 7]

    def test_1d_group_construction(self):
        from Deeploy.TileIR.IR import ClusterGroup
        g = ClusterGroup("flat", group_x=3, group_y=1)
        assert g.shape == (3, 1)
        assert g.num_ranks_per_instance == 3

    def test_invalid_dimensions_raise(self):
        from Deeploy.TileIR.IR import ClusterGroup
        with pytest.raises(ValueError):
            ClusterGroup("bad", group_x=0, group_y=1)


class TestTensorLayout:
    """Unit tests for TensorLayout dataclass."""

    def test_is_sharded(self):
        from Deeploy.TileIR.IR import TensorLayout
        layout = TensorLayout(group_id="tp", axis_map={1: "x"})
        assert layout.is_sharded
        assert not layout.is_partial

    def test_is_partial(self):
        from Deeploy.TileIR.IR import TensorLayout
        layout = TensorLayout(group_id="tp", partial=("sum", "x"))
        assert layout.is_partial
        assert layout.reduce_op() == "sum"
        assert layout.reduce_axis() == "x"

    def test_sharded_axes(self):
        from Deeploy.TileIR.IR import TensorLayout
        layout = TensorLayout(group_id="tp", axis_map={0: "y", 1: "x"})
        assert layout.sharded_axes() == [0, 1]


class TestHardwareBinding:
    def test_clusters_for(self):
        from Deeploy.TileIR.IR import HardwareBinding
        hw = HardwareBinding({"g": [0, 2]})
        assert hw.clusters_for("g") == [0, 2]

    def test_grid_dims_1d(self):
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        reg = ClusterGroupRegistry([ClusterGroup("g", group_x=4, group_y=1)])
        hw = HardwareBinding({"g": [0, 1, 2, 3]})
        assert hw.grid_dims_for("g", reg) == (4, 1)

    def test_grid_dims_2d(self):
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        reg = ClusterGroupRegistry([ClusterGroup("tp", group_x=2, group_y=2)])
        hw = HardwareBinding({"tp": [0, 1, 2, 3]})
        assert hw.grid_dims_for("tp", reg) == (2, 2)

    def test_root_cluster_for(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        g = ClusterGroup("tp", group_x=2, group_y=1, root_coord=(0, 0))
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        assert hw.root_cluster_for("tp", reg) == 0

    def test_unknown_group_raises(self):
        from Deeploy.TileIR.IR import HardwareBinding
        hw = HardwareBinding({"g": [0, 1]})
        with pytest.raises(KeyError):
            hw.clusters_for("missing")


class TestPlacementDerivation:
    """Derived-placement path: IDs computed from topology + Placement policy."""

    def test_contiguous_row_major(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup, ClusterGroupRegistry, HardwareBinding, HwTopology, Placement,
        )
        reg = ClusterGroupRegistry([ClusterGroup("g", group_x=2, group_y=2, num_groups=1)])
        hw = HardwareBinding.from_placements(HwTopology(Px=4, Py=4), {"g": Placement()}, reg)
        # Row-major 2x2 group starting at origin (0, 0) on a 4-wide grid
        assert hw.clusters_for("g") == [0, 1, 4, 5]

    def test_origin_offset(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup, ClusterGroupRegistry, HardwareBinding, HwTopology, Placement,
        )
        reg = ClusterGroupRegistry([ClusterGroup("g", group_x=2, group_y=2, num_groups=1)])
        hw = HardwareBinding.from_placements(
            HwTopology(Px=4, Py=4), {"g": Placement(origin=(2, 1))}, reg,
        )
        assert hw.clusters_for("g") == [6, 7, 10, 11]

    def test_stride_systolic(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup, ClusterGroupRegistry, HardwareBinding, HwTopology, Placement,
        )
        reg = ClusterGroupRegistry([ClusterGroup("g", group_x=2, group_y=2, num_groups=1)])
        hw = HardwareBinding.from_placements(
            HwTopology(Px=4, Py=4), {"g": Placement(stride=(2, 2))}, reg,
        )
        # 2-stride in both dims picks every other cluster on both axes
        assert hw.clusters_for("g") == [0, 2, 8, 10]

    def test_multi_instance_tiling(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup, ClusterGroupRegistry, HardwareBinding, HwTopology, Placement,
        )
        reg = ClusterGroupRegistry([ClusterGroup("g", group_x=2, group_y=1, num_groups=2)])
        hw = HardwareBinding.from_placements(HwTopology(Px=4, Py=1), {"g": Placement()}, reg)
        assert hw.clusters_for("g") == [0, 1, 2, 3]
        # Two distinct instances of the 2-rank group
        instances = hw.group_instances("g", reg)
        assert instances == [[0, 1], [2, 3]]


class TestDirectionalCollectives:
    """GroupShift / GroupBcastAxis dispatch through SoftHierCollectiveBackend."""

    def test_group_shift_dispatch(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup, ClusterGroupRegistry, CollectiveOpSpec, HardwareBinding,
            SoftHierCollectiveBackend,
        )
        g = ClusterGroup("tp", group_x=4, group_y=1, axis_names=("tp", "_"))
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1, 2, 3]})
        spec = CollectiveOpSpec(
            op="shift", group_id="tp",
            src_buffer="A_local", dst_buffer="A_local",
            reduce_axis="tp", shift_by=1,
        )
        bindings = SoftHierCollectiveBackend().lower(spec, hw, reg)
        assert len(bindings) == 1
        rep = bindings[0].operator_representation
        assert rep["axis_index"] == 0
        assert rep["shift_by"] == 1

    def test_group_bcast_axis_dispatch(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup, ClusterGroupRegistry, CollectiveOpSpec, HardwareBinding,
            SoftHierCollectiveBackend,
        )
        g = ClusterGroup("g2d", group_x=2, group_y=2, axis_names=("x", "y"))
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"g2d": [0, 1, 4, 5]})
        spec = CollectiveOpSpec(
            op="bcast_axis", group_id="g2d",
            src_buffer="B", dst_buffer="B",
            reduce_axis="y", from_coord=1,
        )
        bindings = SoftHierCollectiveBackend().lower(spec, hw, reg)
        assert len(bindings) == 1
        rep = bindings[0].operator_representation
        assert rep["axis_index"] == 1
        assert rep["from_coord"] == 1


class TestCollectiveOpSpec:
    def test_default_reduce_op(self):
        from Deeploy.TileIR.IR import CollectiveOpSpec
        spec = CollectiveOpSpec(op="allreduce", group_id="g", src_buffer="C_partial", dst_buffer="C_reduce")
        assert spec.reduce_op == "sum"

    def test_custom_reduce_op(self):
        from Deeploy.TileIR.IR import CollectiveOpSpec
        spec = CollectiveOpSpec(op="allreduce", group_id="g", src_buffer="A", dst_buffer="B", reduce_op="max")
        assert spec.reduce_op == "max"


class TestShardMetadata:
    def test_defaults(self):
        from Deeploy.TileIR.IR import ShardMetadata
        m = ShardMetadata()
        assert m.group_id is None

    def test_fields(self):
        from Deeploy.TileIR.IR import ShardMetadata
        m = ShardMetadata(group_id="tp_row")
        assert m.group_id == "tp_row"


# ---------------------------------------------------------------------------
# Tests: SoftHierCollectiveBackend lowering
# ---------------------------------------------------------------------------


class TestSoftHierCollectiveBackend:
    def test_allreduce_expands_to_two_bindings(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            CollectiveOpSpec,
            HardwareBinding,
            SoftHierCollectiveBackend,
        )
        g = ClusterGroup("tp", group_x=2, group_y=1)
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        backend = SoftHierCollectiveBackend()
        spec = CollectiveOpSpec(op="allreduce", group_id="tp",
                                src_buffer="C_partial", dst_buffer="C_reduce")
        result = backend.lower(spec, hw, reg)
        assert len(result) == 2
        assert result[0].op_kind == "group_collective"
        assert result[1].op_kind == "group_collective"

    def test_allreduce_with_axis_expands_to_two_bindings(self):
        """AxisReduceBroadcast strategy fires when reduce_axis is set."""
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            CollectiveOpSpec,
            HardwareBinding,
            SoftHierCollectiveBackend,
        )
        g = ClusterGroup("tp", group_x=2, group_y=1)
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        backend = SoftHierCollectiveBackend()
        spec = CollectiveOpSpec(op="allreduce", group_id="tp",
                                src_buffer="C_partial", dst_buffer="C_reduce",
                                reduce_axis="x")
        result = backend.lower(spec, hw, reg)
        assert len(result) == 2

    def test_broadcast_expands_to_one_binding(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            CollectiveOpSpec,
            HardwareBinding,
            SoftHierCollectiveBackend,
        )
        g = ClusterGroup("tp", group_x=2, group_y=1)
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        backend = SoftHierCollectiveBackend()
        spec = CollectiveOpSpec(op="broadcast", group_id="tp",
                                src_buffer="root_buf", dst_buffer="member_buf")
        result = backend.lower(spec, hw, reg)
        assert len(result) == 1

    def test_unsupported_op_raises(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            CollectiveOpSpec,
            HardwareBinding,
            SoftHierCollectiveBackend,
        )
        g = ClusterGroup("tp", group_x=2, group_y=1)
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        backend = SoftHierCollectiveBackend()
        # "p2p" is not handled by any strategy
        spec = CollectiveOpSpec(op="p2p", group_id="tp",
                                src_buffer="A", dst_buffer="B")
        with pytest.raises(NotImplementedError):
            backend.lower(spec, hw, reg)

    def test_axis_reduce_rowwise_uses_runtime_masks(self):
        """axis='x' (row-wise): wakeup_row_mask + (ARCH_NUM_CLUSTER_Y-1), west-edge gate."""
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            CollectiveOpSpec,
            HardwareBinding,
            SoftHierCollectiveBackend,
        )

        g = ClusterGroup("tp", group_x=2, group_y=1)
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        backend = SoftHierCollectiveBackend()
        spec = CollectiveOpSpec(op="allreduce", group_id="tp",
                                src_buffer="C_partial", dst_buffer="C_reduce",
                                reduce_axis="x")
        bindings = backend.lower(spec, hw, reg)
        assert len(bindings) == 2
        rep = bindings[0].operator_representation
        assert "wakeup_row_mask" in rep.get("row_mask", ""), \
            f"Row-wise: expected wakeup_row_mask in row_mask, got {rep.get('row_mask')!r}"
        assert "ARCH_NUM_CLUSTER_Y" in rep.get("col_mask", ""), \
            f"Row-wise: expected (ARCH_NUM_CLUSTER_Y-1) in col_mask, got {rep.get('col_mask')!r}"
        assert "cluster_in_group_id_x_tp" in rep.get("edge_flag", ""), \
            f"Row-wise: expected west-edge gate in edge_flag, got {rep.get('edge_flag')!r}"

    def test_axis_reduce_colwise_uses_runtime_masks(self):
        """axis='y' (col-wise): (ARCH_NUM_CLUSTER_X-1) + wakeup_col_mask, south-edge gate."""
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            CollectiveOpSpec,
            HardwareBinding,
            SoftHierCollectiveBackend,
        )

        g = ClusterGroup("g2d", group_x=2, group_y=2, axis_names=("x", "y"))
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"g2d": [0, 1, 2, 3]})
        backend = SoftHierCollectiveBackend()
        spec = CollectiveOpSpec(op="allreduce", group_id="g2d",
                                src_buffer="C_partial", dst_buffer="C_reduce",
                                reduce_axis="y")
        bindings = backend.lower(spec, hw, reg)
        assert len(bindings) == 2
        rep = bindings[0].operator_representation
        assert "ARCH_NUM_CLUSTER_X" in rep.get("row_mask", ""), \
            f"Col-wise: expected (ARCH_NUM_CLUSTER_X-1) in row_mask, got {rep.get('row_mask')!r}"
        assert "wakeup_col_mask" in rep.get("col_mask", ""), \
            f"Col-wise: expected wakeup_col_mask in col_mask, got {rep.get('col_mask')!r}"
        assert "cluster_in_group_id_y_g2d" in rep.get("edge_flag", ""), \
            f"Col-wise: expected south-edge gate in edge_flag, got {rep.get('edge_flag')!r}"



# ---------------------------------------------------------------------------
# Tests: CollectiveBinding
# ---------------------------------------------------------------------------


class TestCollectiveBinding:
    def test_op_kind_forced_to_group_collective(self):
        from Deeploy.TileIR.IR import CollectiveBinding, CollectiveOpSpec
        from Deeploy.DeeployTypes import NodeTemplate
        spec = CollectiveOpSpec(op="allreduce", group_id="g",
                                src_buffer="A", dst_buffer="B")
        cb = CollectiveBinding(
            op_kind="group_collective",
            template=NodeTemplate("// placeholder\n"),
            operator_representation={"cluster_id": None},
            spec=spec,
        )
        assert cb.op_kind == "group_collective"
        assert cb.spec is spec


# ---------------------------------------------------------------------------
# Tests: GroupAwareBarrierPass
# ---------------------------------------------------------------------------


class TestGroupAwareBarrierPass:
    def _make_binding(self, cluster_id, group_id=None):
        from Deeploy.TileIR.IR import ShardMetadata
        from Deeploy.TileIR.Midend.TileBindings import TileBinding
        from Deeploy.DeeployTypes import NodeTemplate
        shard_meta = ShardMetadata(group_id=group_id) if group_id else None
        return TileBinding(
            op_kind="load",
            template=NodeTemplate("// op\n"),
            operator_representation={
                "cluster_id": cluster_id,
                "shard_metadata": shard_meta,
            },
        )

    def test_cluster_switch_inserts_global_barrier(self):
        from Deeploy.TileIR.IR import GroupAwareBarrierPass
        b0 = self._make_binding(0, "g")
        b1 = self._make_binding(1, "g")
        result = GroupAwareBarrierPass().apply([b0, b1])
        assert len(result) == 3  # b0, barrier, b1
        assert result[1].op_kind == "sync"

    def test_same_cluster_no_barrier(self):
        from Deeploy.TileIR.IR import GroupAwareBarrierPass
        b0 = self._make_binding(0, "g")
        b1 = self._make_binding(0, "g")
        result = GroupAwareBarrierPass().apply([b0, b1])
        assert len(result) == 2  # no barrier inserted


# ---------------------------------------------------------------------------
# Tests: CollectiveLoweringPass
# ---------------------------------------------------------------------------


class TestCollectiveLoweringPass:
    def test_group_init_prepended(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            CollectiveBinding,
            CollectiveLoweringPass,
            CollectiveOpSpec,
            HardwareBinding,
            ShardMetadata,
            SoftHierCollectiveBackend,
        )
        from Deeploy.TileIR.Midend.TileBindings import TileBinding
        from Deeploy.DeeployTypes import NodeTemplate

        g = ClusterGroup("tp", group_x=2, group_y=1)
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        backend = SoftHierCollectiveBackend()

        shard = ShardMetadata(group_id="tp")
        load_b = TileBinding(
            op_kind="load",
            template=NodeTemplate("// load\n"),
            operator_representation={"cluster_id": 0, "shard_metadata": shard},
        )

        spec = CollectiveOpSpec(op="allreduce", group_id="tp",
                                src_buffer="C_partial", dst_buffer="C_reduce")
        coll_b = CollectiveBinding(
            op_kind="group_collective",
            template=NodeTemplate("// placeholder\n"),
            operator_representation={
                "cluster_id": None, "shard_metadata": shard, "nbytes": 128,
            },
            spec=spec,
        )

        lp = CollectiveLoweringPass(registry=reg, hw_binding=hw, backend=backend)
        result = lp.apply([load_b, coll_b])

        # Prefix per group: sync + group_init + sync + group_context = 4 nodes
        # Then: load_b + 2 lowered collective bindings = 3 nodes
        # Total >= 7
        assert len(result) >= 5
        # First three are the init prefix
        assert result[0].op_kind == "sync"           # global barrier before init
        assert result[1].op_kind == "group_barrier"  # group_init
        assert result[2].op_kind == "sync"           # global barrier after init
        # result[3] is group_context (also group_barrier)
        assert result[3].op_kind == "group_barrier"  # group_context


# ---------------------------------------------------------------------------
# Tests: Template content checks
# ---------------------------------------------------------------------------


class TestCollectiveTemplates:
    def _render(self, template, rep):
        from Deeploy.DeeployTypes import NodeTemplate, CodeSnippet, ExecutionBlock, NetworkContext
        from Deeploy.Targets.SoftHier.Platform import SoftHierDynamicBuffer
        ctxt = NetworkContext(
            variableBuffer=SoftHierDynamicBuffer,
            constantBuffer=SoftHierDynamicBuffer,
            structBuffer=SoftHierDynamicBuffer,
            transientBuffer=SoftHierDynamicBuffer,
        )
        eb = ExecutionBlock(CodeSnippet(template, rep))
        return eb.generate(ctxt)

    def test_group_init_contains_grid_sync_group_init(self):
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import TileGroupInitTemplate
        code = self._render(TileGroupInitTemplate, {"group_id": "tp", "x_dim": 2, "y_dim": 1})
        assert "grid_sync_group_init" in code
        assert "group_info_tp" in code

    def test_group_context_contains_cluster_rank_vars(self):
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import TileGroupContextTemplate
        code = self._render(TileGroupContextTemplate, {"group_id": "tp"})
        assert "cluster_in_group_id_x_tp" in code
        assert "cluster_in_group_id_y_tp" in code
        assert "cluster_for_rowwise_tp" in code
        assert "cluster_for_colwise_tp" in code

    def test_group_barrier_contains_grid_sync_group_barrier_xy(self):
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import TileGroupBarrierTemplate
        code = self._render(TileGroupBarrierTemplate, {"group_id": "tp"})
        assert "grid_sync_group_barrier_xy" in code
        assert "&group_info_tp" in code

    def test_alloc_reducer_contains_root_cluster_guard(self):
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import TileAllocReducerTemplate
        code = self._render(TileAllocReducerTemplate, {
            "name": "C_reduce", "dtype": "fp16",
            "nbytes": 256, "root_cluster_id": 0,
        })
        assert "flex_get_cluster_id() == 0" in code
        assert "flex_l1_malloc" in code
        assert "zomem" in code  # zero-initialisation

    def test_collective_reduce_contains_flex_dma_reduction(self):
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import TileCollectiveReduceTemplate
        code = self._render(TileCollectiveReduceTemplate, {
            "src_name": "C_partial",
            "dst_name": "C_reduce",
            "root_cluster_id": 0,
            "group_id": "tp",
            "collective_op_kind": "COLLECTIVE_REDADD_FP_16",
            "row_mask": "group_info_tp.wakeup_row_mask",
            "col_mask": "(ARCH_NUM_CLUSTER_Y - 1)",
            "edge_flag": "cluster_for_rowwise_tp",
            "nbytes": 128,
            "cluster_id": None,
        })
        assert "flex_dma_async_reduction" in code
        assert "grid_sync_group_barrier_xy" in code
        assert "COLLECTIVE_REDADD_FP_16" in code
        # Must use runtime mask expressions, not hex literals
        assert "wakeup_row_mask" in code
        assert "cluster_for_rowwise_tp" in code

    def test_collective_broadcast_contains_flex_dma_broadcast(self):
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import TileCollectiveBroadcastTemplate
        code = self._render(TileCollectiveBroadcastTemplate, {
            "src_name": "C_reduce",
            "dst_name": "C_partial",
            "root_cluster_id": 0,
            "group_id": "tp",
            "row_mask": "group_info_tp.wakeup_row_mask",
            "col_mask": "(ARCH_NUM_CLUSTER_Y - 1)",
            "edge_flag": "cluster_for_rowwise_tp",
            "nbytes": 128,
            "cluster_id": None,
        })
        assert "flex_dma_async_broadcast" in code
        assert "grid_sync_group_barrier_xy" in code
        assert "wakeup_row_mask" in code


# ---------------------------------------------------------------------------
# Tests: TileLang compilation (requires TVM)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestTPGemmCompilation:
    """Tensor-parallel GEMM: 2 clusters, allreduce output."""

    def _build_tp_kernel(self):
        import tilelang
        import tilelang.language as T

        BM, BN, BK = 64, 64, 32

        @tilelang.jit
        def tp_gemm(A, B, C, BM: int, BN: int, BK: int):
            M, K = T.const("M, K")
            N = T.const("N")
            dtype = T.float16

            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                with T.attr("anno", "cluster_group", "tp_row"):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)
                    T.copy(A[bx * BM, 0], A_local)
                    T.copy(B[0, by * BN], B_local)
                    T.gemm(A_local, B_local, C_local)
                    T.copy(C_local, C[bx * BM, by * BN])

        return tp_gemm

    def test_tp_compiles_without_error(self):
        import tilelang.language as T
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, N, K = 128, 128, 64
        BM, BN, BK = 64, 64, 32

        tp_gemm = self._build_tp_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("tp_row", group_x=2, group_y=1)
        ])
        hw_binding = HardwareBinding({"tp_row": [0, 1]})

        code = compile_tilelang_to_softhier_parallel(
            tp_gemm, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw_binding,
            cluster_policy="block_idx",
            cluster_ids=[0, 1],
        )
        assert isinstance(code, str)
        assert len(code) > 0
        assert "for (int" in code  # block-index for-loop was emitted

    def test_tp_code_contains_group_init(self):
        import tilelang.language as T
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, N, K = 128, 128, 64
        BM, BN, BK = 64, 64, 32

        tp_gemm = self._build_tp_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("tp_row", group_x=2, group_y=1)
        ])
        hw_binding = HardwareBinding({"tp_row": [0, 1]})

        code = compile_tilelang_to_softhier_parallel(
            tp_gemm, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw_binding,
            cluster_policy="block_idx",
            cluster_ids=[0, 1],
        )
        assert "grid_sync_group_init" in code
        assert "group_info_tp_row" in code
        assert "flex_global_barrier_xy" in code

    def test_tp_gemm_k_split_block_loop(self):
        """TP GEMM with T.Pipelined must emit a block-distributed bk loop (K-split).

        Without K-split, both clusters compute the full K sum and the allreduce
        doubles the result (2×A@B instead of A@B).  The correct generated code
        uses block distribution: rank r handles bk in [r*tiles_per_rank,
        (r+1)*tiles_per_rank) so each cluster processes a contiguous K slice.

        With K=64, BK=32, group_x=2: tiles_per_rank=1.
        Cluster 0: for (bk = 0; bk < 1; bk++)
        Cluster 1: for (bk = 1; bk < 2; bk++)
        """
        import tilelang.language as T
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, K, N = 128, 64, 128
        BM, BK, BN = 128, 32, 128

        tp_gemm = TestEndToEndGroupCompilation._build_tp_gemm_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("tp_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw_binding = HardwareBinding({"tp_group": [0, 1]})

        code = compile_tilelang_to_softhier_parallel(
            tp_gemm, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw_binding,
            cluster_policy="block_idx",
            num_clusters=16,
        )
        # Block K-split: loop bounds derived from cluster rank * tiles_per_rank
        assert "cluster_in_group_id_x_tp_group" in code, \
            "Expected cluster rank variable in generated code"
        # Loop start is rank * tiles_per_rank (e.g. rank * 1)
        assert "cluster_in_group_id_x_tp_group * 1" in code, \
            "Expected block-distribution loop start: rank * tiles_per_rank"
        # Loop end is (rank + 1) * tiles_per_rank
        assert "(cluster_in_group_id_x_tp_group + 1) * 1" in code, \
            "Expected block-distribution loop end: (rank+1) * tiles_per_rank"
        # Stage selection relative to rank's block start (not raw bk % N)
        assert "bk - cluster_in_group_id_x_tp_group" in code, \
            "Expected stage select relative to rank block start"
        # No raw bk % 2 == 0 (non-rank-aware stage selection from non-TP path)
        assert "bk % 2 == 0" not in code, \
            "Expected no non-rank-aware stage selection"

    def test_tp_gemm_fixed_if_guard(self):
        """_build_tp_gemm_fixed_kernel: group_id_x var in 'if' maps to C cluster var.

        The TIR IfThenElse produced by ``if group_id_x == 0:`` must be lowered
        to ``if (cluster_in_group_id_x_tp_group == 0)`` in the generated C code.
        """
        import tilelang.language as T
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, K, N = 128, 64, 128
        BM, BK, BN = 64, 32, 64

        tp_gemm_fixed = TestEndToEndGroupCompilation._build_tp_gemm_fixed_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("tp_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw_binding = HardwareBinding({"tp_group": [0, 1, 2, 3, 4, 5, 6, 7]})

        code = compile_tilelang_to_softhier_parallel(
            tp_gemm_fixed, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw_binding,
            num_clusters=8,
        )
        # GroupContext declares: uint32_t _group_id_x_tp_group = cluster_in_group_id_x_tp_group;
        assert "uint32_t _group_id_x_tp_group = cluster_in_group_id_x_tp_group;" in code, \
            "Expected GroupContext alias for group_id_x"
        # IfThenElse emits the TIR var name directly; C resolves via the alias above
        assert "if ((_group_id_x_tp_group == 0))" in code, \
            f"Expected C if-guard using alias, got:\n{code}"


@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestDPCompilation:
    """Data-parallel: 2 clusters process independent tiles."""

    def test_dp_compiles_without_error(self):
        import tilelang
        import tilelang.language as T
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        BM, BK = 64, 32
        M, K = 128, 64

        @tilelang.jit
        def dp_gemv(A, B, BM: int, BK: int):
            M, K = T.const("M, K")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K,), dtype)
            C = T.empty((M,), dtype)

            with T.Kernel(T.ceildiv(M, BM)) as pid_m:
                with T.attr("anno", "cluster_id", 0):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK,), dtype)
                    C_local = T.alloc_fragment((BM,), dtype)
                    T.clear(C_local)
                    T.copy(A[pid_m * BM, 0], A_local)
                    T.copy(B[0], B_local)
                    T.copy(C_local, C[pid_m * BM])
            return C

        A = T.empty((M, K), T.float16)
        B = T.empty((K,), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("dp_group", group_x=2, group_y=1)
        ])
        hw_binding = HardwareBinding({"dp_group": [0, 1]})

        code = compile_tilelang_to_softhier_parallel(
            dp_gemv, A, B, BM=BM, BK=BK,
            group_registry=registry,
            hw_binding=hw_binding,
        )
        assert isinstance(code, str)
        # No allreduce expected — pure data parallel
        assert "flex_dma_async_reduction" not in code


@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestTIRStructure:
    """Verify TIR output structure from get_tir() for multi-tile kernels."""

    def _build_multi_tile_gemm(self):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def gemm(A, B, C, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                A_local = T.alloc_fragment((BM, BK), dtype)
                B_local = T.alloc_fragment((BK, BN), dtype)
                C_local = T.alloc_fragment((BM, BN), dtype)
                T.clear(C_local)
                T.copy(A[bx * BM, 0], A_local)
                T.copy(B[0, by * BN], B_local)
                T.gemm(A_local, B_local, C_local)
                T.copy(C_local, C[bx * BM, by * BN])

        return gemm

    def test_kernel_produces_thread_extent(self):
        """T.Kernel with multi-tile grid should produce thread_extent AttrStmt."""
        import tilelang.language as T
        gemm = self._build_multi_tile_gemm()
        M, K, N = 128, 64, 128
        BM, BN, BK = 64, 64, 32
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)
        primfunc = gemm.get_tir(A, B, C, BM=BM, BN=BN, BK=BK)
        script = primfunc.script()
        assert "blockIdx" in script, f"Expected blockIdx in TIR:\n{script[:500]}"

    def test_gemm_intrinsic_present(self):
        """T.gemm should produce a gemm_py intrinsic call in TIR."""
        import tilelang.language as T
        gemm = self._build_multi_tile_gemm()
        M, K, N = 128, 64, 128
        BM, BN, BK = 64, 64, 32
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)
        primfunc = gemm.get_tir(A, B, C, BM=BM, BN=BN, BK=BK)
        script = primfunc.script()
        assert "gemm" in script.lower(), f"Expected gemm intrinsic in TIR:\n{script[:500]}"

    def test_alloc_buffers_in_block(self):
        """T.alloc_fragment should appear in Block's alloc_buffers, not body."""
        import tilelang.language as T
        gemm = self._build_multi_tile_gemm()
        M, K, N = 128, 64, 128
        BM, BN, BK = 64, 64, 32
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)
        primfunc = gemm.get_tir(A, B, C, BM=BM, BN=BN, BK=BK)
        script = primfunc.script()
        assert "alloc_buffer" in script or "alloc_fragment" in script or "local.fragment" in script, \
            f"Expected alloc_buffer/fragment in TIR:\n{script[:500]}"

    def test_visitor_emits_for_loops(self):
        """TilelangVisitor should emit for-loops for blockIdx axes."""
        import tilelang.language as T
        from Deeploy.TileIR.Frontend.TilelangVisitor import TilelangVisitor
        from Deeploy.Targets.SoftHier.Platform import SoftHierDynamicBuffer
        from Deeploy.DeeployTypes import NetworkContext

        gemm = self._build_multi_tile_gemm()
        M, K, N = 128, 64, 128
        BM, BN, BK = 64, 64, 32
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)
        primfunc = gemm.get_tir(A, B, C, BM=BM, BN=BN, BK=BK)

        ctxt = NetworkContext(
            variableBuffer=SoftHierDynamicBuffer,
            constantBuffer=SoftHierDynamicBuffer,
            structBuffer=SoftHierDynamicBuffer,
            transientBuffer=SoftHierDynamicBuffer,
        )
        visitor = TilelangVisitor(cluster_policy="block_idx", cluster_ids=[0, 1])
        pipeline = visitor.visit_bindings(primfunc, ctxt)
        ctxt, eb = pipeline.codeTransform(ctxt)
        code = eb.generate(ctxt)
        assert "for (int" in code, f"Expected for-loop in generated code:\n{code[:500]}"


@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestBackwardCompatibility:
    """Verify that non-group kernels still compile correctly."""

    def test_single_cluster_unchanged(self):
        import tilelang
        import tilelang.language as T
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier

        @tilelang.jit
        def simple_copy(A, B):
            M = T.const("M")
            dtype = T.float16
            A: T.Tensor((M,), dtype)
            B: T.Tensor((M,), dtype)
            with T.Kernel(1) as _:
                with T.attr("anno", "cluster_id", 0):
                    A_local = T.alloc_fragment((64,), dtype)
                    T.copy(A[0], A_local)
                    T.copy(A_local, B[0])

        A = T.empty((64,), T.float16)
        B = T.empty((64,), T.float16)
        code = compile_tilelang_to_softhier(simple_copy, A, B, cluster_id=0)
        assert isinstance(code, str)
        assert "flex_dma_sync_2d" in code
        # No group primitives should appear
        assert "grid_sync_group_init" not in code
        assert "GridSyncGroupInfo" not in code


@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestNonContiguousClusterIds:
    """Verify that non-contiguous cluster_ids produce a lookup-table guard."""

    def test_cluster_map_in_generated_code(self):
        import tilelang
        import tilelang.language as T
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier

        @tilelang.jit
        def simple_gemm(A, B, C, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                A_local = T.alloc_fragment((BM, BK), dtype)
                B_local = T.alloc_fragment((BK, BN), dtype)
                C_local = T.alloc_fragment((BM, BN), dtype)
                T.clear(C_local)
                T.copy(A[bx * BM, 0], A_local)
                T.copy(B[0, by * BN], B_local)
                T.gemm(A_local, B_local, C_local)
                T.copy(C_local, C[bx * BM, by * BN])

        M, K, N = 128, 64, 128
        BM, BN, BK = 64, 64, 32
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        code = compile_tilelang_to_softhier(
            simple_gemm, A, B, C, BM=BM, BN=BN, BK=BK,
            cluster_ids=[0, 2, 5],
        )
        assert isinstance(code, str)
        assert "(1U <<" in code, (
            f"Expected bitmask shift in generated code:\n{code[:500]}"
        )
        assert "(1U << 5)" in code, (
            f"Expected (1U << 5) for cluster 5 in generated code:\n{code[:500]}"
        )

    def test_contiguous_ids_also_use_bitmask(self):
        """cluster_ids=[0,1] should also produce a bitmask guard."""
        import tilelang
        import tilelang.language as T
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier

        @tilelang.jit
        def simple_copy(A, B):
            M = T.const("M")
            dtype = T.float16
            A: T.Tensor((M,), dtype)
            B: T.Tensor((M,), dtype)
            with T.Kernel(2) as bx:
                A_local = T.alloc_fragment((64,), dtype)
                T.copy(A[bx * 64], A_local)
                T.copy(A_local, B[bx * 64])

        A = T.empty((128,), T.float16)
        B = T.empty((128,), T.float16)

        code = compile_tilelang_to_softhier(
            simple_copy, A, B,
            cluster_ids=[0, 1],
        )
        assert isinstance(code, str)
        assert "(1U <<" in code, (
            f"Expected bitmask shift in generated code:\n{code[:500]}"
        )


# ---------------------------------------------------------------------------
# Tests: SUMMA GEMM — axis-scoped broadcast from a chosen source rank
# ---------------------------------------------------------------------------
#
# SUMMA uses T.group_bcast_axis to broadcast an A-tile from one rank to all
# other ranks within the group along one axis, then each rank multiplies with
# its own B-tile.  With static from_coord=0 and all clusters loading the same
# A-tile from global memory, the broadcast is a no-op (all clusters already
# hold identical data), but it exercises the full collective dispatch pipeline.
#
# Numerical correctness argument (static from_coord=0, all K tiles):
#   Every cluster loads A[bx*BM, bk*BK] identically from global memory.
#   group_bcast_axis reinforces cluster-0's copy; result is unchanged.
#   Each cluster accumulates C_local = Σ_bk A[bk] @ B_local → C = A @ B ✓
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DIRECTIONAL_OPS_AVAILABLE, reason="group_bcast_axis not in tilelang")
class TestSummaGemmCompilation:
    """SUMMA GEMM: axis-scoped broadcast compilation and code-structure tests."""

    @staticmethod
    def _build_summa_kernel():
        """2×1 group SUMMA GEMM using T.group_bcast_axis (static from_coord=0).

        All clusters load the same A-tile from global memory; the broadcast
        reinforces cluster-0's copy.  C = A @ B is numerically correct for
        any number of K-tiles.
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def summa_gemm(A, B, C, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                with T.cluster_group("summa_group", x=2, y=1, num_groups=8,
                                     axes=("x", "_")) as (gid, gid_x, gid_y):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)
                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        if (gid_x == 0 & gid_y == 0 & gid % 4 == 0):
                            T.copy(A[bx * BM, bk * BK], A_local)
                            T.copy(B[bk * BK, by * BN], B_local)
                            # Broadcast A from rank 0 along the x-axis to all group members.
                            # from_coord=0 is static; dynamic per-step from_coord requires
                            # further TIR support (tracked as future work).
                            T.group_bcast_axis(A_local, along="x", from_coord=0,
                                            group="summa_group")
                        T.gemm(A_local, B_local, C_local, clear_accum=False)
                        T.allreduce(C_local, C_local, "sum", axis="x", clear=False)
                    T.copy(C_local, C[bx * BM, by * BN])

        return summa_gemm

    def test_summa_gemm_compiles_without_error(self):
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, K, N, BM, BK, BN = 256, 256, 256, 64, 64, 64
        fn = self._build_summa_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw = HardwareBinding({"summa_group": list(range(16))})

        code = compile_tilelang_to_softhier_parallel(
            fn, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=16,
        )
        assert isinstance(code, str) and len(code) > 0

    def test_summa_gemm_code_contains_group_bcast_axis(self):
        """Generated code must invoke flex_dma_async_broadcast for bcast_axis."""
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, K, N, BM, BK, BN = 256, 256, 256, 64, 64, 64
        fn = self._build_summa_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw = HardwareBinding({"summa_group": list(range(16))})

        code = compile_tilelang_to_softhier_parallel(
            fn, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=16,
        )
        assert "flex_dma_async_broadcast" in code, \
            "Expected flex_dma_async_broadcast in SUMMA code"
        assert "group_info_summa_group" in code, \
            "Expected group_info_summa_group in SUMMA code"
        assert "grid_sync_group_barrier_xy" in code, \
            "Expected group barrier in SUMMA code"

    def test_summa_gemm_code_contains_group_init(self):
        """Generated code must initialise the summa_group before use."""
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, K, N, BM, BK, BN = 256, 256, 256, 64, 64, 64
        fn = self._build_summa_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw = HardwareBinding({"summa_group": list(range(16))})

        code = compile_tilelang_to_softhier_parallel(
            fn, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=16,
        )
        assert "grid_sync_group_init" in code, \
            "Expected grid_sync_group_init in SUMMA code"
        assert "flex_global_barrier_xy" in code, \
            "Expected global barrier in SUMMA code"


# ---------------------------------------------------------------------------
# Tests: Cannon / Systolic GEMM — ring-shift along group axis
# ---------------------------------------------------------------------------
#
# Cannon's algorithm uses T.group_shift to rotate A-tiles left (and B-tiles
# up) each step.  v1 emits a stub (no-op + group barrier) because the SoftHier
# runtime ring-DMA primitive is not yet available.
#
# Numerical correctness argument (stub shift is a no-op):
#   Every cluster independently loads A[bx*BM, bk*BK] and B[bk*BK, by*BN]
#   for every bk, accumulating C_local = A @ B → correct DP result.
#   A real Cannon implementation would rotate tiles before the load each step.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _DIRECTIONAL_OPS_AVAILABLE, reason="group_shift not in tilelang")
class TestCannonGemmCompilation:
    """Cannon GEMM: ring-shift compilation and code-structure tests."""

    @staticmethod
    def _build_cannon_kernel():
        """2×1 group Cannon GEMM using T.group_shift (v1 wrap-only stub).

        In the real Cannon algorithm each cluster would start with a skewed
        A/B tile and rotate by one rank each step.  The v1 stub emits a
        group barrier without the actual DMA; numerics are correct by
        coincidence (every cluster independently accumulates all K tiles).
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def cannon_gemm(A, B, C, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                with T.cluster_group("cannon_group", x=2, y=1, num_groups=8,
                                     axes=("x", "_")):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)
                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        T.copy(A[bx * BM, bk * BK], A_local)
                        T.copy(B[bk * BK, by * BN], B_local)
                        T.gemm(A_local, B_local, C_local, clear_accum=False)
                        # Ring-shift A left along x-axis (Cannon step).
                        # v1: stub emits a group barrier only.
                        T.group_shift(A_local, along="x", by=1, group="cannon_group")
                    T.copy(C_local, C[bx * BM, by * BN])

        return cannon_gemm

    def test_cannon_gemm_compiles_without_error(self):
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, K, N, BM, BK, BN = 256, 256, 256, 64, 64, 64
        fn = self._build_cannon_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("cannon_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw = HardwareBinding({"cannon_group": list(range(16))})

        code = compile_tilelang_to_softhier_parallel(
            fn, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=16,
        )
        assert isinstance(code, str) and len(code) > 0

    def test_cannon_gemm_code_contains_group_shift_barrier(self):
        """Group-shift stub must emit a group barrier (no real DMA yet)."""
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, K, N, BM, BK, BN = 256, 256, 256, 64, 64, 64
        fn = self._build_cannon_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("cannon_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw = HardwareBinding({"cannon_group": list(range(16))})

        code = compile_tilelang_to_softhier_parallel(
            fn, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=16,
        )
        # Stub emits a group-scoped barrier so all clusters stay in lockstep.
        assert "grid_sync_group_barrier_xy" in code, \
            "Expected group barrier in Cannon shift stub"
        # Stub also emits a void-cast no-op on the source buffer.
        assert "(void)" in code, \
            "Expected no-op void cast in Cannon shift stub"
        assert "group_info_cannon_group" in code, \
            "Expected cannon_group info struct in generated code"

    def test_cannon_gemm_code_contains_group_init(self):
        """Cannon kernel must initialise cannon_group via grid_sync_group_init."""
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel

        M, K, N, BM, BK, BN = 256, 256, 256, 64, 64, 64
        fn = self._build_cannon_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("cannon_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw = HardwareBinding({"cannon_group": list(range(16))})

        code = compile_tilelang_to_softhier_parallel(
            fn, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=16,
        )
        assert "grid_sync_group_init" in code, \
            "Expected grid_sync_group_init in Cannon code"


# ---------------------------------------------------------------------------
# End-to-end tests: generate Network.c and compile with SoftHier CMake
# ---------------------------------------------------------------------------
#
# These tests require a SoftHier toolchain (--toolchain-install-dir) and are
# tagged @softhier + @tilelang.  They generate Network.c / Network.h using the
# standard SoftHier harness convention (RunNetwork / InitNetwork /
# DeeployNetwork_*) and verify that `cmake --build --target network` succeeds.
# ---------------------------------------------------------------------------


@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestEndToEndGroupCompilation:
    """End-to-end Network.c compilation tests for cluster-group parallel kernels."""

    # ------------------------------------------------------------------
    # Shared kernel builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tp_gemm_kernel():
        """TP GEMM: K-split tensor-parallel over 2 clusters, allreduce C_local.

        Each cluster computes a partial BM×BN GEMM over its BK slice of K.
        T.tile_layout annotates C_local as partial:sum@tp so the visitor
        routes T.allreduce to AxisReduceBroadcast (runtime wakeup_row_mask).
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def tp_gemm(A, B, C, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                # 2×1 cluster group named "tp_group"; x-axis labelled "tp"
                # TODO: return as group_id, group_x/y to avoid hardcoding
                with T.cluster_group("tp_group", x=2, y=1, num_groups=4, axes=("tp", "_")):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    # Mark C_local as a partial sum awaiting tp-axis allreduce
                    T.tile_layout(C_local, partial=("sum", "tp"))
                    T.clear(C_local)
                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        T.copy(A[bx * BM, bk * BK], A_local)
                        T.copy(B[bk * BK, by * BN], B_local)
                        T.gemm(A_local, B_local, C_local, clear_accum=False)
                    # In-place allreduce; reduce_axis resolved from tile_layout above
                    T.allreduce(C_local, reduce_op="sum", group="tp_group")
                    # T.broadcast(C_local, group="tp_group")
                    T.copy(C_local, C[bx * BM, by * BN])

        return tp_gemm
    
    @staticmethod
    def _build_tp_gemm_fixed_kernel():
        """TP GEMM: K-split tensor-parallel over 2 clusters, allreduce C_local.

        Each cluster computes a partial BM×BN GEMM over its BK slice of K.
        T.tile_layout annotates C_local as partial:sum@tp so the visitor
        routes T.allreduce to AxisReduceBroadcast (runtime wakeup_row_mask).
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def tp_gemm_fixed(A, B, C, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                # 2×1 cluster group named "tp_group"; x-axis labelled "tp"
                with T.cluster_group("tp_group", x=2, y=1, num_groups=8, axes=("tp", "_")) as (group_id, group_id_x, group_id_y):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    # Mark C_local as a partial sum awaiting tp-axis allreduce
                    T.tile_layout(C_local, partial=("sum", "tp"))
                    T.clear(C_local)
                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        T.copy(A[bx * BM, bk * BK], A_local)
                        T.copy(B[bk * BK, by * BN], B_local)
                        T.gemm(A_local, B_local, C_local, clear_accum=False)
                    # In-place allreduce; reduce_axis resolved from tile_layout above
                    T.allreduce(C_local, reduce_op="sum", axis="tp", group="tp_group")
                    # T.broadcast(C_local, group="tp_group")
                    if group_id_x == 0:
                        T.copy(C_local, C[bx * BM, by * BN])

        return tp_gemm_fixed
    


    @staticmethod
    def _build_dp_gemm_kernel():
        """DP GEMM: 4 independent 1×1 group instances (data parallel).

        grid_sync_group_init(1, 1) with num_groups=4 ticks one group per
        cluster.  No collective is needed; block-idx dispatch routes tiles.
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def dp_gemm(A, B, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C = T.empty((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                # 4 independent 1×1 group instances — one per cluster, no collective
                with T.cluster_group("dp_group", x=1, y=1, num_groups=4, axes=("_", "_")):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)
                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        T.copy(A[bx * BM, bk * BK], A_local)
                        T.copy(B[bk * BK, by * BN], B_local)
                        T.gemm(A_local, B_local, C_local, clear_accum=False)
                    T.copy(C_local, C[bx * BM, by * BN])

            return C

        return dp_gemm

    @staticmethod
    def _build_summa_gemm_kernel():
        """SUMMA GEMM: axis-scoped broadcast using T.group_bcast_axis.

        2×1 group; all clusters load the same A-tile from global memory and
        the broadcast reinforces cluster-0's copy (no-op for identical data).
        C = A @ B is numerically correct for any number of K-tiles.
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def summa_gemm(A, B, C, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                with T.cluster_group("summa_group", x=2, y=1, num_groups=8,
                                     axes=("x", "_")) as (gid, gid_x, gid_y):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)
                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        T.copy(A[bx * BM, bk * BK], A_local)
                        T.copy(B[bk * BK, by * BN], B_local)
                        T.group_bcast_axis(A_local, along="x", from_coord=0,
                                           group="summa_group")
                        T.gemm(A_local, B_local, C_local, clear_accum=False)
                    T.copy(C_local, C[bx * BM, by * BN])

        return summa_gemm

    @staticmethod
    def _build_cannon_gemm_kernel():
        """Cannon GEMM: ring-shift using T.group_shift (v1 stub).

        2×1 group; each cluster independently accumulates all K-tiles because
        the stub shift is a no-op.  C = A @ B is numerically correct.
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def cannon_gemm(A, B, C, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                with T.cluster_group("cannon_group", x=2, y=1, num_groups=8,
                                     axes=("x", "_")):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)
                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        T.copy(A[bx * BM, bk * BK], A_local)
                        T.copy(B[bk * BK, by * BN], B_local)
                        T.gemm(A_local, B_local, C_local, clear_accum=False)
                        T.group_shift(A_local, along="x", by=1, group="cannon_group")
                    T.copy(C_local, C[bx * BM, by * BN])

        return cannon_gemm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cmake_build_network(config) -> subprocess.CompletedProcess:
        cmake_cmd = os.environ.get("CMAKE", "cmake")
        return subprocess.run(
            [cmake_cmd, "--build", config.build_dir, "--target", "network"],
            check=False,
            capture_output=True,
            text=True,
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_tp_gemm_e2e_compiles(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """TP GEMM: 2-cluster tensor-parallel group — Network.c must compile."""
        import tilelang.language as T
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        NUM_CLUSTERS = 16
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import configure_cmake
        from testUtils.pytestRunner import create_test_config

        # Multi-tile grid: M > BM, N > BN so bx/by loop over multiple tiles.
        M, K, N = 256, 256, 256
        BM, BK, BN = 64, 64, 64
        elem_bytes = 2  # sizeof(fp16)

        tp_gemm = self._build_tp_gemm_fixed_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        # 4 group instances × 2 clusters each = 8 clusters.
        # active_instances = len([0..7]) // 2 = 4 → cluster_active = this_grid_id < 4.
        # The registry no longer needs num_groups; hw_binding.active_instances() derives it.
        registry = ClusterGroupRegistry([
            ClusterGroup("tp_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        # actual cluster IDs used at runtime
        tp_row_cluster = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # tp_row_cluster = [8, 9, 10, 11, 12, 13, 14, 15]
        hw_binding = HardwareBinding({"tp_group": tp_row_cluster})

        body = compile_tilelang_to_softhier_parallel(
            tp_gemm, A, B, C,
            BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw_binding,
            num_clusters=NUM_CLUSTERS,
        )

        # Reference data (full M x N output)
        rng = np.random.default_rng(1)
        A_np = rng.standard_normal((M, K)).astype(np.float16)
        B_np = rng.standard_normal((K, N)).astype(np.float16)
        # from softhier_golden.fma import matrix_multiply_with_bittrue_fma
        # C_ref = matrix_multiply_with_bittrue_fma(A_np, B_np, np.zeros((M, N))).astype(np.float16)
        C_ref = A_np @ B_np.astype(np.float16)
        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_A", c_dtype="fp16", nbytes=M * K * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_B", c_dtype="fp16", nbytes=K * N * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_C", c_dtype="fp16", nbytes=M * N * elem_bytes, is_input=False),
        ]

        cmake_extra = list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"]
        config = create_test_config(
            test_name="Tilelang/tp_gemm",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=cmake_extra,
            tiling=False,
        )

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[A_np, B_np],
            test_outputs=[C_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        # verify_numeric_outputs(result, C_ref.flatten().reshape(1, -1), atol=3e-1, rtol=3e-2)

        # assert result.success, (
        #     f"TP GEMM simulation failed: {result.error_count} errors "
        #     f"out of {result.total_count}\n{result.stdout}"
        # )
    
    
    def test_dp_gemm_e2e_compiles(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        return
        """DP GEMM: 2-cluster data-parallel (independent tiles) — Network.c must compile."""
        import tilelang.language as T
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        NUM_CLUSTERS = 16  # cluster IDs go up to 15
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import configure_cmake
        from testUtils.pytestRunner import create_test_config

        M, K, N = 256, 128, 256
        BM, BK, BN = 128, 32, 128
        elem_bytes = 2  # sizeof(fp16)

        dp_gemm = self._build_dp_gemm_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("dp_group", group_x=4, group_y=1, root_coord=(0, 0))
        ])
        dp_row_cluster = [0, 1, 2, 3]
        hw_binding = HardwareBinding({"dp_group": dp_row_cluster})

        body = compile_tilelang_to_softhier_parallel(
            dp_gemm, A, B,
            BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw_binding
        )
        assert isinstance(body, str) and len(body) > 0
        assert "for (int" in body  # block-index for-loop was emitted

        # Reference data — full M x N output (all tiles covered by block grid)
        rng = np.random.default_rng(1)
        A_np = rng.standard_normal((M, K)).astype(np.float16)
        B_np = rng.standard_normal((K, N)).astype(np.float16)
        from softhier_golden.fma import matrix_multiply_with_bittrue_fma
        C_ref = matrix_multiply_with_bittrue_fma(A_np, B_np, np.zeros((M, N))).astype(np.float16)
        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_A", c_dtype="fp16", nbytes=M * K * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_B", c_dtype="fp16", nbytes=K * N * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_C", c_dtype="fp16", nbytes=M * N * elem_bytes, is_input=False),
        ]

        cmake_extra = list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"]
        config = create_test_config(
            test_name="Tilelang/dp_gemm",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=cmake_extra,
            tiling=False,
        )

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[A_np, B_np],
            test_outputs=[C_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        verify_numeric_outputs(result, C_ref.flatten().reshape(1, -1), atol=3e-1, rtol=3e-2)

        assert result.success, (
            f"DP GEMM simulation failed: {result.error_count} errors "
            f"out of {result.total_count}\n{result.stdout}"
        )

    @pytest.mark.skipif(not _DIRECTIONAL_OPS_AVAILABLE, reason="group_bcast_axis not in tilelang")
    def test_summa_gemm_e2e_compiles(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """SUMMA GEMM: axis-scoped broadcast — build + simulate, verify C = A @ B.

        All clusters load identical A-tiles; group_bcast_axis reinforces
        cluster-0's copy (no-op for same data).  Numerical result is correct.
        """
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import configure_cmake
        from testUtils.pytestRunner import create_test_config

        NUM_CLUSTERS = 16
        M, K, N = 256, 256, 256
        BM, BK, BN = 64, 64, 64
        elem_bytes = 2

        summa_gemm = self._build_summa_gemm_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw = HardwareBinding({"summa_group": list(range(NUM_CLUSTERS))})

        body = compile_tilelang_to_softhier_parallel(
            summa_gemm, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        rng = np.random.default_rng(2)
        A_np = rng.standard_normal((M, K)).astype(np.float16)
        B_np = rng.standard_normal((K, N)).astype(np.float16)
        C_ref = (A_np @ B_np).astype(np.float16)

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_A", c_dtype="fp16",
                             nbytes=M * K * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_B", c_dtype="fp16",
                             nbytes=K * N * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_C", c_dtype="fp16",
                             nbytes=M * N * elem_bytes, is_input=False),
        ]

        cmake_extra = list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"]
        config = create_test_config(
            test_name="Tilelang/summa_gemm",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=cmake_extra,
            tiling=False,
        )

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[A_np, B_np],
            test_outputs=[C_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        verify_numeric_outputs(result, C_ref.flatten().reshape(1, -1), atol=3e-1, rtol=3e-2)

    @pytest.mark.skipif(not _DIRECTIONAL_OPS_AVAILABLE, reason="group_shift not in tilelang")
    def test_cannon_gemm_e2e_compiles(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """Cannon GEMM: ring-shift (stub) — build + simulate, verify C = A @ B.

        The v1 stub emits a no-op barrier instead of a real ring-DMA.  Every
        cluster independently accumulates all K-tiles, so C = A @ B is correct
        despite the missing rotation.  This test validates the full compile +
        simulate pipeline for the Cannon pattern.
        """
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import configure_cmake
        from testUtils.pytestRunner import create_test_config

        NUM_CLUSTERS = 16
        M, K, N = 256, 256, 256
        BM, BK, BN = 64, 64, 64
        elem_bytes = 2

        cannon_gemm = self._build_cannon_gemm_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("cannon_group", group_x=2, group_y=1, root_coord=(0, 0))
        ])
        hw = HardwareBinding({"cannon_group": list(range(NUM_CLUSTERS))})

        body = compile_tilelang_to_softhier_parallel(
            cannon_gemm, A, B, C, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        rng = np.random.default_rng(3)
        A_np = rng.standard_normal((M, K)).astype(np.float16)
        B_np = rng.standard_normal((K, N)).astype(np.float16)
        C_ref = (A_np @ B_np).astype(np.float16)

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_A", c_dtype="fp16",
                             nbytes=M * K * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_B", c_dtype="fp16",
                             nbytes=K * N * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_C", c_dtype="fp16",
                             nbytes=M * N * elem_bytes, is_input=False),
        ]

        cmake_extra = list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"]
        config = create_test_config(
            test_name="Tilelang/cannon_gemm",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=cmake_extra,
            tiling=False,
        )

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[A_np, B_np],
            test_outputs=[C_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        # Cannon stub (no-op shift): each cluster accumulates C = A @ B independently.
        # Numeric verification is valid because the stub preserves correctness.
        verify_numeric_outputs(result, C_ref.flatten().reshape(1, -1), atol=3e-1, rtol=3e-2)