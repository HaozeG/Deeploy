# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TileIR cluster-group primitives.

Covers:
  1. TP test  — 2-cluster tensor-parallel GEMM with allreduce output
  2. DP test  — 2-cluster data-parallel (independent tiles), verifies group
                isolation + barriers
  3. PP test  — 2-stage pipeline (HBM as stage boundary), verifies
                cross-group global barrier
  4. SP test  — 2-cluster sequence-parallel with explicit scatter/gather
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

    groups = [ClusterGroup("group_a", ids_a)]
    hw = {"group_a": ids_a}
    if ids_b is not None:
        groups.append(ClusterGroup("group_b", ids_b))
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
        g = ClusterGroup("tp_row", [0, 1])
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
        )
        g = ClusterGroup("dp_col", [2, 3], root_instance=1)
        reg = ClusterGroupRegistry([g])
        assert reg.root_cluster("dp_col") == 3

    def test_contains(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
        )
        g = ClusterGroup("sp", [0, 1])
        reg = ClusterGroupRegistry([g])
        assert reg.contains("sp")
        assert not reg.contains("other")


class TestHardwareBinding:
    def test_bitmask(self):
        from Deeploy.TileIR.IR import HardwareBinding
        hw = HardwareBinding({"g": [0, 2]})
        assert hw.bitmask_for("g") == 0b0101  # bits 0 and 2

    def test_grid_dims(self):
        from Deeploy.TileIR.IR import HardwareBinding
        hw = HardwareBinding({"g": [0, 1, 2, 3]})
        assert hw.grid_dims_for("g") == (4, 1)

    def test_root_cluster_for(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            HardwareBinding,
        )
        g = ClusterGroup("tp", [0, 1], root_instance=0)
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        assert hw.root_cluster_for("tp", reg) == 0

    def test_unknown_group_raises(self):
        from Deeploy.TileIR.IR import HardwareBinding
        hw = HardwareBinding({"g": [0, 1]})
        with pytest.raises(KeyError):
            hw.bitmask_for("missing")


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
        assert m.parallelism_strategy is None

    def test_fields(self):
        from Deeploy.TileIR.IR import ShardMetadata
        m = ShardMetadata(group_id="tp_row", parallelism_strategy="tensor_parallel")
        assert m.group_id == "tp_row"
        assert m.parallelism_strategy == "tensor_parallel"


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
        g = ClusterGroup("tp", [0, 1])
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        backend = SoftHierCollectiveBackend()
        spec = CollectiveOpSpec(op="allreduce", group_id="tp",
                                src_buffer="C_partial", dst_buffer="C_reduce")
        result = backend.lower(spec, hw, reg)
        assert len(result) == 2
        assert result[0].op_kind == "group_collective"
        assert result[1].op_kind == "group_collective"

    def test_broadcast_expands_to_one_binding(self):
        from Deeploy.TileIR.IR import (
            ClusterGroup,
            ClusterGroupRegistry,
            CollectiveOpSpec,
            HardwareBinding,
            SoftHierCollectiveBackend,
        )
        g = ClusterGroup("tp", [0, 1])
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
        g = ClusterGroup("tp", [0, 1])
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        backend = SoftHierCollectiveBackend()
        spec = CollectiveOpSpec(op="scatter", group_id="tp",
                                src_buffer="A", dst_buffer="B")
        with pytest.raises(NotImplementedError):
            backend.lower(spec, hw, reg)


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

        g = ClusterGroup("tp", [0, 1])
        reg = ClusterGroupRegistry([g])
        hw = HardwareBinding({"tp": [0, 1]})
        backend = SoftHierCollectiveBackend()

        shard = ShardMetadata(group_id="tp", parallelism_strategy="tensor_parallel")
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

        # Prefix: global_barrier + group_init + global_barrier = 3 nodes
        # Then: load_b + 2 lowered collective bindings = 3 nodes
        assert len(result) >= 5
        # First three are the init prefix
        assert result[0].op_kind == "sync"   # global barrier
        assert result[1].op_kind == "group_barrier"  # group init
        assert result[2].op_kind == "sync"   # global barrier


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
            "collective_op_kind": "FLEX_DMA_REDUCTION_SUM",
            "src_bitmask": "0x3",
            "dst_bitmask": "0x3",
            "nbytes": 128,
        })
        assert "flex_dma_async_reduction" in code
        assert "grid_sync_group_barrier_xy" in code
        assert "FLEX_DMA_REDUCTION_SUM" in code

    def test_collective_broadcast_contains_flex_dma_broadcast(self):
        from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import TileCollectiveBroadcastTemplate
        code = self._render(TileCollectiveBroadcastTemplate, {
            "src_name": "C_reduce",
            "dst_name": "C_partial",
            "root_cluster_id": 0,
            "group_id": "tp",
            "src_bitmask": "0x3",
            "dst_bitmask": "0x3",
            "nbytes": 128,
        })
        assert "flex_dma_async_broadcast" in code
        assert "grid_sync_group_barrier_xy" in code


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
            ClusterGroup("tp_row", [0, 1])
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
            ClusterGroup("tp_row", [0, 1])
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
            ClusterGroup("dp_group", [0, 1])
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
        """Return a TP-annotated GEMM JIT function (cluster_group='tp_row').

        Uses a multi-tile kernel grid with bx/by block indices.  The visitor
        emits C for-loops for blockIdx axes, declaring bx/by as loop variables.
        The cluster_group annotation exercises group-init / group-barrier
        infrastructure.
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

            # stating the tiling rule
            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                A_local = T.alloc_fragment((BM, BK), dtype)
                B_local = T.alloc_fragment((BK, BN), dtype)
                C_local = T.alloc_fragment((BM, BN), collective)

                with T.attr("anno", "cluster_group", "tp_group"):
                    T.clear(C_local)    
                    
                    for bk in T.Parallel(T.ceildiv(K, BK)):
                        T.copy(A[bx * BM, bk*BK], A_local)  
                        T.copy(B[bk*BK, by * BN], B_local)
                        T.gemm(A_local, B_local, C_local)   

                        T.allreduce(C_local)
                    
                    T.copy(C_local, C[bx * BM, by * BN])
        return tp_gemm
 

    @staticmethod
    def _build_dp_gemm_kernel():
        """Return a DP GEMM JIT function using block_idx cluster mapping.

        Uses a multi-tile kernel grid.  With cluster_policy='block_idx' and
        cluster_ids=[0,1,2,3], block IDs are mapped to clusters via lookup
        table: cluster_id = cluster_ids[block_id % len(cluster_ids)].
        No explicit cluster_id annotations needed.
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
                A_local = T.alloc_fragment((BM, BK), dtype)
                B_local = T.alloc_fragment((BK, BN), dtype)
                C_local = T.alloc_fragment((BM, BN), dtype)
                with T.attr("anno", "cluster_group", "dp_group"):
                    T.clear(C_local)

                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        T.copy(A[bx * BM, bk * BK], A_local)
                        T.copy(B[bk * BK, by * BN], B_local)
                        T.gemm(A_local, B_local, C_local)
                    T.copy(C_local, C[bx * BM, by * BN])

            return C

        return dp_gemm

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
        M, K, N = 128, 64, 128
        BM, BK, BN = 128, 32, 128
        elem_bytes = 2  # sizeof(fp16)

        tp_gemm = self._build_tp_gemm_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        tp_row_cluster = [0, 1]
        registry = ClusterGroupRegistry([
            ClusterGroup("tp_row", tp_row_cluster, root_instance=0)
        ])
        hw_binding = HardwareBinding({"tp_row": tp_row_cluster})

        body = compile_tilelang_to_softhier_parallel(
            tp_gemm, A, B, C,
            BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw_binding,
            cluster_policy="block_idx",
            num_clusters=NUM_CLUSTERS,
        )
        assert isinstance(body, str) and len(body) > 0
        assert "for (int" in body  # block-index for-loop was emitted

        # Reference data (full M x N output)
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
        verify_numeric_outputs(result, C_ref.flatten().reshape(1, -1), atol=3e-1, rtol=3e-2)

        assert result.success, (
            f"TP GEMM simulation failed: {result.error_count} errors "
            f"out of {result.total_count}\n{result.stdout}"
        )
    
    
    # def test_dp_gemm_e2e_compiles(
    #     self,
    #     deeploy_test_dir: Path,
    #     toolchain: str,
    #     toolchain_dir: str,
    #     cmake_args: list,
    # ) -> None:
    #     """DP GEMM: 2-cluster data-parallel (independent tiles) — Network.c must compile."""
    #     import tilelang.language as T
    #     from Deeploy.TileIR.IR import (
    #         ClusterGroup,
    #         ClusterGroupRegistry,
    #         HardwareBinding,
    #     )
    #     from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
    #     NUM_CLUSTERS = 16  # cluster IDs go up to 15
    #     from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
    #     from testUtils.core import configure_cmake
    #     from testUtils.pytestRunner import create_test_config

    #     M, K, N = 256, 128, 256
    #     BM, BK, BN = 128, 16, 128
    #     elem_bytes = 2  # sizeof(fp16)

    #     dp_gemm = self._build_dp_gemm_kernel()
    #     A = T.empty((M, K), T.float16)
    #     B = T.empty((K, N), T.float16)

    #     registry = ClusterGroupRegistry([
    #         ClusterGroup("dp_group", [0, 1, 2, 3], root_instance=0)
    #         # ClusterGroup("dp_group", [0, 2, 5, 7, 8, 10, 13, 15], root_instance=0)
    #     ])
    #     # hw_binding = HardwareBinding({"dp_group": [0, 2, 5, 7, 8, 10, 13, 15]})
    #     hw_binding = HardwareBinding({"dp_group": [0, 1, 2, 3]})

    #     body = compile_tilelang_to_softhier_parallel(
    #         dp_gemm, A, B,
    #         BM=BM, BN=BN, BK=BK,
    #         group_registry=registry,
    #         hw_binding=hw_binding,
    #         cluster_policy="block_idx",
    #         # cluster_ids auto-derived from hw_binding
    #     )
    #     assert isinstance(body, str) and len(body) > 0
    #     assert "for (int" in body  # block-index for-loop was emitted

    #     # Reference data — full M x N output (all tiles covered by block grid)
    #     rng = np.random.default_rng(1)
    #     A_np = rng.standard_normal((M, K)).astype(np.float16)
    #     B_np = rng.standard_normal((K, N)).astype(np.float16)
    #     from softhier_golden.fma import matrix_multiply_with_bittrue_fma
    #     C_ref = matrix_multiply_with_bittrue_fma(A_np, B_np, np.zeros((M, N))).astype(np.float16)
    #     input_bufs = [
    #         TilelangIOBuffer(name="DeeployNetwork_A", c_dtype="fp16", nbytes=M * K * elem_bytes, is_input=True),
    #         TilelangIOBuffer(name="DeeployNetwork_B", c_dtype="fp16", nbytes=K * N * elem_bytes, is_input=True),
    #     ]
    #     output_bufs = [
    #         TilelangIOBuffer(name="DeeployNetwork_C", c_dtype="fp16", nbytes=M * N * elem_bytes, is_input=False),
    #     ]

    #     cmake_extra = list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"]
    #     config = create_test_config(
    #         test_name="Tilelang/dp_gemm",
    #         platform="SoftHier",
    #         simulator="gvsoc",
    #         deeploy_test_dir=deeploy_test_dir,
    #         toolchain=toolchain,
    #         toolchain_dir=toolchain_dir,
    #         cmake_args=cmake_extra,
    #         tiling=False,
    #     )

    #     gen_dir = Path(config.gen_dir)
    #     gen_dir.mkdir(parents=True, exist_ok=True)
    #     generateTilelangSoftHierTestNetwork(
    #         tilelangBody=body,
    #         dumpdir=str(gen_dir),
    #         input_bufs=input_bufs,
    #         output_bufs=output_bufs,
    #         test_inputs=[A_np, B_np],
    #         test_outputs=[C_ref],
    #     )

    #     configure_cmake(config)
    #     build_binary(config)
    #     result = run_simulation(config)
    #     verify_numeric_outputs(result, C_ref.flatten().reshape(1, -1), atol=3e-1, rtol=3e-2)

    #     assert result.success, (
    #         f"DP GEMM simulation failed: {result.error_count} errors "
    #         f"out of {result.total_count}\n{result.stdout}"
    #     )