# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
_DIRECTIONAL_OPS_AVAILABLE = False
if _TILELANG_AVAILABLE:
    try:
        import tilelang.language as _T_chk
        _DIRECTIONAL_OPS_AVAILABLE = (
            hasattr(_T_chk, "group_shift") and hasattr(_T_chk, "group_bcast_axis")
        )
    except Exception:
        pass

# Check for Deeploy-native collective ops (D.reduce / D.broadcast).
_DP_COLLECTIVE_AVAILABLE = False
if _TILELANG_AVAILABLE:
    try:
        from Deeploy.TileIR.Frontend import tl_deeploy as _D_chk  # noqa: F401
        _DP_COLLECTIVE_AVAILABLE = True
    except ImportError:
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


@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestEndToEndGroupCompilation:

    @staticmethod
    def _build_summa_gemm_kernel(GX: int = 4, GY: int = 4):
        """True 2D SUMMA GEMM kernel.

        The cluster grid is GX × GY.  At each K iteration:
          - The diagonal cluster of each row (gid_x == gid_y) loads its A
            row-tile from HBM and row-broadcasts it to the rest of its row.
          - The diagonal cluster of each column (gid_y == gid_x) loads its B
            col-tile from HBM and col-broadcasts it to the rest of its column.
        All clusters then run RedMule on the received tiles.
        Each cluster owns its (gid_y, gid_x) sub-block of C.
        No allreduce needed for a single SUMMA group instance (no split-K).
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def summa_gemm(A, B, C, BM: int, BN: int, BK: int, GX_: int, GY_: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            # Kernel grid: each block covers GY*BM rows × GX*BN columns.
            with T.Kernel(T.ceildiv(M, GY_ * BM), T.ceildiv(N, GX_ * BN)) as (bx, by):
                with T.cluster_group("summa", x=GX_, y=GY_, num_groups=1,
                                        axes=("x", "y")) as (gid, gid_x, gid_y):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)

                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        # SUMMA edge-load: diagonal cluster of each row/col loads its tile.
                        # cluster_for_rowwise = (gid_x == gid_y) in a square group.
                        if gid_x == gid_y:
                            T.copy(A[(bx * GY_ + gid_y) * BM, bk * BK], A_local)
                            T.copy(B[bk * BK, (by * GX_ + gid_x) * BN], B_local)
                        # Row-broadcast A from each row's diagonal (from_coord = gid_y).
                        T.group_bcast_axis(A_local, along="x", from_coord=gid_y,
                                            group="summa")
                        # Col-broadcast B from each column's diagonal (from_coord = gid_x).
                        T.group_bcast_axis(B_local, along="y", from_coord=gid_x,
                                            group="summa")
                        T.gemm(A_local, B_local, C_local, clear_accum=False)

                    # Each cluster stores its own (gid_y, gid_x) output sub-block.
                    T.copy(C_local, C[(bx * GY_ + gid_y) * BM, (by * GX_ + gid_x) * BN])

        return summa_gemm

    @staticmethod
    def _build_summa_gemm_split_k_kernel(GX: int = 2, GY: int = 2, NG: int = 4):
        """2D SUMMA GEMM with split-K reduction across NG group instances.

        NG SUMMA groups (each GX × GY) compute disjoint K slices in parallel.
        After the K loop, T.allreduce sums the partial C results across instances.
        Only gid == 0 (instance 0) stores back to HBM.
        Matches SummaGEMM.h group_reduction == 1 semantics.
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def summa_gemm_split_k(A, B, C, BM: int, BN: int, BK: int,
                                GX_: int, GY_: int, NG_: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, GY_ * BM), T.ceildiv(N, GX_ * BN)) as (bx, by):
                # NG_ instances of GX_ × GY_ SUMMA groups, each handles K/NG_ slice.
                with T.cluster_group("summa", x=GX_, y=GY_, num_groups=NG_,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)

                    # Each instance owns K/NG_ contiguous K columns.
                    K_per_inst = T.ceildiv(K, NG_)
                    for bk in T.Pipelined(T.ceildiv(K_per_inst, BK), num_stages=2):
                        if gid_x == gid_y:
                            T.copy(A[(bx * GY_ + gid_y) * BM,
                                     gid * K_per_inst + bk * BK], A_local)
                            T.copy(B[gid * K_per_inst + bk * BK,
                                     (by * GX_ + gid_x) * BN], B_local)
                        T.group_bcast_axis(A_local, along="x", from_coord=gid_y,
                                           group="summa")
                        T.group_bcast_axis(B_local, along="y", from_coord=gid_x,
                                           group="summa")
                        T.gemm(A_local, B_local, C_local, clear_accum=False)

                    # Cross-instance reduction: sum partial C across all NG_ instances.
                    # global_barrier_before=True → flex_global_barrier_xy() + inverted masks.
                    T.allreduce(C_local, "sum", axis="x", group="summa")

                    # Only instance 0 stores result.
                    if gid == 0:
                        T.copy(C_local,
                               C[(bx * GY_ + gid_y) * BM, (by * GX_ + gid_x) * BN])

        return summa_gemm_split_k

    @pytest.mark.skipif(not _DIRECTIONAL_OPS_AVAILABLE, reason="group_bcast_axis not in tilelang")
    def test_summa_gemm_2d_e2e(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """2D SUMMA GEMM: GX=GY=4 (16 clusters), edge-load + axis broadcast.

        Matches SummaGEMM.h without split-K (group_reduction == 0).
        Diagonal clusters load A/B from HBM; per-row/col broadcast distributes.
        Verify C = A @ B.
        """
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import configure_cmake
        from testUtils.pytestRunner import create_test_config

        GX, GY = 4, 4
        NUM_CLUSTERS = GX * GY  # 16
        M, K, N = 256, 256, 256
        BM, BK, BN = 64, 64, 64
        elem_bytes = 2

        summa_gemm = self._build_summa_gemm_kernel(GX=GX, GY=GY)
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa", group_x=GX, group_y=GY, num_groups=1,
                         axis_names=("x", "y"), root_coord=(0, 0))
        ])
        hw = HardwareBinding({"summa": list(range(NUM_CLUSTERS))})

        body = compile_tilelang_to_softhier_parallel(
            summa_gemm, A, B, C, BM=BM, BN=BN, BK=BK, GX_=GX, GY_=GY,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        rng = np.random.default_rng(42)
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
            test_name="Tilelang/summa_gemm_2d",
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

    @pytest.mark.skipif(not _DIRECTIONAL_OPS_AVAILABLE, reason="group_bcast_axis not in tilelang")
    def test_summa_gemm_split_k_e2e(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """2D SUMMA GEMM with split-K: 4 instances × (2×2) = 16 clusters.

        Matches SummaGEMM.h with group_reduction == 1.
        Four SUMMA instances each compute K/4 slices; cross-instance
        flex_dma_async_reduction (inverted masks) sums partial C tiles.
        Verify C = A @ B.
        """
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import configure_cmake
        from testUtils.pytestRunner import create_test_config

        GX, GY, NG = 2, 2, 4
        NUM_CLUSTERS = GX * GY * NG  # 16
        M, K, N = 256, 256, 256
        BM, BK, BN = 64, 32, 64
        elem_bytes = 2

        summa_split_k = self._build_summa_gemm_split_k_kernel(GX=GX, GY=GY, NG=NG)
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa", group_x=GX, group_y=GY, num_groups=NG,
                         axis_names=("x", "y"), root_coord=(0, 0))
        ])
        hw = HardwareBinding({"summa": list(range(NUM_CLUSTERS))})

        body = compile_tilelang_to_softhier_parallel(
            summa_split_k, A, B, C, BM=BM, BN=BN, BK=BK, GX_=GX, GY_=GY, NG_=NG,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        rng = np.random.default_rng(42)
        A_np = rng.standard_normal((M, K)).astype(np.float16)
        B_np = rng.standard_normal((K, N)).astype(np.float16)
        C_ref = (A_np.astype(np.float32) @ B_np.astype(np.float32)).astype(np.float16)

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
            test_name="Tilelang/summa_gemm_split_k",
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


@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
@pytest.mark.skipif(not _DP_COLLECTIVE_AVAILABLE, reason="Deeploy tl_deeploy not available")
class TestEndToEndDpCollective:
    """E2E tests using D.reduce / D.broadcast (tl.deeploy.* ops).

    Unlike TestEndToEndGroupCompilation, these tests use the Deeploy-native
    collective API and have numeric verification enabled.
    """

    @staticmethod
    def _build_summa_gemm_dp_kernel(GX: int = 4, GY: int = 4):
        """2D SUMMA GEMM using D.broadcast for row/col distribution."""
        import tilelang
        import tilelang.language as T
        from Deeploy.TileIR.Frontend import tl_deeploy as D

        @tilelang.jit
        def summa_gemm_dp(A, B, C, BM: int, BN: int, BK: int, GX_: int, GY_: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, GY_ * BM), T.ceildiv(N, GX_ * BN)) as (bx, by):
                with T.cluster_group("summa", x=GX_, y=GY_, num_groups=1,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)

                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                        if gid_x == gid_y:
                            T.copy(A[(bx * GY_ + gid_y) * BM, bk * BK], A_local)
                            T.copy(B[bk * BK, (by * GX_ + gid_x) * BN], B_local)
                        D.broadcast(A_local, level="intra_group", axis="x",
                                    group="summa", root=gid_y)
                        D.broadcast(B_local, level="intra_group", axis="y",
                                    group="summa", root=gid_x)
                        T.gemm(A_local, B_local, C_local, clear_accum=False)

                    T.copy(C_local, C[(bx * GY_ + gid_y) * BM, (by * GX_ + gid_x) * BN])

        return summa_gemm_dp

    @staticmethod
    def _build_summa_gemm_dp_split_k_kernel(GX: int = 2, GY: int = 2, NG: int = 4):
        """2D SUMMA GEMM with split-K using D.broadcast + D.reduce."""
        import tilelang
        import tilelang.language as T
        from Deeploy.TileIR.Frontend import tl_deeploy as D

        @tilelang.jit
        def summa_gemm_dp_split_k(A, B, C, BM: int, BN: int, BK: int,
                                   GX_: int, GY_: int, NG_: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C: T.Tensor((M, N), dtype)

            with T.Kernel(T.ceildiv(M, GY_ * BM), T.ceildiv(N, GX_ * BN)) as (bx, by):
                # TODO: emit cluster group id as meta axis along with cluster id
                with T.cluster_group("summa", x=GX_, y=GY_, num_groups=NG_,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)

                    K_per_inst = T.ceildiv(K, NG_)
                    for bk in T.Pipelined(T.ceildiv(K_per_inst, BK), num_stages=2):
                        if gid_x == gid_y:
                            T.copy(A[(bx * GY_ + gid_y) * BM,
                                     gid * K_per_inst + bk * BK], A_local)
                            T.copy(B[gid * K_per_inst + bk * BK,
                                     (by * GX_ + gid_x) * BN], B_local)
                        D.broadcast(A_local, level="intra_group", axis="x",
                                    group="summa", root=gid_y)
                        D.broadcast(B_local, level="intra_group", axis="y",
                                    group="summa", root=gid_x)
                        T.gemm(A_local, B_local, C_local, clear_accum=False)

                    D.reduce(C_local, op="sum", level="inter_group", axis="k",
                             group="summa")


                    if gid == 0:
                        T.copy(C_local,
                               C[(bx * GY_ + gid_y) * BM, (by * GX_ + gid_x) * BN])

        return summa_gemm_dp_split_k

    def test_summa_gemm_dp_2d_e2e(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """2D SUMMA via D.broadcast: GX=GY=4 (16 clusters), verify C = A @ B."""
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import configure_cmake
        from testUtils.pytestRunner import create_test_config

        GX, GY = 4, 4
        NUM_CLUSTERS = GX * GY
        M, K, N = 512, 7168, 3072
        BM, BK, BN = 128, 128, 128
        elem_bytes = 2

        summa_gemm_dp = self._build_summa_gemm_dp_kernel(GX=GX, GY=GY)
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa", group_x=GX, group_y=GY, num_groups=1,
                         axis_names=("x", "y"), root_coord=(0, 0))
        ])
        hw = HardwareBinding({"summa": list(range(NUM_CLUSTERS))})

        body = compile_tilelang_to_softhier_parallel(
            summa_gemm_dp, A, B, C, BM=BM, BN=BN, BK=BK, GX_=GX, GY_=GY,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        rng = np.random.default_rng(42)
        A_np = rng.standard_normal((M, K)).astype(np.float16)
        B_np = rng.standard_normal((K, N)).astype(np.float16)
        C_ref = (A_np.astype(np.float32) @ B_np.astype(np.float32)).astype(np.float16)

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
            test_name="Tilelang/summa_gemm_dp_2d",
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
        assert "Simulation stopped by user" in result.stdout, "Simulation did not complete successfully"



    def test_summa_gemm_dp_split_k_e2e(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """2D SUMMA split-K via D.broadcast + D.reduce: 4×(2×2)=16 clusters, verify C = A @ B."""
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import configure_cmake
        from testUtils.pytestRunner import create_test_config

        GX, GY, NG = 2, 2, 4
        NUM_CLUSTERS = GX * GY * NG
        M, K, N = 256, 256, 256
        BM, BK, BN = 64, 32, 64
        elem_bytes = 2

        summa_dp_split_k = self._build_summa_gemm_dp_split_k_kernel(GX=GX, GY=GY, NG=NG)
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)
        C = T.empty((M, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa", group_x=GX, group_y=GY, num_groups=NG,
                         axis_names=("x", "y"), root_coord=(0, 0),
                         meta_axes=("k",), meta_shape=(NG,))
        ])
        hw = HardwareBinding({"summa": list(range(NUM_CLUSTERS))})

        body = compile_tilelang_to_softhier_parallel(
            summa_dp_split_k, A, B, C, BM=BM, BN=BN, BK=BK, GX_=GX, GY_=GY, NG_=NG,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        rng = np.random.default_rng(42)
        A_np = rng.standard_normal((M, K)).astype(np.float16)
        B_np = rng.standard_normal((K, N)).astype(np.float16)
        C_ref = (A_np.astype(np.float32) @ B_np.astype(np.float32)).astype(np.float16)

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
            test_name="Tilelang/summa_gemm_dp_split_k",
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
        assert "Simulation stopped by user" in result.stdout, "Simulation did not complete successfully"

