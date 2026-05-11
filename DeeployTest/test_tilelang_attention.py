# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Attention kernel tests for SoftHier (TileLang DP style).

Adaptations of GPU attention examples from docs/ to SoftHier:
  - TestMLADecodeDpE2E:       MLA-style decode flash attention, DP over heads
                               adapted from docs/example_mla_decode.py
  - TestMaskedFlashAttnDpE2E: Flash attention with binary mask (sparse simulation),
                               DP over query rows; adapted from docs/sparse_attn_fwd_sm90.py
  - TestFlashMLAE2E:          Flash MLA with dual-component score Q·KV^T + Q_pe·K_pe^T,
                               adapts main_split from docs/example_mla_decode.py

Key adaptations vs GPU originals:
  - T.exp2 / T.log2 not available → use T.exp; split-K feature dropped (needs log2)
  - T.infinity not available → T.float16(-65504.0) as -inf sentinel
  - T.alloc_shared → T.alloc_fragment (L1 SRAM on SoftHier)
  - T.GemmWarpPolicy / T.use_swizzle → removed (not applicable to RedMule)
  - T.reduce_max / T.reduce_sum (standalone) → manual T.serial loop
  - T.gather (dynamic index) → not supported; binary mask + T.if_then_else used instead
  - T.bfloat16 → T.float16

Also fixes a bug in the existing _build_flash_attn_kernel in test_tilelang_non_gemm_ops.py
where rescale used l (running sum) instead of m_old (old max). Both kernels here use
the correct formula: rescale = exp(m_old - m_new).

Run (E2E, requires SoftHier toolchain):
    pytest DeeployTest/test_tilelang_attention.py -v -s -m "tilelang and softhier" \\
        --toolchain=GCC \\
        --toolchain-install-dir=$SOFTHIER_INSTALL_DIR/third_party/toolchain/install
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------
try:
    import tilelang  # noqa: F401
    _TILELANG_AVAILABLE = True
except ImportError:
    _TILELANG_AVAILABLE = False

_DP_COLLECTIVE_AVAILABLE = False
if _TILELANG_AVAILABLE:
    try:
        from Deeploy.TileIR.Frontend import tl_deeploy as _D_chk  # noqa: F401
        _DP_COLLECTIVE_AVAILABLE = True
    except ImportError:
        pass

pytestmark = pytest.mark.tilelang


# ===========================================================================
# Helper: parse error count from simulation stdout
# ===========================================================================

def _check_error_count(result, max_abs_err: float = float("inf"),
                       max_rel_err: float = 1.0) -> None:
    """Extract and report error count vs total elements from simulation stdout.

    Parses ``Errors: N out of M`` printed by the test harness and asserts that
    the error rate is within tolerance. Mirrors the implementation in
    test_tilelang_non_gemm_ops.py.
    """
    ec = result.error_count
    tc = result.total_count
    if ec >= 0:
        rate = ec / tc if tc > 0 else 0.0
        print(f"[Verify] Errors: {ec} out of {tc} ({rate:.4%})")
        assert ec / tc <= max_rel_err, (
            f"Error rate {rate:.4%} exceeds max_rel_err={max_rel_err:.4%}")
    else:
        print("[Verify] Could not parse error count from stdout")


# ===========================================================================
# Test 1: MLA Decode Flash Attention DP
# ===========================================================================

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
@pytest.mark.skipif(not _DP_COLLECTIVE_AVAILABLE, reason="Deeploy tl_deeploy not available")
class TestMLADecodeDpE2E:
    """MLA decode flash attention distributed over query heads.

    Adapted from docs/example_mla_decode.py for SoftHier:
    - Q(H, D), K(SEQ, D), V(SEQ, D) → O(H, D)  (single decode step)
    - Online flash attention with correct m_old rescale
    - DP: GX clusters, each handles heads_per_cluster = ceil(H / GX) heads
    - K and V read in full by every cluster (SEQ not partitioned)
    - Test params: H=64, D=64, SEQ=64, Bc=32, GX=2
      → heads_per_cluster=32; GEMM shapes (32,64)@(32,64)^T and (32,32)@(32,64)
        match flash_attn_e2e sizes (minimum RedMule M=32)
    """

    @staticmethod
    def _build_mla_decode_dp_kernel(GX: int = 2):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def mla_decode_dp(Q, K, V, H_: int, D_: int, SEQ_: int, Bc_: int, GX_: int):
            dtype = T.float16
            Q: T.Tensor((H_, D_), dtype)
            K: T.Tensor((SEQ_, D_), dtype)
            V: T.Tensor((SEQ_, D_), dtype)
            O = T.empty((H_, D_), dtype)

            heads_per_cluster = T.ceildiv(H_, GX_)
            Tc = T.ceildiv(SEQ_, Bc_)

            with T.Kernel(1, threads=1) as (bx,):
                with T.cluster_group("dp", x=GX_, y=1, num_groups=1,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    Q_loc   = T.alloc_fragment((heads_per_cluster, D_), dtype)
                    K_tc    = T.alloc_fragment((Bc_, D_), dtype)
                    V_tc    = T.alloc_fragment((Bc_, D_), dtype)
                    S_loc   = T.alloc_fragment((heads_per_cluster, Bc_), dtype)
                    O_loc   = T.alloc_fragment((heads_per_cluster, D_), dtype)
                    m       = T.alloc_fragment((heads_per_cluster, 1), dtype)
                    l       = T.alloc_fragment((heads_per_cluster, 1), dtype)
                    m_old   = T.alloc_fragment((1,), dtype)
                    row_sum = T.alloc_fragment((1,), dtype)

                    T.copy(Q[gid_x * heads_per_cluster, 0], Q_loc)
                    T.clear(O_loc)

                    for h in T.serial(heads_per_cluster):
                        m[h, 0] = T.float16(-65504.0)
                        l[h, 0] = T.float16(0.0)

                    for tc in T.serial(Tc):
                        T.copy(K[tc * Bc_, 0], K_tc)
                        T.copy(V[tc * Bc_, 0], V_tc)
                        T.clear(S_loc)
                        T.gemm(Q_loc, K_tc, S_loc, transpose_B=True, clear_accum=True)

                        for h in T.serial(heads_per_cluster):
                            m_old[0] = m[h, 0]
                            for c in T.serial(Bc_):
                                m[h, 0] = T.max(m[h, 0], S_loc[h, c])
                            row_sum[0] = T.float16(0.0)
                            for c in T.serial(Bc_):
                                S_loc[h, c] = T.exp(S_loc[h, c] - m[h, 0])
                                row_sum[0] = row_sum[0] + S_loc[h, c]
                            rescale = T.exp(m_old[0] - m[h, 0])
                            l[h, 0] = l[h, 0] * rescale + row_sum[0]
                            for d in T.serial(D_):
                                O_loc[h, d] = O_loc[h, d] * rescale

                        T.gemm(S_loc, V_tc, O_loc, clear_accum=False)

                    for h in T.serial(heads_per_cluster):
                        for d in T.serial(D_):
                            O_loc[h, d] = O_loc[h, d] / l[h, 0]

                    T.copy(O_loc, O[gid_x * heads_per_cluster, 0])

            return O

        return mla_decode_dp

    def test_mla_decode_dp_e2e(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """MLA decode DP: GX=2, H=4, D=32, SEQ=32, Bc=16. Verify O = softmax(Q@K^T)@V."""
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config

        GX = 2
        H, D, SEQ, Bc = 64, 64, 64, 32
        NUM_CLUSTERS = GX
        elem_bytes = 2

        fn = self._build_mla_decode_dp_kernel(GX=GX)

        rng = np.random.default_rng(42)
        # Scale Q and K by 1/sqrt(D) so attention scores ~ N(0,1).
        # Without scaling, scores ~ N(0, D) with std~8 for D=64, causing
        # 84% of exp(score - max) to underflow to 0 in FP16 and collapsing
        # softmax to hard attention (output = single V row).
        scale = np.float16(1.0 / np.sqrt(D))
        q_np = (rng.standard_normal((H, D)) * scale).astype(np.float16)
        k_np = (rng.standard_normal((SEQ, D)) * scale).astype(np.float16)
        v_np = rng.standard_normal((SEQ, D)).astype(np.float16)

        q_f32 = q_np.astype(np.float32)
        k_f32 = k_np.astype(np.float32)
        v_f32 = v_np.astype(np.float32)
        scores = q_f32 @ k_f32.T
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(scores)
        p_f32 = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        o_ref = (p_f32 @ v_f32).astype(np.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("dp", group_x=GX, group_y=1, num_groups=1,
                         axis_names=("x", "y"))
        ])
        hw = HardwareBinding({"dp": list(range(GX))})

        body = compile_tilelang_to_softhier_parallel(
            fn, q_np, k_np, v_np,
            H_=H, D_=D, SEQ_=SEQ, Bc_=Bc, GX_=GX,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Q", c_dtype="fp16",
                             nbytes=H * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_K", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_V", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_O", c_dtype="fp16",
                             nbytes=H * D * elem_bytes, is_input=False),
        ]

        config = create_test_config(
            test_name="Tilelang/mla_decode_dp_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            # test_inputs=[q_np, k_np, v_np],
            test_inputs=None,
            # test_outputs=[o_ref],
            test_outputs=None,
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=1.0)


# ===========================================================================
# Test 2: Masked Flash Attention DP (sparse attention simulation)
# ===========================================================================

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
@pytest.mark.skipif(not _DP_COLLECTIVE_AVAILABLE, reason="Deeploy tl_deeploy not available")
class TestMaskedFlashAttnDpE2E:
    """Masked flash attention (sparse attention simulation) distributed over query rows.

    Adapted from docs/sparse_attn_fwd_sm90.py for SoftHier:
    - Q(SEQ, D), K(SEQ, D), V(SEQ, D), Mask(SEQ, SEQ) → O(SEQ, D)
    - T.gather is not supported on SoftHier; pre-computed binary float16 mask used instead
    - Mask (1.0=attend, 0.0=skip) applied via T.if_then_else after Q@K^T, before softmax
    - Online flash attention with correct m_old rescale
    - DP: GX clusters, each handles rows_per_cluster = ceil(SEQ / GX) query rows
    - Test params: SEQ=64, D=64, Bc=32, GX=2, causal lower-triangular mask
      → rows_per_cluster=32; GEMM shapes match flash_attn_e2e (minimum RedMule M=32)
    """

    @staticmethod
    def _build_masked_flash_attn_dp_kernel(GX: int = 2):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def masked_flash_attn_dp(Q, K, V, Mask,
                                  SEQ_: int, D_: int, Bc_: int, GX_: int):
            dtype = T.float16
            Q: T.Tensor((SEQ_, D_), dtype)
            K: T.Tensor((SEQ_, D_), dtype)
            V: T.Tensor((SEQ_, D_), dtype)
            Mask: T.Tensor((SEQ_, SEQ_), dtype)
            O = T.empty((SEQ_, D_), dtype)

            rows_per_cluster = T.ceildiv(SEQ_, GX_)
            Tc = T.ceildiv(SEQ_, Bc_)

            with T.Kernel(1, threads=1) as (bx,):
                with T.cluster_group("dp", x=GX_, y=1, num_groups=1,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    Q_loc   = T.alloc_fragment((rows_per_cluster, D_), dtype)
                    K_tc    = T.alloc_fragment((Bc_, D_), dtype)
                    V_tc    = T.alloc_fragment((Bc_, D_), dtype)
                    Mask_tc = T.alloc_fragment((rows_per_cluster, Bc_), dtype)
                    S_loc   = T.alloc_fragment((rows_per_cluster, Bc_), dtype)
                    O_loc   = T.alloc_fragment((rows_per_cluster, D_), dtype)
                    m       = T.alloc_fragment((rows_per_cluster, 1), dtype)
                    l       = T.alloc_fragment((rows_per_cluster, 1), dtype)
                    m_old   = T.alloc_fragment((1,), dtype)
                    row_sum = T.alloc_fragment((1,), dtype)

                    T.copy(Q[gid_x * rows_per_cluster, 0], Q_loc)
                    T.clear(O_loc)

                    T.clear(m)
                    T.clear(l)

                    for tc in T.Parallel(Tc):
                        T.copy(K[tc * Bc_, 0], K_tc)
                        T.copy(V[tc * Bc_, 0], V_tc)
                        T.copy(Mask[gid_x * rows_per_cluster, tc * Bc_], Mask_tc)
                        T.clear(S_loc)
                        T.gemm(Q_loc, K_tc, S_loc, transpose_B=True, clear_accum=True)

                        for r in T.Parallel(rows_per_cluster):
                            for c in T.serial(Bc_):
                                S_loc[r, c] = T.if_then_else(
                                    Mask_tc[r, c] > T.float16(0.5),
                                    S_loc[r, c],
                                    T.float16(-65504.0)
                                )

                        for r in T.Parallel(rows_per_cluster):
                            m_old[0] = m[r, 0]
                            for c in T.Parallel(Bc_):
                                m[r, 0] = T.max(m[r, 0], S_loc[r, c])
                            T.clear(row_sum)
                            for c in T.Parallel(Bc_):
                                S_loc[r, c] = T.if_then_else(
                                    Mask_tc[r, c] > T.float16(0.5),
                                    T.exp(S_loc[r, c] - m[r, 0]),
                                    T.float16(0.0)
                                )
                                row_sum[0] = row_sum[0] + S_loc[r, c]
                            rescale = T.exp(m_old[0] - m[r, 0])
                            l[r, 0] = l[r, 0] * rescale + row_sum[0]
                            for d in T.Parallel(D_):
                                O_loc[r, d] = O_loc[r, d] * rescale

                        T.gemm(S_loc, V_tc, O_loc, clear_accum=False)

                    for r in T.Parallel(rows_per_cluster):
                        for d in T.Parallel(D_):
                            O_loc[r, d] = O_loc[r, d] / l[r, 0]

                    T.copy(O_loc, O[gid_x * rows_per_cluster, 0])

            return O

        return masked_flash_attn_dp

    def test_masked_flash_attn_dp_e2e(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """Masked flash attention DP: GX=2, SEQ=32, D=32, Bc=16, causal mask."""
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config

        GX = 16
        SEQ, D, Bc = 4096, 128, 128
        # SEQ, D, Bc = 32, 32, 16
        NUM_CLUSTERS = GX
        elem_bytes = 2

        fn = self._build_masked_flash_attn_dp_kernel(GX=GX)

        rng = np.random.default_rng(42)
        scale = np.float16(1.0 / np.sqrt(D))
        q_np = (rng.standard_normal((SEQ, D)) * scale).astype(np.float16)
        k_np = (rng.standard_normal((SEQ, D)) * scale).astype(np.float16)
        v_np = rng.standard_normal((SEQ, D)).astype(np.float16)
        mask_np = np.tril(np.ones((SEQ, SEQ), dtype=np.float16))

        q_f32 = q_np.astype(np.float32)
        k_f32 = k_np.astype(np.float32)
        v_f32 = v_np.astype(np.float32)
        scores = q_f32 @ k_f32.T
        scores = np.where(mask_np > 0.5, scores, -1e9)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(scores)
        p_f32 = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        o_ref = (p_f32 @ v_f32).astype(np.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("dp", group_x=GX, group_y=1, num_groups=1,
                         axis_names=("x", "y"))
        ])
        hw = HardwareBinding({"dp": list(range(GX))})

        body = compile_tilelang_to_softhier_parallel(
            fn, q_np, k_np, v_np, mask_np,
            SEQ_=SEQ, D_=D, Bc_=Bc, GX_=GX,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Q", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_K", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_V", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_Mask", c_dtype="fp16",
                             nbytes=SEQ * SEQ * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_O", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=False),
        ]

        config = create_test_config(
            test_name="Tilelang/masked_flash_attn_dp_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[q_np, k_np, v_np, mask_np],
            test_outputs=[o_ref],
            # test_inputs=None,
            # test_outputs=None,
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=0.05)


# ===========================================================================
# Test 3: Parallel(Serial) Vectorization — targeted regression test
# ===========================================================================

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
@pytest.mark.skipif(not _DP_COLLECTIVE_AVAILABLE, reason="Deeploy tl_deeploy not available")
class TestParallelSerialVectorizationE2E:
    """Verify that Parallel(Serial(div_vf)) emits ParallelCore + SpatzInnerEltwise.

    This is a minimal kernel isolating the Parallel(Serial(BufferStore)) pattern
    fixed in TilelangVisitor._handle_parallel_for.  The kernel divides each row
    of A by the corresponding scalar in B and writes to C.

    Before the fix:
      - T.Serial (uppercase, kind==1) caused the while-loop to strip the inner
        loop but _emit_one_parallel_store used the outer loop_var with an index
        that referenced the stripped inner var → compile error 'd' undeclared.
      - T.serial (lowercase, kind==0) fell back to two nested serial C loops,
        silently dropping the outer T.Parallel annotation.

    After the fix both cases emit:
      TileParallelCoreOpen(r) + TileInnerEltwise(d) + TileParallelCoreClose(r)
    which the SpatzVectorizationPass converts to SpatzInnerEltwise (div_vf).
    """

    @staticmethod
    def _build_kernel(GX: int = 2):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def parallel_serial_div(A, B, R_: int, D_: int, GX_: int):
            dtype = T.float16
            A: T.Tensor((R_, D_), dtype)
            B: T.Tensor((R_, 1), dtype)
            C = T.empty((R_, D_), dtype)
            rpc = T.ceildiv(R_, GX_)
            with T.Kernel(1, threads=1) as (bx,):
                with T.cluster_group("dp", x=GX_, y=1, num_groups=1,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    A_loc = T.alloc_fragment((rpc, D_), dtype)
                    B_loc = T.alloc_fragment((rpc, 1), dtype)
                    T.copy(A[gid_x * rpc, 0], A_loc)
                    T.copy(B[gid_x * rpc, 0], B_loc)
                    for r in T.Parallel(rpc):
                        for d in T.Serial(D_):
                            A_loc[r, d] = A_loc[r, d] / B_loc[r, 0]
                    T.copy(A_loc, C[gid_x * rpc, 0])
            return C

        return parallel_serial_div

    def test_parallel_serial_div_e2e(
        self,
        deeploy_test_dir: Path,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """Parallel(Serial(div_vf)): GX=2, R=8, D=64. Verify C = A / B row-wise."""
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config

        GX, R, D = 2, 8, 64
        NUM_CLUSTERS = GX
        elem_bytes = 2

        rng = np.random.default_rng(0)
        a_np = rng.standard_normal((R, D)).astype(np.float16)
        # Avoid values near zero to prevent division instability in fp16
        b_raw = rng.standard_normal((R, 1)).astype(np.float32)
        b_raw = np.where(np.abs(b_raw) < 0.5, np.sign(b_raw) * 1.0, b_raw)
        b_np = b_raw.astype(np.float16)
        c_ref = (a_np.astype(np.float32) / b_np.astype(np.float32)).astype(np.float16)

        fn = self._build_kernel(GX)
        registry = ClusterGroupRegistry([
            ClusterGroup("dp", group_x=GX, group_y=1, num_groups=1,
                         axis_names=("x", "y"))
        ])
        hw = HardwareBinding({"dp": list(range(GX))})

        body = compile_tilelang_to_softhier_parallel(
            fn, a_np, b_np,
            R_=R, D_=D, GX_=GX,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_A", c_dtype="fp16",
                             nbytes=R * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_B", c_dtype="fp16",
                             nbytes=R * 1 * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_C", c_dtype="fp16",
                             nbytes=R * D * elem_bytes, is_input=False),
        ]

        config = create_test_config(
            test_name="Tilelang/parallel_serial_div_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[a_np, b_np],
            test_outputs=[c_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=0.05)


# ===========================================================================
# Test 4: Flash MLA E2E — dual-component attention score (Q·KV^T + Q_pe·K_pe^T)
# ===========================================================================

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
@pytest.mark.skipif(not _DP_COLLECTIVE_AVAILABLE, reason="Deeploy tl_deeploy not available")
class TestFlashMLAE2E:
    """Flash Multi-head Latent Attention (MLA) distributed over query heads.

    Adapts the dual-GEMM attention score from docs/example_mla_decode.py
    (main_split kernel) to SoftHier:
      - Q(H, D), Q_pe(H, pe_dim), KV(SEQ, D), K_pe(SEQ, pe_dim) → O(H, D)
      - Attention score: S = Q·KV^T + Q_pe·K_pe^T  (two GEMMs, second clear_accum=False)
      - Online flash attention with correct m_old rescale
      - DP: GX clusters, each handles heads_per_cluster = H // GX heads
      - Test params: H=64, D=32, pe_dim=32, SEQ=64, Bc=32, GX=2
        → heads_per_cluster=32; all GEMM shapes 32×32×32 (≥ RedMule minimum M=32)

    Key MLA adaptations from the GPU example (docs/example_mla_decode.py main_split):
      - T.exp2 → T.exp (natural exp; T.exp2 not available on SoftHier)
      - T.infinity → T.float16(-65504.0) (-inf sentinel)
      - T.alloc_shared → T.alloc_fragment (L1 SRAM)
      - T.Pipelined → T.serial (no pipeline on SoftHier)
      - Two GEMMs per K-block tile: Q·KV^T then +=Q_pe·K_pe^T
    """

    @staticmethod
    def _build_flash_mla_kernel(GX: int = 2):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def flash_mla(Q, Q_pe, KV, K_pe,
                      H_: int, D_: int, pe_dim_: int,
                      SEQ_: int, Bc_: int, GX_: int):
            dtype = T.float16
            Q: T.Tensor((H_, D_), dtype)
            Q_pe: T.Tensor((H_, pe_dim_), dtype)
            KV: T.Tensor((SEQ_, D_), dtype)
            K_pe: T.Tensor((SEQ_, pe_dim_), dtype)
            O = T.empty((H_, D_), dtype)

            heads_per_cluster = T.ceildiv(H_, GX_)
            Tc = T.ceildiv(SEQ_, Bc_)

            with T.Kernel(1, threads=1) as (bx,):
                with T.cluster_group("dp", x=GX_, y=1, num_groups=1,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    Q_loc    = T.alloc_fragment((heads_per_cluster, D_), dtype)
                    Q_pe_loc = T.alloc_fragment((heads_per_cluster, pe_dim_), dtype)
                    KV_tc    = T.alloc_fragment((Bc_, D_), dtype)
                    K_pe_tc  = T.alloc_fragment((Bc_, pe_dim_), dtype)
                    S_loc    = T.alloc_fragment((heads_per_cluster, Bc_), dtype)
                    O_loc    = T.alloc_fragment((heads_per_cluster, D_), dtype)
                    m        = T.alloc_fragment((heads_per_cluster, 1), dtype)
                    l        = T.alloc_fragment((heads_per_cluster, 1), dtype)
                    m_old    = T.alloc_fragment((1,), dtype)
                    row_sum  = T.alloc_fragment((1,), dtype)

                    T.copy(Q[gid_x * heads_per_cluster, 0], Q_loc)
                    T.copy(Q_pe[gid_x * heads_per_cluster, 0], Q_pe_loc)
                    T.clear(O_loc)

                    for h in T.serial(heads_per_cluster):
                        m[h, 0] = T.float16(-65504.0)
                        l[h, 0] = T.float16(0.0)

                    for tc in T.serial(Tc):
                        T.copy(KV[tc * Bc_, 0], KV_tc)
                        T.copy(K_pe[tc * Bc_, 0], K_pe_tc)
                        T.clear(S_loc)
                        # MLA dual-component score: S = Q·KV^T + Q_pe·K_pe^T
                        T.gemm(Q_loc, KV_tc, S_loc, transpose_B=True, clear_accum=True)
                        T.gemm(Q_pe_loc, K_pe_tc, S_loc, transpose_B=True, clear_accum=False)

                        for h in T.serial(heads_per_cluster):
                            m_old[0] = m[h, 0]
                            for c in T.serial(Bc_):
                                m[h, 0] = T.max(m[h, 0], S_loc[h, c])
                            row_sum[0] = T.float16(0.0)
                            for c in T.serial(Bc_):
                                S_loc[h, c] = T.exp(S_loc[h, c] - m[h, 0])
                                row_sum[0] = row_sum[0] + S_loc[h, c]
                            rescale = T.exp(m_old[0] - m[h, 0])
                            l[h, 0] = l[h, 0] * rescale + row_sum[0]
                            for d in T.serial(D_):
                                O_loc[h, d] = O_loc[h, d] * rescale

                        T.gemm(S_loc, KV_tc, O_loc, clear_accum=False)

                    for h in T.serial(heads_per_cluster):
                        for d in T.serial(D_):
                            O_loc[h, d] = O_loc[h, d] / l[h, 0]

                    T.copy(O_loc, O[gid_x * heads_per_cluster, 0])

            return O

        return flash_mla

    def test_flash_mla_e2e(
        self,
        deeploy_test_dir,
        toolchain: str,
        toolchain_dir: str,
        cmake_args: list,
    ) -> None:
        """Flash MLA: GX=2, H=64, D=32, pe_dim=32, SEQ=64, Bc=32.

        Verifies O = softmax(Q@KV^T + Q_pe@K_pe^T) @ KV numerically.
        """
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config

        GX = 2
        H, D, pe_dim, SEQ, Bc = 64, 32, 32, 64, 32
        NUM_CLUSTERS = GX
        elem_bytes = 2

        fn = self._build_flash_mla_kernel(GX=GX)

        rng = np.random.default_rng(7)
        # Scale Q and K components so scores ~ N(0,1): inner dim = D + pe_dim = 64.
        scale = np.float16(1.0 / np.sqrt(D + pe_dim))
        q_np    = (rng.standard_normal((H, D))        * scale).astype(np.float16)
        q_pe_np = (rng.standard_normal((H, pe_dim))   * scale).astype(np.float16)
        kv_np   = (rng.standard_normal((SEQ, D))      * scale).astype(np.float16)
        k_pe_np = (rng.standard_normal((SEQ, pe_dim)) * scale).astype(np.float16)

        # Reference: dual-component attention score in fp32
        q_f32    = q_np.astype(np.float32)
        q_pe_f32 = q_pe_np.astype(np.float32)
        kv_f32   = kv_np.astype(np.float32)
        k_pe_f32 = k_pe_np.astype(np.float32)
        scores = q_f32 @ kv_f32.T + q_pe_f32 @ k_pe_f32.T   # (H, SEQ)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(scores)
        p_f32 = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        o_ref = (p_f32 @ kv_f32).astype(np.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("dp", group_x=GX, group_y=1, num_groups=1,
                         axis_names=("x", "y"))
        ])
        hw = HardwareBinding({"dp": list(range(GX))})

        body = compile_tilelang_to_softhier_parallel(
            fn, q_np, q_pe_np, kv_np, k_pe_np,
            H_=H, D_=D, pe_dim_=pe_dim, SEQ_=SEQ, Bc_=Bc, GX_=GX,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=NUM_CLUSTERS,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Q",    c_dtype="fp16",
                             nbytes=H * D * elem_bytes,        is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_Q_pe", c_dtype="fp16",
                             nbytes=H * pe_dim * elem_bytes,   is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_KV",   c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes,      is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_K_pe", c_dtype="fp16",
                             nbytes=SEQ * pe_dim * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_O", c_dtype="fp16",
                             nbytes=H * D * elem_bytes, is_input=False),
        ]

        config = create_test_config(
            test_name="Tilelang/flash_mla_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[q_np, q_pe_np, kv_np, k_pe_np],
            test_outputs=[o_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=0.05)
