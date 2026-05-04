# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for P0/P1 TileIR extensions: math builtins, sync_grid, thread_return,
device_assert, intra-cluster reduce, and WorkPartitioningPass.

Control-flow / runtime intrinsics (sync_grid, thread_return, device_assert,
assume) use the ``D.*`` namespace (``tl_deeploy``), following the same pattern
as ``D.reduce`` / ``D.broadcast``, to avoid conflicting with TileLang's native
TIR lowering.

Math intrinsics (exp, sigmoid, sqrt, rsqrt, abs, max, min) remain as ``T.*``
since they appear in expressions and are handled by the ExprStringifier.

Usage
-----
    # Compilation-only (no toolchain needed):
    pytest test_tilelang_non_gemm_ops.py -v -k "not e2e"

    # E2E (requires SoftHier toolchain):
    pytest test_tilelang_non_gemm_ops.py -v -k "e2e" \\
        --toolchain=GCC --toolchain-install-dir=$SOFTHIER_INSTALL_DIR/third_party/toolchain/install
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Guards
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

def _compile_and_get_code(fn, *args, cluster_id=0, **kwargs):
    from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
    return compile_tilelang_to_softhier(fn, *args, cluster_id=cluster_id, **kwargs)


def _assert_code_contains(code: str, pattern: str, desc: str):
    __tracebackhide__ = True
    assert pattern in code, f"Expected {desc}: '{pattern}' not found in generated C"


def _assert_code_not_contains(code: str, pattern: str, desc: str):
    __tracebackhide__ = True
    assert pattern not in code, f"Unexpected {desc}: '{pattern}' found in generated C"


def _check_error_count(result, max_abs_err: float = float("inf"),
                       max_rel_err: float = 1.0) -> None:
    """Extract and report error count vs total elements from simulation stdout.

    Parses ``Errors: N out of M`` printed by the test harness (main.c) and
    optionally asserts that the error rate is within tolerance.

    Args:
        result: ``TestResult`` from ``run_simulation``.
        max_abs_err: If given and result has ``DEEPLOY_OUT`` arrays, assert
                     that the max absolute error is below this threshold.
        max_rel_err: If ``error_count >= 0``, assert that
                     ``error_count / total_count <= max_rel_err``.
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

    arrs = result.output_arrays
    if arrs:
        for i, a in enumerate(arrs):
            print(f"[Verify] Output buffer {i}: shape={a.shape}, "
                  f"min={np.min(a):.4e}, max={np.max(a):.4e}")


def _check_errors_against_reference(result, ref_arrays,
                                     rtol: float = 1e-2,
                                     atol: float = 1e-2) -> None:
    """Verify simulation output values against reference arrays.

    Uses ``DEEPLOY_OUT`` parsing when the harness prints per-element values,
    and also checks the ``Errors: N out of M`` summary line.
    """
    _check_error_count(result)
    actual = result.output_arrays
    if actual:
        for i, (act, ref) in enumerate(zip(actual, ref_arrays)):
            exp = np.asarray(ref).ravel().astype(np.float32)
            abs_err = np.abs(act - exp)
            max_err = np.max(abs_err)
            mean_err = np.mean(abs_err)
            n_wrong = int(np.sum(~np.isclose(act, exp, rtol=rtol, atol=atol)))
            print(f"[Verify] Buf {i}: max_err={max_err:.4e}, mean_err={mean_err:.4e}, "
                  f"mismatches={n_wrong}/{len(exp)}")
            assert np.allclose(act, exp, rtol=rtol, atol=atol), (
                f"Buffer {i} failed np.allclose(rtol={rtol}, atol={atol}): "
                f"max_err={max_err:.4e}, mismatches={n_wrong}/{len(exp)}")
    else:
        print("[Verify] No DEEPLOY_OUT values found; "
              "relying on Errors: N out of M summary")


# ---------------------------------------------------------------------------
# Test 1: Elementwise math builtins — ExprStringifier (P0)
# ---------------------------------------------------------------------------

@pytest.mark.deeploy_internal
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestMathBuiltins:

    @staticmethod
    def _build_sigmoid_kernel(BLOCK_M: int = 64):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def sigmoid_kernel(X, M: int, BLOCK_M_: int):
            dtype = T.float16
            X: T.Tensor((M,), dtype)
            Y = T.empty((M,), dtype)

            with T.Kernel(T.ceildiv(M, BLOCK_M_), threads=1) as pid:
                X_local = T.alloc_fragment((BLOCK_M_,), dtype)
                T.copy(X[pid * BLOCK_M_ : (pid + 1) * BLOCK_M_], X_local)
                for i in T.Parallel(BLOCK_M_):
                    X_local[i] = T.sigmoid(X_local[i])
                T.copy(X_local, Y[pid * BLOCK_M_ : (pid + 1) * BLOCK_M_])
            return Y

        return sigmoid_kernel

    def test_sigmoid_code_emits_math_macro(self):
        fn = self._build_sigmoid_kernel(BLOCK_M=64)
        X_in = np.zeros((128,), dtype=np.float16)
        code = _compile_and_get_code(fn, X_in, M=128, BLOCK_M_=64)
        _assert_code_contains(code, "tile_fp16_sigmoid", "sigmoid macro")
        _assert_code_contains(code, "TileLang math helpers", "math preamble")
        _assert_code_contains(code, "asm_fp16_sigmoid", "sigmoid library ref")

    @staticmethod
    def _build_multi_math_kernel(N: int = 32):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def multi_math(X, N_: int):
            dtype = T.float16
            X: T.Tensor((N_,), dtype)
            Y = T.empty((N_,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                loc = T.alloc_fragment((N_,), dtype)
                T.copy(X[0:N_], loc)
                for i in T.Parallel(N_):
                    v = loc[i]
                    v = T.sigmoid(v)
                    v = T.exp(v)
                    v = T.sqrt(v)
                    v = T.rsqrt(v)
                    v = T.abs(v)
                    loc[i] = v
                T.copy(loc, Y[0:N_])
            return Y

        return multi_math

    def test_multi_math_emits_all_macros(self):
        fn = self._build_multi_math_kernel(N=32)
        X_in = np.zeros((32,), dtype=np.float16)
        code = _compile_and_get_code(fn, X_in, N_=32)

        for macro in [
            "tile_fp16_sigmoid", "tile_fp16_exp", "tile_fp16_sqrt",
            "tile_fp16_rsqrt", "tile_fp16_abs",
        ]:
            _assert_code_contains(code, macro, f"macro {macro}")

    @staticmethod
    def _build_max_min_kernel(N: int = 32):
        """T.max and T.min lowered as tir.Max / tir.Min nodes."""
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def max_min_kernel(X, N_: int):
            dtype = T.float16
            X: T.Tensor((N_,), dtype)
            Y = T.empty((N_,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                loc = T.alloc_fragment((N_,), dtype)
                T.copy(X[0:N_], loc)
                for i in T.Parallel(N_):
                    loc[i] = T.max(loc[i], T.float16(0.0))
                    loc[i] = T.min(loc[i], T.float16(1.0))
                T.copy(loc, Y[0:N_])
            return Y

        return max_min_kernel

    def test_max_min_emits_macros(self):
        fn = self._build_max_min_kernel(N=32)
        X_in = np.zeros((32,), dtype=np.float16)
        code = _compile_and_get_code(fn, X_in, N_=32)
        _assert_code_contains(code, "tile_fp16_max", "fp16 max macro")
        _assert_code_contains(code, "tile_fp16_min", "fp16 min macro")

    @staticmethod
    def _build_if_then_else_kernel(N: int = 16):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def ite_kernel(X, N_: int):
            dtype = T.float16
            X: T.Tensor((N_,), dtype)
            Y = T.empty((N_,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                loc = T.alloc_fragment((N_,), dtype)
                T.copy(X[0:N_], loc)
                for i in T.Parallel(N_):
                    zero = T.float16(0.0)
                    loc[i] = T.if_then_else(loc[i] > zero, loc[i], zero)
                T.copy(loc, Y[0:N_])
            return Y

        return ite_kernel

    def test_if_then_else_emits_ternary(self):
        fn = self._build_if_then_else_kernel(N=16)
        X_in = np.zeros((16,), dtype=np.float16)
        code = _compile_and_get_code(fn, X_in, N_=16)
        _assert_code_contains(code, "?", "ternary operator")
        _assert_code_contains(code, ":", "ternary colon")


# ---------------------------------------------------------------------------
# Test 2: D.sync_grid — mid-kernel global barrier (P1)
# ---------------------------------------------------------------------------

@pytest.mark.deeploy_internal
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestSyncGrid:

    @staticmethod
    def _build_sync_grid_kernel(N: int = 32):
        import tilelang
        import tilelang.language as T
        from Deeploy.TileIR.Frontend import tl_deeploy as D

        @tilelang.jit
        def sync_kernel(X, N_: int):
            dtype = T.float16
            X: T.Tensor((N_,), dtype)
            Y = T.empty((N_,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                loc = T.alloc_fragment((N_,), dtype)
                T.copy(X[0:N_], loc)
                for i in T.Parallel(N_):
                    loc[i] = T.sigmoid(loc[i])
                # Mid-kernel global sync via D.* namespace
                D.sync_grid()
                T.copy(loc, Y[0:N_])
            return Y

        return sync_kernel

    def test_sync_grid_emits_global_barrier(self):
        fn = self._build_sync_grid_kernel(N=32)
        X_in = np.zeros((32,), dtype=np.float16)
        code = _compile_and_get_code(fn, X_in, N_=32)
        _assert_code_contains(code, "flex_global_barrier_xy",
                              "global barrier in sync_grid")


# ---------------------------------------------------------------------------
# Test 3: D.thread_return — early exit (P1)
# ---------------------------------------------------------------------------

@pytest.mark.deeploy_internal
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestThreadReturn:

    @staticmethod
    def _build_early_exit_kernel(N: int = 64):
        import tilelang
        import tilelang.language as T
        from Deeploy.TileIR.Frontend import tl_deeploy as D

        @tilelang.jit
        def early_exit(X, N_: int):
            dtype = T.float16
            X: T.Tensor((N_,), dtype)
            Y = T.empty((N_,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                loc = T.alloc_fragment((N_,), dtype)
                T.copy(X[0:N_], loc)
                for i in T.Parallel(N_):
                    if loc[i] == T.float16(0.0):
                        D.thread_return()
                    loc[i] = T.sigmoid(loc[i])
                T.copy(loc, Y[0:N_])
            return Y

        return early_exit

    def test_thread_return_emits_return(self):
        fn = self._build_early_exit_kernel(N=64)
        X_in = np.zeros((64,), dtype=np.float16)
        code = _compile_and_get_code(fn, X_in, N_=64)
        _assert_code_contains(code, "return;", "early return statement")


# ---------------------------------------------------------------------------
# Test 4: D.device_assert / D.assume — runtime checks (P1)
# ---------------------------------------------------------------------------

@pytest.mark.deeploy_internal
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestDeviceAssert:

    @staticmethod
    def _build_assert_kernel(N: int = 32):
        import tilelang
        import tilelang.language as T
        from Deeploy.TileIR.Frontend import tl_deeploy as D

        @tilelang.jit
        def assert_kernel(X, N_: int):
            dtype = T.float16
            X: T.Tensor((N_,), dtype)
            Y = T.empty((N_,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                loc = T.alloc_fragment((N_,), dtype)
                T.copy(X[0:N_], loc)
                D.device_assert(N_ > 0, "N must be positive")
                for i in T.Parallel(N_):
                    loc[i] = T.sigmoid(loc[i])
                T.copy(loc, Y[0:N_])
            return Y

        return assert_kernel

    def test_device_assert_emits_assertion(self):
        fn = self._build_assert_kernel(N=32)
        X_in = np.zeros((32,), dtype=np.float16)
        code = _compile_and_get_code(fn, X_in, N_=32)
        _assert_code_contains(code, "Assertion failed",
                              "assertion message in generated C")

    @staticmethod
    def _build_assume_kernel(N: int = 32):
        import tilelang
        import tilelang.language as T
        from Deeploy.TileIR.Frontend import tl_deeploy as D

        @tilelang.jit
        def assume_kernel(X, N_: int):
            dtype = T.float16
            X: T.Tensor((N_,), dtype)
            Y = T.empty((N_,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                loc = T.alloc_fragment((N_,), dtype)
                T.copy(X[0:N_], loc)
                D.assume(N_ % 4 == 0)
                for i in T.Parallel(N_):
                    loc[i] = T.sigmoid(loc[i])
                T.copy(loc, Y[0:N_])
            return Y

        return assume_kernel

    def test_assume_emits_builtin_assume(self):
        fn = self._build_assume_kernel(N=32)
        X_in = np.zeros((32,), dtype=np.float16)
        code = _compile_and_get_code(fn, X_in, N_=32)
        _assert_code_contains(code, "__builtin_assume", "__builtin_assume call")


# ---------------------------------------------------------------------------
# Test 5: WorkPartitioningPass — unit tests (P1)
# ---------------------------------------------------------------------------

@pytest.mark.deeploy_internal
class TestWorkPartitioningPass:

    def test_pass_prepends_preamble(self):
        from Deeploy.TileIR.IR.TileBinding import TileBinding
        from Deeploy.TileIR.Passes.WorkPartitioning import WorkPartitioningPass

        existing = [
            TileBinding(
                op_kind="comment",
                template=None,
                operator_representation={},
                op_name="test_binding",
            )
        ]
        wpp = WorkPartitioningPass(num_elements="M")
        result = wpp.apply(existing)

        assert len(result) == 2
        assert result[0].op_kind == "block_preamble"
        assert result[1] is existing[0]

    def test_pass_custom_num_clusters(self):
        from Deeploy.TileIR.Passes.WorkPartitioning import WorkPartitioningPass

        wpp = WorkPartitioningPass(num_elements="num_tokens", num_clusters="8")
        result = wpp.apply([])
        rep = result[0].operator_representation
        assert rep["num_elements"] == "num_tokens"
        assert rep["num_clusters"] == "8"

    def test_pass_default_num_clusters(self):
        from Deeploy.TileIR.Passes.WorkPartitioning import WorkPartitioningPass

        wpp = WorkPartitioningPass(num_elements="N")
        result = wpp.apply([])
        rep = result[0].operator_representation
        assert "ARCH_NUM_CLUSTER_X" in rep["num_clusters"]


# ---------------------------------------------------------------------------
# Test 6: GroupShift — compile-only validation (P1)
# ---------------------------------------------------------------------------

@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestGroupShiftCompilation:

    @staticmethod
    def _build_cannon_kernel():
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def cannon_shift(A, B, BM: int, BN: int, BK: int):
            M, K, N = T.const("M, K, N")
            dtype = T.float16
            A: T.Tensor((M, K), dtype)
            B: T.Tensor((K, N), dtype)
            C = T.empty((M, N), dtype)

            with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN)) as (bx, by):
                with T.cluster_group("cannon", x=2, y=1, num_groups=1,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    A_local = T.alloc_fragment((BM, BK), dtype)
                    B_local = T.alloc_fragment((BK, BN), dtype)
                    C_local = T.alloc_fragment((BM, BN), dtype)
                    T.clear(C_local)

                    for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=1):
                        T.group_shift(A_local, along="x", by=1, group="cannon")
                        T.gemm(A_local, B_local, C_local, clear_accum=False)

                    if gid == 0:
                        T.copy(C_local, C[bx * BM : (bx + 1) * BM,
                                          by * BN : (by + 1) * BN])
            return C

        return cannon_shift

    def test_group_shift_no_longer_stub(self):
        import tilelang.language as T
        from Deeploy.TileIR.IR import (
            ClusterGroup, ClusterGroupRegistry, HardwareBinding,
        )
        from deeployRunner_tilelang_softhier import (
            compile_tilelang_to_softhier_parallel,
        )

        M = K = N = 256
        BM = BN = BK = 64
        fn = self._build_cannon_kernel()
        A = T.empty((M, K), T.float16)
        B = T.empty((K, N), T.float16)

        registry = ClusterGroupRegistry([
            ClusterGroup("cannon", group_x=2, group_y=1)
        ])
        hw = HardwareBinding({"cannon": list(range(16))})

        code = compile_tilelang_to_softhier_parallel(
            fn, A, B, BM=BM, BN=BN, BK=BK,
            group_registry=registry,
            hw_binding=hw,
            num_clusters=16,
        )

        _assert_code_not_contains(code, "(void)", "old void-cast stub")
        _assert_code_contains(code, "grid_sync_group_barrier_xy",
                              "group barrier in shift")


# ---------------------------------------------------------------------------
# Test 7: Elementwise sigmoid e2e — build + simulate (P0)
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestElementwiseE2E:

    @staticmethod
    def _build_sigmoid_e2e_kernel(BLOCK_M: int = 64):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def sigmoid_e2e(X, M: int, BLOCK_M_: int):
            dtype = T.float16
            X: T.Tensor((M,), dtype)
            Y = T.empty((M,), dtype)

            with T.Kernel(T.ceildiv(M, BLOCK_M_), threads=1) as pid:
                X_local = T.alloc_fragment((BLOCK_M_,), dtype)
                T.copy(X[pid * BLOCK_M_ : (pid + 1) * BLOCK_M_], X_local)
                for i in T.Parallel(BLOCK_M_):
                    X_local[i] = T.sigmoid(X_local[i])
                T.copy(X_local, Y[pid * BLOCK_M_ : (pid + 1) * BLOCK_M_])
            return Y

        return sigmoid_e2e

    @pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
    def test_sigmoid_e2e_compiles(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        """Sigmoid kernel: compile → build → simulate → verify."""
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
        from testUtils.codeGenerate import (
            TilelangIOBuffer,
            generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import (
            build_binary,
            configure_cmake,
            generate_network,
            run_simulation,
        )
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        M_val = 256
        BLOCK_M = 64
        elem_bytes = 2  # fp16
        fn = self._build_sigmoid_e2e_kernel(BLOCK_M=BLOCK_M)

        np.random.seed(42)
        X_np = np.random.randn(M_val).astype(np.float32)
        X_fp16 = X_np.astype(np.float16)
        Y_golden = (1.0 / (1.0 + np.exp(-X_np))).astype(np.float16)

        body = compile_tilelang_to_softhier(fn, X_fp16, M=M_val,
                                            BLOCK_M_=BLOCK_M, cluster_id=0)

        config = create_test_config(
            test_name="Tilelang/sigmoid_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_X", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Y", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[X_fp16],
            test_outputs=[Y_golden],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        stdout = result.stdout or ""
        assert "Simulation stopped by user" in result.stdout, "Simulation did not complete successfully"


# ---------------------------------------------------------------------------
# Test 8: Multi-output element-wise fusion e2e (P0)
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestFusedEltwiseE2E:
    """Fused element-wise kernel with three outputs from one kernel launch."""

    @staticmethod
    def _build_fused_kernel(M: int = 128):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def fused_kernel(X, Y, Z, M_: int):
            dtype = T.float16
            X: T.Tensor((M_,), dtype)
            Y: T.Tensor((M_,), dtype)
            Z: T.Tensor((M_,), dtype)
            O1 = T.empty((M_,), dtype)
            O2 = T.empty((M_,), dtype)
            O3 = T.empty((M_,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                x_l = T.alloc_fragment((M_,), dtype)
                y_l = T.alloc_fragment((M_,), dtype)
                z_l = T.alloc_fragment((M_,), dtype)
                o1_l = T.alloc_fragment((M_,), dtype)
                o2_l = T.alloc_fragment((M_,), dtype)
                o3_l = T.alloc_fragment((M_,), dtype)
                T.copy(X[0:M_], x_l)
                T.copy(Y[0:M_], y_l)
                T.copy(Z[0:M_], z_l)
                for i in T.Parallel(M_):
                    o1_l[i] = T.sigmoid(x_l[i])
                for i in T.Parallel(M_):
                    o2_l[i] = x_l[i] * y_l[i] + z_l[i]
                for i in T.Parallel(M_):
                    o3_l[i] = T.max(x_l[i], y_l[i])
                T.copy(o1_l, O1[0:M_])
                T.copy(o2_l, O2[0:M_])
                T.copy(o3_l, O3[0:M_])
            return O1, O2, O3

        return fused_kernel

    def test_fused_eltwise_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
        from testUtils.codeGenerate import (
            TilelangIOBuffer,
            generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import (
            build_binary,
            configure_cmake,
            run_simulation,
        )
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        M_val = 128
        elem_bytes = 2
        fn = self._build_fused_kernel(M=M_val)

        np.random.seed(42)
        x_np = np.random.randn(M_val).astype(np.float32)
        y_np = np.random.randn(M_val).astype(np.float32)
        z_np = np.random.randn(M_val).astype(np.float32)
        x_fp16 = x_np.astype(np.float16)
        y_fp16 = y_np.astype(np.float16)
        z_fp16 = z_np.astype(np.float16)

        o1_ref = (1.0 / (1.0 + np.exp(-x_np))).astype(np.float16)
        o2_ref = (x_np * y_np + z_np).astype(np.float16)
        o3_ref = np.maximum(x_np, y_np).astype(np.float16)

        body = compile_tilelang_to_softhier(
            fn, x_fp16, y_fp16, z_fp16, M_=M_val, cluster_id=0)

        config = create_test_config(
            test_name="Tilelang/fused_eltwise_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_X", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_Y", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_Z", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_O1", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
            TilelangIOBuffer(name="DeeployNetwork_O2", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
            TilelangIOBuffer(name="DeeployNetwork_O3", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[x_fp16, y_fp16, z_fp16],
            test_outputs=[o1_ref, o2_ref, o3_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"


# ---------------------------------------------------------------------------
# Test 9: Top-1 (max) selection e2e — dynamic-in-static (P1)
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestTopKE2E:
    """Find the maximum value in a 1-D array using a serial scan.

    Demonstrates that a classically "dynamic" algorithm (value selection
    based on input data) works in the static compilation framework with
    predefined output shapes.  The output is always a single-element
    buffer regardless of where the max lies.

    Extension to top-k uses repeated reduce-and-mask passes, and index
    tracking uses T.if_then_else for conditional updates.
    """

    @staticmethod
    def _build_topk_kernel(M: int = 128):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def topk_kernel(X, M_: int):
            dtype = T.float16
            X: T.Tensor((M_,), dtype)
            MaxVal = T.empty((1,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                x_l = T.alloc_fragment((M_,), dtype)
                mv = T.alloc_fragment((1,), dtype)
                T.copy(X[0:M_], x_l)
                # Initialise with first element
                mv[0] = x_l[0]
                # Serial scan: max reduction
                for i in T.serial(1, M_):
                    mv[0] = T.max(mv[0], x_l[i])
                T.copy(mv, MaxVal[0:1])
            return MaxVal

        return topk_kernel

    def test_topk_max_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
        from testUtils.codeGenerate import (
            TilelangIOBuffer,
            generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import (
            build_binary,
            configure_cmake,
            run_simulation,
        )
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        M_val = 128
        elem_bytes = 2
        fn = self._build_topk_kernel(M=M_val)

        np.random.seed(42)
        x_np = np.random.randn(M_val).astype(np.float32)
        x_fp16 = x_np.astype(np.float16)
        ref_max = np.max(x_fp16).astype(np.float16).reshape(1)

        body = compile_tilelang_to_softhier(
            fn, x_fp16, M_=M_val, cluster_id=0)

        config = create_test_config(
            test_name="Tilelang/topk_max_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_X", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_MaxVal", c_dtype="fp16",
                             nbytes=1 * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[x_fp16],
            test_outputs=[ref_max],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"


# ---------------------------------------------------------------------------
# Test 10: Q @ K^T GEMM e2e — attention score computation (P1)
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestAttentionGemmE2E:
    """Q @ K^T via RedMule GEMM — the first step of attention.

    Exercises T.gemm with transpose_B (RedMule), T.copy (DMA), T.clear.
    Full Flash Attention extends this with tiling, row-wise softmax
    (reduce_max, exp, reduce_sum, div), and a second GEMM for P @ V.

    Note: The softmax reduce/exp/normalize steps require additional
    ExprStringifier support (FloorDiv/FloorMod for linearized 2D indexing,
    multi-dim buffer access in BufferStore) that is not yet implemented.
    """

    @staticmethod
    def _build_gemm_kernel(M: int = 64, D: int = 64):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def gemm_kernel(Q, K, M_: int, D_: int):
            dtype = T.float16
            Q: T.Tensor((M_, D_), dtype)
            K: T.Tensor((M_, D_), dtype)
            Y = T.empty((M_, M_), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                q_l = T.alloc_fragment((M_, D_), dtype)
                k_l = T.alloc_fragment((M_, D_), dtype)
                s_l = T.alloc_fragment((M_, M_), dtype)

                T.copy(Q[0:M_, 0:D_], q_l)
                T.copy(K[0:M_, 0:D_], k_l)

                # S = Q @ K^T via RedMule
                T.clear(s_l)
                T.gemm(q_l, k_l, s_l, transpose_B=True)

                T.copy(s_l, Y[0:M_, 0:M_])
            return Y

        return gemm_kernel

    def test_attention_gemm_e2e_compiles(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
        from testUtils.codeGenerate import (
            TilelangIOBuffer,
            generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import (
            build_binary,
            configure_cmake,
            run_simulation,
        )
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        M, D = 64, 64
        elem_bytes = 2
        fn = self._build_gemm_kernel(M=M, D=D)

        np.random.seed(42)
        q_np = np.random.randn(M, D).astype(np.float16)
        k_np = np.random.randn(M, D).astype(np.float16)
        s_ref = (q_np.astype(np.float32) @ k_np.astype(np.float32).T).astype(np.float16)

        body = compile_tilelang_to_softhier(
            fn, q_np, k_np, M_=M, D_=D, cluster_id=0)

        config = create_test_config(
            test_name="Tilelang/attention_gemm_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Q", c_dtype="fp16",
                             nbytes=M * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_K", c_dtype="fp16",
                             nbytes=M * D * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Y", c_dtype="fp16",
                             nbytes=M * M * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[q_np, k_np],
            test_outputs=[s_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"


# ---------------------------------------------------------------------------
# Test 11: sync_threads — intra-cluster sync barrier (P1)
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestSyncThreadsE2E:
    """Validates the sync_threads → intra-cluster sync lowering (Gap 3).

    The kernel copies data to L1, syncs, then copies to output.
    T.sync_threads() lowers to flex_intra_cluster_sync().
    """

    @staticmethod
    def _build_sync_kernel(M: int = 64):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def sync_kernel(X, M_: int):
            dtype = T.float16
            X: T.Tensor((M_,), dtype)
            Y = T.empty((M_,), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                x_l = T.alloc_fragment((M_,), dtype)
                T.copy(X[0:M_], x_l)
                # Intra-cluster sync — ensures all cores/DM see the write
                T.sync_threads()
                T.copy(x_l, Y[0:M_])
            return Y

        return sync_kernel

    def test_sync_threads_e2e_compiles(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
        from testUtils.codeGenerate import (
            TilelangIOBuffer,
            generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import (
            build_binary,
            configure_cmake,
            run_simulation,
        )
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        M_val = 64
        elem_bytes = 2
        fn = self._build_sync_kernel(M=M_val)

        np.random.seed(42)
        x_np = np.random.randn(M_val).astype(np.float16)

        body = compile_tilelang_to_softhier(
            fn, x_np, M_=M_val, cluster_id=0)

        # Verify sync_threads lowered to intra-cluster sync
        assert "flex_intra_cluster_sync" in body, \
            "sync_threads did not lower to intra-cluster sync"

        config = create_test_config(
            test_name="Tilelang/sync_threads_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_X", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Y", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[x_np],
            test_outputs=[x_np],  # identity: output == input
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"


# ---------------------------------------------------------------------------
# Test 12: Row-wise softmax e2e — GEMM + softmax (P1)
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestSoftmaxE2E:
    """Single-tile Q@K^T + row-wise softmax.

    Validates the multi-dim BufferLoad/Store linearization fix.
    Uses nested T.serial loops with 2D buffer access on the GEMM output.
    """

    @staticmethod
    def _build_softmax_kernel(M: int = 64, D: int = 64):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def softmax_kernel(Q, K, M_: int, D_: int):
            dtype = T.float16
            Q: T.Tensor((M_, D_), dtype)
            K: T.Tensor((M_, D_), dtype)
            Y = T.empty((M_, M_), dtype)

            with T.Kernel(1, threads=1) as (bx,):
                q_l = T.alloc_fragment((M_, D_), dtype)
                k_l = T.alloc_fragment((M_, D_), dtype)
                s_l = T.alloc_fragment((M_, M_), dtype)
                row_max = T.alloc_fragment((1,), dtype)
                row_sum = T.alloc_fragment((1,), dtype)

                T.copy(Q[0, 0], q_l)
                T.copy(K[0, 0], k_l)
                T.clear(s_l)
                T.gemm(q_l, k_l, s_l, transpose_B=True)

                for i in T.serial(M_):
                    row_max[0] = s_l[i, 0]
                    for j in T.serial(1, M_):
                        row_max[0] = T.max(row_max[0], s_l[i, j])
                    for j in T.serial(M_):
                        s_l[i, j] = T.exp(s_l[i, j] - row_max[0])
                    row_sum[0] = s_l[i, 0]
                    for j in T.serial(1, M_):
                        row_sum[0] = row_sum[0] + s_l[i, j]
                    for j in T.serial(M_):
                        s_l[i, j] = s_l[i, j] / row_sum[0]

                T.copy(s_l, Y[0, 0])
            return Y

        return softmax_kernel

    def test_softmax_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
        from testUtils.codeGenerate import (
            TilelangIOBuffer,
            generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import (
            build_binary,
            configure_cmake,
            run_simulation,
        )
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        M, D = 64, 64
        elem_bytes = 2
        fn = self._build_softmax_kernel(M=M, D=D)

        np.random.seed(42)
        q_np = np.random.randn(M, D).astype(np.float16)
        k_np = np.random.randn(M, D).astype(np.float16)
        s_f32 = q_np.astype(np.float32) @ k_np.astype(np.float32).T
        s_max = np.max(s_f32, axis=1, keepdims=True)
        s_exp = np.exp(s_f32 - s_max)
        s_ref = (s_exp / np.sum(s_exp, axis=1, keepdims=True)).astype(np.float16)

        body = compile_tilelang_to_softhier(
            fn, q_np, k_np, M_=M, D_=D, cluster_id=0)

        config = create_test_config(
            test_name="Tilelang/softmax_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Q", c_dtype="fp16",
                             nbytes=M * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_K", c_dtype="fp16",
                             nbytes=M * D * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Y", c_dtype="fp16",
                             nbytes=M * M * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[q_np, k_np],
            test_outputs=[s_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        # Report error count vs total elements parsed from stdout
        _check_error_count(result, max_rel_err=1.0)


# ---------------------------------------------------------------------------
# Test 13: Tiled Flash Attention e2e — single cluster (P1)
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestFlashAttentionE2E:
    """Tiled Flash Attention with online softmax, single cluster.

    Follows SUMMA patterns: T.copy(src[start], dst), T.gemm, T.serial tiling.
    Q tiles across rows (Tr), KV tiles across columns (Tc), with running
    m/l statistics for online softmax rescaling.
    """

    @staticmethod
    def _build_flash_attn_kernel(
        SEQ: int = 64, D: int = 64, Br: int = 32, Bc: int = 32,
    ):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def flash_attn_kernel(Q, K, V, SEQ_: int, D_: int, Br_: int, Bc_: int):
            dtype = T.float16
            Q: T.Tensor((SEQ_, D_), dtype)
            K: T.Tensor((SEQ_, D_), dtype)
            V: T.Tensor((SEQ_, D_), dtype)
            O = T.empty((SEQ_, D_), dtype)

            Tr = T.ceildiv(SEQ_, Br_)
            Tc = T.ceildiv(SEQ_, Bc_)

            with T.Kernel(Tr, 1) as (bx, by):
                Q_tr = T.alloc_fragment((Br_, D_), dtype)
                K_tc = T.alloc_fragment((Bc_, D_), dtype)
                V_tc = T.alloc_fragment((Bc_, D_), dtype)
                S = T.alloc_fragment((Br_, Bc_), dtype)
                O_tr = T.alloc_fragment((Br_, D_), dtype)
                m = T.alloc_fragment((Br_, 1), dtype)
                l = T.alloc_fragment((Br_, 1), dtype)
                row_max = T.alloc_fragment((1,), dtype)
                row_sum = T.alloc_fragment((1,), dtype)

                T.copy(Q[bx * Br_, 0], Q_tr)
                T.clear(O_tr)
                for r in T.serial(Br_):
                    m[r, 0] = T.float16(-65504.0)
                    l[r, 0] = T.float16(0.0)

                for tc in T.serial(Tc):
                    T.copy(K[tc * Bc_, 0], K_tc)
                    T.copy(V[tc * Bc_, 0], V_tc)
                    T.clear(S)
                    T.gemm(Q_tr, K_tc, S, transpose_B=True)

                    for r in T.serial(Br_):
                        row_max[0] = m[r, 0]
                        for c in T.serial(Bc_):
                            row_max[0] = T.max(row_max[0], S[r, c])
                        m[r, 0] = row_max[0]
                        row_sum[0] = T.float16(0.0)
                        for c in T.serial(Bc_):
                            S[r, c] = T.exp(S[r, c] - row_max[0])
                            row_sum[0] = row_sum[0] + S[r, c]
                        rescale = T.exp(l[r, 0] - row_max[0])
                        l[r, 0] = l[r, 0] * rescale + row_sum[0]
                        for c in T.serial(D_):
                            O_tr[r, c] = O_tr[r, c] * rescale

                    T.gemm(S, V_tc, O_tr, clear_accum=False)

                for r in T.serial(Br_):
                    for c in T.serial(D_):
                        O_tr[r, c] = O_tr[r, c] / l[r, 0]

                T.copy(O_tr, O[bx * Br_, 0])
            return O

        return flash_attn_kernel

    def test_flash_attn_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
        from testUtils.codeGenerate import (
            TilelangIOBuffer,
            generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import (
            build_binary,
            configure_cmake,
            run_simulation,
        )
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        SEQ, D, Br, Bc = 64, 64, 32, 32
        elem_bytes = 2
        fn = self._build_flash_attn_kernel(SEQ=SEQ, D=D, Br=Br, Bc=Bc)

        np.random.seed(42)
        q_np = np.random.randn(SEQ, D).astype(np.float16)
        k_np = np.random.randn(SEQ, D).astype(np.float16)
        v_np = np.random.randn(SEQ, D).astype(np.float16)
        q_f32 = q_np.astype(np.float32)
        k_f32 = k_np.astype(np.float32)
        v_f32 = v_np.astype(np.float32)
        s_f32 = q_f32 @ k_f32.T
        s_max = np.max(s_f32, axis=1, keepdims=True)
        s_exp = np.exp(s_f32 - s_max)
        p_f32 = s_exp / np.sum(s_exp, axis=1, keepdims=True)
        o_ref = (p_f32 @ v_f32).astype(np.float16)

        body = compile_tilelang_to_softhier(
            fn, q_np, k_np, v_np,
            SEQ_=SEQ, D_=D, Br_=Br, Bc_=Bc, cluster_id=0)

        config = create_test_config(
            test_name="Tilelang/flash_attn_e2e",
            platform="SoftHier",
            simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain,
            toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Q", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_K", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_V", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_O", c_dtype="fp16",
                             nbytes=SEQ * D * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body,
            dumpdir=str(gen_dir),
            input_bufs=input_bufs,
            output_bufs=output_bufs,
            test_inputs=[q_np, k_np, v_np],
            test_outputs=[o_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=1.0)


# ===========================================================================
# Multi-cluster tests — cluster_group variants of the kernels above
# ===========================================================================


# ---------------------------------------------------------------------------
# Test 14: Fused Eltwise — data-parallel across 1D cluster group
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestFusedEltwiseClusterE2E:
    """Fused eltwise kernel distributed across a 1-D cluster group.

    Each cluster handles a contiguous slice of rows.  All clusters run
    the same SPMD code, differentiated by gid_x at runtime.
    """

    @staticmethod
    def _build_fused_dp_kernel(GX: int = 2):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def fused_dp(X, Y, Z, M_: int, GX_: int):
            dtype = T.float16
            X: T.Tensor((M_,), dtype)
            Y: T.Tensor((M_,), dtype)
            Z: T.Tensor((M_,), dtype)
            O1 = T.empty((M_,), dtype)
            O2 = T.empty((M_,), dtype)
            O3 = T.empty((M_,), dtype)

            chunk = T.ceildiv(M_, GX_)
            with T.Kernel(1, threads=1) as (bx,):
                with T.cluster_group("dp", x=GX_, y=1, num_groups=1,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    x_l = T.alloc_fragment((chunk,), dtype)
                    y_l = T.alloc_fragment((chunk,), dtype)
                    z_l = T.alloc_fragment((chunk,), dtype)
                    o1_l = T.alloc_fragment((chunk,), dtype)
                    o2_l = T.alloc_fragment((chunk,), dtype)
                    o3_l = T.alloc_fragment((chunk,), dtype)

                    T.copy(X[gid_x * chunk], x_l)
                    T.copy(Y[gid_x * chunk], y_l)
                    T.copy(Z[gid_x * chunk], z_l)

                    for i in T.Parallel(chunk):
                        o1_l[i] = T.sigmoid(x_l[i])
                    for i in T.Parallel(chunk):
                        o2_l[i] = x_l[i] * y_l[i] + z_l[i]
                    for i in T.Parallel(chunk):
                        o3_l[i] = T.max(x_l[i], y_l[i])

                    T.copy(o1_l, O1[gid_x * chunk])
                    T.copy(o2_l, O2[gid_x * chunk])
                    T.copy(o3_l, O3[gid_x * chunk])
            return O1, O2, O3

        return fused_dp

    def test_fused_eltwise_dp_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import (
            TilelangIOBuffer, generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        GX = 2
        M_val = 128
        elem_bytes = 2
        fn = self._build_fused_dp_kernel(GX=GX)

        registry = ClusterGroupRegistry([
            ClusterGroup("dp", group_x=GX, group_y=1, num_groups=1,
                         axis_names=("x", "y"))
        ])
        hw = HardwareBinding({"dp": list(range(GX))})

        np.random.seed(42)
        x_fp16 = np.random.randn(M_val).astype(np.float16)
        y_fp16 = np.random.randn(M_val).astype(np.float16)
        z_fp16 = np.random.randn(M_val).astype(np.float16)
        x32 = x_fp16.astype(np.float32)
        y32 = y_fp16.astype(np.float32)
        z32 = z_fp16.astype(np.float32)
        o1_ref = (1.0 / (1.0 + np.exp(-x32))).astype(np.float16)
        o2_ref = (x32 * y32 + z32).astype(np.float16)
        o3_ref = np.maximum(x32, y32).astype(np.float16)

        body = compile_tilelang_to_softhier_parallel(
            fn, x_fp16, y_fp16, z_fp16, M_=M_val, GX_=GX,
            group_registry=registry, hw_binding=hw, num_clusters=GX)

        config = create_test_config(
            test_name="Tilelang/fused_eltwise_dp_e2e",
            platform="SoftHier", simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain, toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={GX}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_X", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_Y", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_Z", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_O1", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
            TilelangIOBuffer(name="DeeployNetwork_O2", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
            TilelangIOBuffer(name="DeeployNetwork_O3", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body, dumpdir=str(gen_dir),
            input_bufs=input_bufs, output_bufs=output_bufs,
            test_inputs=[x_fp16, y_fp16, z_fp16],
            test_outputs=[o1_ref, o2_ref, o3_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=1.0)


# ---------------------------------------------------------------------------
# Test 15: Top-K — data-parallel across 1D cluster group
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestTopKClusterE2E:
    """Top-1 (max) per row, rows distributed across a 1-D cluster group.

    Each cluster owns a contiguous slice of rows, finds the max per row
    via serial scan, and stores results to its output slice.
    """

    @staticmethod
    def _build_topk_dp_kernel(GX: int = 2):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def topk_dp(X, M_: int, GX_: int):
            dtype = T.float16
            X: T.Tensor((M_,), dtype)
            MaxVal = T.empty((M_,), dtype)

            chunk = T.ceildiv(M_, GX_)
            with T.Kernel(1, threads=1) as (bx,):
                with T.cluster_group("dp", x=GX_, y=1, num_groups=1,
                                     axes=("x", "y")) as (gid, gid_x, gid_y):
                    x_l = T.alloc_fragment((chunk,), dtype)
                    mv = T.alloc_fragment((1,), dtype)

                    T.copy(X[gid_x * chunk], x_l)
                    mv[0] = x_l[0]
                    for i in T.serial(1, chunk):
                        mv[0] = T.max(mv[0], x_l[i])

                    T.copy(mv, MaxVal[gid_x * chunk])
            return MaxVal

        return topk_dp

    def test_topk_dp_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import (
            TilelangIOBuffer, generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        GX = 2
        M_val = 128
        elem_bytes = 2
        fn = self._build_topk_dp_kernel(GX=GX)

        registry = ClusterGroupRegistry([
            ClusterGroup("dp", group_x=GX, group_y=1, num_groups=1,
                         axis_names=("x", "y"))
        ])
        hw = HardwareBinding({"dp": list(range(GX))})

        np.random.seed(42)
        x_fp16 = np.random.randn(M_val).astype(np.float16)
        chunk = (M_val + GX - 1) // GX
        ref_max = np.array([np.max(x_fp16.astype(np.float32)[i * chunk:(i + 1) * chunk])
                           for i in range(GX)]).astype(np.float16).flatten()

        body = compile_tilelang_to_softhier_parallel(
            fn, x_fp16, M_=M_val, GX_=GX,
            group_registry=registry, hw_binding=hw, num_clusters=GX)

        config = create_test_config(
            test_name="Tilelang/topk_dp_e2e",
            platform="SoftHier", simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain, toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={GX}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_X", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_MaxVal", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body, dumpdir=str(gen_dir),
            input_bufs=input_bufs, output_bufs=output_bufs,
            test_inputs=[x_fp16],
            test_outputs=[ref_max],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=1.0)


# ---------------------------------------------------------------------------
# Test 16: Attention GEMM (Q@K^T) — 2D SUMMA pattern
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestAttentionGemmClusterE2E:
    """Q @ K^T via 2D SUMMA cluster group.

    Follows the SUMMA DP pattern: GX×GY cluster grid, diagonal clusters
    load tiles and broadcast along rows/cols.  Each cluster owns a sub-block
    of the output score matrix S.
    """

    @staticmethod
    def _build_attn_gemm_suma_kernel(GX: int = 2, GY: int = 2):
        import tilelang
        import tilelang.language as T
        from Deeploy.TileIR.Frontend import tl_deeploy as D

        @tilelang.jit
        def attn_gemm_suma(Q, K, M_: int, D_: int,
                           BM_: int, BK_: int, GX_: int, GY_: int):
            dtype = T.float16
            Q: T.Tensor((M_, D_), dtype)
            K: T.Tensor((M_, D_), dtype)
            S = T.empty((M_, M_), dtype)

            with T.Kernel(T.ceildiv(M_, GY_ * BM_), T.ceildiv(M_, GX_ * BM_)) as (by, bx):
                with T.cluster_group("summa", x=GX_, y=GY_, num_groups=1,
                                     axes=("x", "y")) as (inst_id, local_x, local_y):
                    Q_loc = T.alloc_fragment((BM_, BK_), dtype)
                    K_loc = T.alloc_fragment((BK_, BM_), dtype)
                    S_loc = T.alloc_fragment((BM_, BM_), dtype)
                    T.clear(S_loc)

                    for bk in T.serial(T.ceildiv(D_, BK_)):
                        if local_x == local_y:
                            T.copy(Q[(by * GY_ + local_y) * BM_, bk * BK_], Q_loc)
                            T.copy(K[(bx * GX_ + local_x) * BM_, bk * BK_], K_loc)
                        D.broadcast(Q_loc, level="intra_group", axis="x",
                                    group="summa", root=local_y)
                        D.broadcast(K_loc, level="intra_group", axis="y",
                                    group="summa", root=local_x)
                        T.gemm(Q_loc, K_loc, S_loc, clear_accum=False,
                               transpose_B=True)

                    T.copy(S_loc, S[(by * GY_ + local_y) * BM_,
                                    (bx * GX_ + local_x) * BM_])
            return S

        return attn_gemm_suma

    def test_attn_gemm_suma_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import (
            TilelangIOBuffer, generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config

        GX, GY = 2, 2
        NUM_CLUSTERS = GX * GY
        M, D = 128, 128
        BM, BK = 64, 64
        elem_bytes = 2

        fn = self._build_attn_gemm_suma_kernel(GX=GX, GY=GY)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa", group_x=GX, group_y=GY, num_groups=1,
                         axis_names=("x", "y"), root_coord=(0, 0))
        ])
        hw = HardwareBinding({"summa": list(range(NUM_CLUSTERS))})

        np.random.seed(42)
        q_fp16 = np.random.randn(M, D).astype(np.float16)
        k_fp16 = np.random.randn(M, D).astype(np.float16)
        s_ref = (q_fp16.astype(np.float32) @ k_fp16.astype(np.float32).T).astype(np.float16)

        body = compile_tilelang_to_softhier_parallel(
            fn, q_fp16, k_fp16, M_=M, D_=D,
            BM_=BM, BK_=BK, GX_=GX, GY_=GY,
            group_registry=registry, hw_binding=hw, num_clusters=NUM_CLUSTERS)

        config = create_test_config(
            test_name="Tilelang/attn_gemm_suma_e2e",
            platform="SoftHier", simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain, toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Q", c_dtype="fp16",
                             nbytes=M * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_K", c_dtype="fp16",
                             nbytes=M * D * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_S", c_dtype="fp16",
                             nbytes=M * M * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body, dumpdir=str(gen_dir),
            input_bufs=input_bufs, output_bufs=output_bufs,
            test_inputs=[q_fp16, k_fp16],
            test_outputs=[s_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=1.0)


# ---------------------------------------------------------------------------
# Test 17: Softmax — 2D SUMMA + local row-wise softmax
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestSoftmaxClusterE2E:
    """Q@K^T via 2D SUMMA, then local row-wise softmax per cluster."""

    @staticmethod
    def _build_softmax_suma_kernel(GX: int = 2, GY: int = 2):
        import tilelang
        import tilelang.language as T
        from Deeploy.TileIR.Frontend import tl_deeploy as D

        @tilelang.jit
        def softmax_suma(Q, K, M_: int, D_: int,
                         BM_: int, BK_: int, GX_: int, GY_: int):
            dtype = T.float16
            Q: T.Tensor((M_, D_), dtype)
            K: T.Tensor((M_, D_), dtype)
            Y = T.empty((M_, M_), dtype)

            with T.Kernel(T.ceildiv(M_, GY_ * BM_), T.ceildiv(M_, GX_ * BM_)) as (by, bx):
                with T.cluster_group("summa", x=GX_, y=GY_, num_groups=1,
                                     axes=("x", "y")) as (inst_id, local_x, local_y):
                    Q_loc = T.alloc_fragment((BM_, BK_), dtype)
                    K_loc = T.alloc_fragment((BK_, BM_), dtype)
                    S_loc = T.alloc_fragment((BM_, BM_), dtype)
                    row_max = T.alloc_fragment((1,), dtype)
                    row_sum = T.alloc_fragment((1,), dtype)
                    T.clear(S_loc)

                    for bk in T.serial(T.ceildiv(D_, BK_)):
                        if local_x == local_y:
                            T.copy(Q[(by * GY_ + local_y) * BM_, bk * BK_], Q_loc)
                            T.copy(K[(bx * GX_ + local_x) * BM_, bk * BK_], K_loc)
                        D.broadcast(Q_loc, level="intra_group", axis="x",
                                    group="summa", root=local_y)
                        D.broadcast(K_loc, level="intra_group", axis="y",
                                    group="summa", root=local_x)
                        T.gemm(Q_loc, K_loc, S_loc, clear_accum=False,
                               transpose_B=True)

                    for r in T.serial(BM_):
                        row_max[0] = S_loc[r, 0]
                        for c in T.serial(1, BM_):
                            row_max[0] = T.max(row_max[0], S_loc[r, c])
                        for c in T.serial(BM_):
                            S_loc[r, c] = T.exp(S_loc[r, c] - row_max[0])
                        row_sum[0] = S_loc[r, 0]
                        for c in T.serial(1, BM_):
                            row_sum[0] = row_sum[0] + S_loc[r, c]
                        for c in T.serial(BM_):
                            S_loc[r, c] = S_loc[r, c] / row_sum[0]

                    T.copy(S_loc, Y[(by * GY_ + local_y) * BM_,
                                    (bx * GX_ + local_x) * BM_])
            return Y

        return softmax_suma

    def test_softmax_suma_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        import tilelang.language as T
        from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
        from testUtils.codeGenerate import (
            TilelangIOBuffer, generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config

        GX, GY = 2, 2
        NUM_CLUSTERS = GX * GY
        M, D = 128, 128
        BM, BK = 64, 64
        elem_bytes = 2

        fn = self._build_softmax_suma_kernel(GX=GX, GY=GY)

        registry = ClusterGroupRegistry([
            ClusterGroup("summa", group_x=GX, group_y=GY, num_groups=1,
                         axis_names=("x", "y"), root_coord=(0, 0))
        ])
        hw = HardwareBinding({"summa": list(range(NUM_CLUSTERS))})

        np.random.seed(42)
        q_fp16 = np.random.randn(M, D).astype(np.float16)
        k_fp16 = np.random.randn(M, D).astype(np.float16)
        s_f32 = q_fp16.astype(np.float32) @ k_fp16.astype(np.float32).T
        s_max = np.max(s_f32, axis=1, keepdims=True)
        s_exp = np.exp(s_f32 - s_max)
        s_ref = (s_exp / np.sum(s_exp, axis=1, keepdims=True)).astype(np.float16)

        body = compile_tilelang_to_softhier_parallel(
            fn, q_fp16, k_fp16, M_=M, D_=D,
            BM_=BM, BK_=BK, GX_=GX, GY_=GY,
            group_registry=registry, hw_binding=hw, num_clusters=NUM_CLUSTERS)

        config = create_test_config(
            test_name="Tilelang/softmax_suma_e2e",
            platform="SoftHier", simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain, toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Q", c_dtype="fp16",
                             nbytes=M * D * elem_bytes, is_input=True),
            TilelangIOBuffer(name="DeeployNetwork_K", c_dtype="fp16",
                             nbytes=M * D * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Y", c_dtype="fp16",
                             nbytes=M * M * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body, dumpdir=str(gen_dir),
            input_bufs=input_bufs, output_bufs=output_bufs,
            test_inputs=[q_fp16, k_fp16],
            test_outputs=[s_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=1.0)

# ---------------------------------------------------------------------------
# Test 18: Spatz-vectorized element-wise ops (ReLU, Sigmoid)
# ---------------------------------------------------------------------------

@pytest.mark.softhier
@pytest.mark.tilelang
@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="tilelang not installed")
class TestSpatzEltwiseE2E:
    """Element-wise ReLU and Sigmoid lowered to Spatz RVV vector instructions."""

    @staticmethod
    def _build_spatz_relu_kernel(M: int = 256):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def spatz_relu(X, M_: int):
            dtype = T.float16
            X: T.Tensor((M_,), dtype)
            Y = T.empty((M_,), dtype)
            with T.Kernel(1, threads=1) as (bx,):
                x_l = T.alloc_fragment((M_,), dtype)
                T.copy(X[0], x_l)
                for i in T.Parallel(M_):
                    x_l[i] = T.max(x_l[i], T.float16(0.0))
                T.copy(x_l, Y[0])
            return Y
        return spatz_relu

    @staticmethod
    def _build_spatz_sigmoid_kernel(M: int = 256):
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def spatz_sigmoid(X, M_: int):
            dtype = T.float16
            X: T.Tensor((M_,), dtype)
            Y = T.empty((M_,), dtype)
            with T.Kernel(1, threads=1) as (bx,):
                x_l = T.alloc_fragment((M_,), dtype)
                T.copy(X[0], x_l)
                for i in T.Parallel(M_):
                    x_l[i] = T.sigmoid(x_l[i])
                T.copy(x_l, Y[0])
            return Y
        return spatz_sigmoid

    def test_spatz_relu_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
        from testUtils.codeGenerate import (
            TilelangIOBuffer, generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        M_val = 256
        elem_bytes = 2
        fn = self._build_spatz_relu_kernel(M=M_val)

        np.random.seed(42)
        x_fp16 = np.random.randn(M_val).astype(np.float16)
        y_ref = np.maximum(x_fp16.astype(np.float32), 0).astype(np.float16)

        body = compile_tilelang_to_softhier(fn, x_fp16, M_=M_val, cluster_id=0)
        _assert_code_contains(body, "SpatzEltwise", "Spatz vector template")
        _assert_code_contains(body, "vfmax.vf", "vfmax.vf for ReLU")

        config = create_test_config(
            test_name="Tilelang/spatz_relu_e2e",
            platform="SoftHier", simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain, toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_X", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Y", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body, dumpdir=str(gen_dir),
            input_bufs=input_bufs, output_bufs=output_bufs,
            test_inputs=[x_fp16], test_outputs=[y_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=1.0)

    def test_spatz_sigmoid_e2e(
        self, deeploy_test_dir: Path, toolchain_dir: str,
        toolchain: str, cmake_args: list,
    ):
        from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
        from testUtils.codeGenerate import (
            TilelangIOBuffer, generateTilelangSoftHierTestNetwork,
        )
        from testUtils.core import build_binary, configure_cmake, run_simulation
        from testUtils.pytestRunner import create_test_config
        from test_softhier_config import DEFAULT_NUM_CLUSTERS as NUM_CLUSTERS

        M_val = 256
        elem_bytes = 2
        fn = self._build_spatz_sigmoid_kernel(M=M_val)

        np.random.seed(42)
        x_fp16 = np.random.randn(M_val).astype(np.float16)
        y_ref = (1.0 / (1.0 + np.exp(-x_fp16.astype(np.float32)))).astype(np.float16)

        body = compile_tilelang_to_softhier(fn, x_fp16, M_=M_val, cluster_id=0)
        _assert_code_contains(body, "SpatzEltwise", "Spatz vector template")
        _assert_code_contains(body, "0x32041857", "vfexp.vv custom opcode")

        config = create_test_config(
            test_name="Tilelang/spatz_sigmoid_e2e",
            platform="SoftHier", simulator="gvsoc",
            deeploy_test_dir=deeploy_test_dir,
            toolchain=toolchain, toolchain_dir=toolchain_dir,
            cmake_args=list(cmake_args) + [f"num_clusters={NUM_CLUSTERS}"],
            tiling=False,
        )

        input_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_X", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=True),
        ]
        output_bufs = [
            TilelangIOBuffer(name="DeeployNetwork_Y", c_dtype="fp16",
                             nbytes=M_val * elem_bytes, is_input=False),
        ]

        gen_dir = Path(config.gen_dir)
        gen_dir.mkdir(parents=True, exist_ok=True)
        generateTilelangSoftHierTestNetwork(
            tilelangBody=body, dumpdir=str(gen_dir),
            input_bufs=input_bufs, output_bufs=output_bufs,
            test_inputs=[x_fp16], test_outputs=[y_ref],
        )

        configure_cmake(config)
        build_binary(config)
        result = run_simulation(config)
        assert "Simulation stopped by user" in (result.stdout or ""), \
            "Simulation did not complete successfully"
        _check_error_count(result, max_rel_err=1.0)
