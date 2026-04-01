# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone pytest integration for TileLang -> SoftHier.

This module validates two steps:
1) TileLang live JIT -> SoftHier C code generation.
2) Compilation of the generated ``Network.c`` in the Deeploy SoftHier CMake flow.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
from test_softhier_config import DEFAULT_NUM_CLUSTERS as SOFTHIER_DEFAULT_NUM_CLUSTERS
from testUtils.core import configure_cmake
from testUtils.codeGenerate import TilelangIOBuffer, generateTilelangSoftHierTestNetwork
from testUtils.pytestRunner import create_test_config


def _build_live_tilelang_gemm():
    """Return (body, M, K, N) for a live TileLang GEMM compiled to SoftHier C."""
    tilelang = pytest.importorskip("tilelang", reason = "tilelang package is required for TileLang tests")
    T = pytest.importorskip("tilelang.language", reason = "tilelang.language is required for TileLang tests")

    # Tilelang JIT function to be compiled
    @tilelang.jit
    def tl_gemm_serial(A, B, BLOCK_M: int, BLOCK_K: int, BLOCK_N: int):
        M, K, N = T.const("M, K, N")
        dtype = T.float16
        accum_dtype = T.float32
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)

        with T.Kernel(1, 1) as (bx, by):
        # with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N)) as (bx, by):
        # TODO: 这里对alloc_fragment的anno实际无效，AST中无法对这几条语句进行修饰，需要考虑其他表达方式
            with T.attr("anno", "cluster_id", T.int32(0)):
                A_local_0 = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
                B_local_0 = T.alloc_fragment((BLOCK_K, BLOCK_N), dtype)
                C_local_0 = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
            with T.attr("anno", "cluster_id", T.int32(1)):
                A_local_1 = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
                B_local_1 = T.alloc_fragment((BLOCK_K, BLOCK_N), dtype)
                C_local_1 = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)

            with T.attr("anno", "cluster_id", T.int32(0)):
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K)):
                    T.copy(A[bx*BLOCK_M:bx*BLOCK_M+BLOCK_M, k*BLOCK_K:k*BLOCK_K+BLOCK_K], A_local_0)
                    T.copy(B[k*BLOCK_K:k*BLOCK_K+BLOCK_K, by*BLOCK_N:by*BLOCK_N+BLOCK_N], B_local_0)
                    T.clear(C_local_0)
                    T.gemm(A_local_0, B_local_0, C_local_0)
                T.copy(C_local_0, C_local_1)
            with T.attr("anno", "cluster_id", T.int32(1)):
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K)):
                    T.copy(A[bx*BLOCK_M:bx*BLOCK_M+BLOCK_M, k*BLOCK_K:k*BLOCK_K+BLOCK_K], A_local_1)
                    T.copy(B[k*BLOCK_K:k*BLOCK_K+BLOCK_K, by*BLOCK_N:by*BLOCK_N+BLOCK_N], B_local_1)
                    T.gemm(A_local_1, B_local_1, C_local_1)
                T.copy(C_local_1, C[bx*BLOCK_M:bx*BLOCK_M+BLOCK_M, by*BLOCK_N:by*BLOCK_N+BLOCK_N])
        return C
    
    # Tilelang JIT function to be compiled
    @tilelang.jit
    def tl_gemm_parallel(A, B, BLOCK_M: int, BLOCK_K: int, BLOCK_N: int):
        M, K, N = T.const("M, K, N")
        dtype = T.float16
        accum_dtype = T.float32
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)

        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N)) as (bx, by):
            with T.attr("anno", "cluster_id", T.int32(0)):
                A_local_0 = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
                B_local_0 = T.alloc_fragment((BLOCK_K, BLOCK_N), dtype)
                C_local_0 = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
                C_local_0_temp = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
                
            with T.attr("anno", "cluster_id", T.int32(1)):
                A_local_1 = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
                B_local_1 = T.alloc_fragment((BLOCK_K, BLOCK_N), dtype)
                C_local_1 = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)

            with T.attr("anno", "cluster_id", T.int32(0)):
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K)):
                    T.copy(A[bx*BLOCK_M:bx*BLOCK_M+BLOCK_M, k*BLOCK_K:k*BLOCK_K+BLOCK_K], A_local_0)
                    T.copy(B[k*BLOCK_K:k*BLOCK_K+BLOCK_K, by*BLOCK_N:by*BLOCK_N+BLOCK_N], B_local_0)
                    T.clear(C_local_0)
                    T.gemm(A_local_0, B_local_0, C_local_0)
            with T.attr("anno", "cluster_id", T.int32(1)):
                for k in T.Pipelined(T.ceildiv(K, BLOCK_K)):
                    T.copy(A[bx*BLOCK_M:bx*BLOCK_M+BLOCK_M, k*BLOCK_K:k*BLOCK_K+BLOCK_K], A_local_1)
                    T.copy(B[k*BLOCK_K:k*BLOCK_K+BLOCK_K, by*BLOCK_N:by*BLOCK_N+BLOCK_N], B_local_1)
                    T.clear(C_local_1)
                    T.gemm(A_local_1, B_local_1, C_local_1)
                T.copy(C_local_1, C_local_0_temp)
            with T.attr("anno", "cluster_id", T.int32(0)):
                T.reduce(C_local_0_temp, C_local_0, "add")
                T.copy(C_local_0, C[bx*BLOCK_M:bx*BLOCK_M+BLOCK_M, by*BLOCK_N:by*BLOCK_N+BLOCK_N])
        return C

    # inputs to the JIT functions
    M = 1024
    N = 1024
    K = 128
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    A = T.empty((M, K), T.float16)
    B = T.empty((K, N), T.float16)

    body = compile_tilelang_to_softhier(tl_gemm_serial, A, B, BLOCK_M = BLOCK_M, BLOCK_N = BLOCK_N, BLOCK_K = BLOCK_K, cluster_id = 0)
    return body, M, K, N


@pytest.mark.softhier
@pytest.mark.tilelang
def test_tilelang_softhier_network_c_compiles_in_deeploy_flow(
    deeploy_test_dir: Path,
    toolchain: str,
    toolchain_dir: str,
    cmake_args: list[str],
) -> None:
    # TODO: now only using a fixed GEMM example. Consider importing from external source
    body, M, K, N = _build_live_tilelang_gemm()

    # Generate numpy reference data
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((M, K)).astype(np.float16)
    B_np = rng.standard_normal((K, N)).astype(np.float16)
    C_ref = (A_np @ B_np).astype(np.float16)

    elem_bytes = 2  # sizeof(fp16)
    # The visitor mangles PrimFunc param names with "DeeployNetwork_" prefix.
    input_bufs = [
        TilelangIOBuffer(name = "DeeployNetwork_A", c_dtype = "fp16", nbytes = M * K * elem_bytes, is_input = True),
        TilelangIOBuffer(name = "DeeployNetwork_B", c_dtype = "fp16", nbytes = K * N * elem_bytes, is_input = True),
    ]
    output_bufs = [
        TilelangIOBuffer(name = "DeeployNetwork_C", c_dtype = "fp16", nbytes = M * N * elem_bytes, is_input = False),
    ]

    softhier_cmake_args = list(cmake_args) + [f"num_clusters={SOFTHIER_DEFAULT_NUM_CLUSTERS}"]
    config = create_test_config(
        test_name = "Tilelang/gemm",
        platform = "SoftHier",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = softhier_cmake_args,
        tiling = False,
    )

    generated_source = Path(config.gen_dir)
    generated_source.mkdir(parents = True, exist_ok = True)
    generateTilelangSoftHierTestNetwork(
        tilelangBody = body,
        dumpdir = str(generated_source),
        input_bufs = input_bufs,
        output_bufs = output_bufs,
        test_inputs = [A_np, B_np],
        test_outputs = [C_ref],
    )

    configure_cmake(config)

    cmake_cmd = os.environ.get("CMAKE", "cmake")
    build_cmd = [cmake_cmd, "--build", config.build_dir, "--target", "network"]
    result = subprocess.run(build_cmd, check = False, capture_output = True, text = True)
    if result.returncode != 0:
        short_error = "\n".join(result.stderr.splitlines()[-20:])
        pytest.fail("TileLang-generated Network.c failed to compile.\n"
                    f"Last compiler lines:\n{short_error}")
