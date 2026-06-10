# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Deeploy runner for TileLang → SoftHier compilation.

Two usage modes
---------------
1. **Live PrimFunc** (recommended): provide a TileLang ``@tilelang.jit``
   function together with its parameter values; the PrimFunc is obtained via
   ``fn.get_tir(...)`` and compiled directly.

2. **Text-file AST** (legacy / debugging): pass ``--textfile`` and a path to
   one of the text dumps in ``Deeploy/TileLang/`` (e.g. ``test_gemv_ast.txt``).
   The file is printed for inspection only — actual code generation still
   requires a live PrimFunc; this mode emits no Network.c.

Usage
-----
::

    # Python API:
    from deeployRunner_tilelang_softhier import compile_tilelang_to_softhier
    code = compile_tilelang_to_softhier(my_gemv, A, B, BLOCK_M=128, BLOCK_K=64,
                                        cluster_id=0)

    # CLI (built-in GEMV demo):
    python deeployRunner_tilelang_softhier.py

    # CLI (text-file inspection):
    python deeployRunner_tilelang_softhier.py --textfile Deeploy/TileLang/test_gemv_ast.txt
"""

import os
import sys

# Append the directory one level up to sys.path so we can import Deeploy if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from typing import List, Optional

from Deeploy.DeeployTypes import NetworkContext, ExecutionBlock
from Deeploy.TileIR.Frontend.TilelangVisitor import TilelangVisitor
from Deeploy.TileIR.IR.CollectivePrimitives import ClusterGroupRegistry
from Deeploy.TileIR.IR.HardwareBinding import CollectiveBackend, HardwareBinding, SoftHierCollectiveBackend
from Deeploy.TileIR.IR.ParallelPasses import CollectiveLoweringPass, GroupAwareBarrierPass
from Deeploy.TileIR.Midend.TileBindings import DedupSyncPass, HoistAllocFreePass, SoftwarePipelinePass
from Deeploy.Targets.SoftHier.Platform import SoftHierDynamicBuffer
from testUtils.codeGenerate import generateTilelangSoftHierTestNetwork

def _make_softhier_ctxt(name: str = "DeeployNetwork") -> NetworkContext:
    """Create a NetworkContext configured for SoftHier with DynamicBuffer."""
    return NetworkContext(
        variableBuffer  = SoftHierDynamicBuffer,
        constantBuffer  = SoftHierDynamicBuffer,
        structBuffer    = SoftHierDynamicBuffer,
        transientBuffer = SoftHierDynamicBuffer,
        name            = name,
    )


# TODO: consider making this as a method of a class that encapsulate the whole deployment flow
def compile_tilelang_to_softhier(jit_fn,
                                  *tir_args,
                                  cluster_id: Optional[int] = None,
                                  cluster_ids: Optional[List[int]] = None,
                                  network_name: str = "DeeployNetwork",
                                  **tir_kwargs) -> str:
    """Compile a TileLang jit function to a SoftHier Network.c body string.

    Parameters
    ----------
    jit_fn :
        A ``@tilelang.jit``-decorated function with a ``.get_tir()`` method.
    *tir_args :
        Positional arguments forwarded to ``jit_fn.get_tir()``.
    cluster_id : Optional[int]
        When set, every operation is wrapped in::

            uint32_t CID = flex_get_cluster_id();
            if (CID == <cluster_id>) { ... }

    network_name : str
        Used for C symbol mangling (default ``'DeeployNetwork'``).
    **tir_kwargs :
        Keyword arguments forwarded to ``jit_fn.get_tir()``.

    Returns
    -------
    str
        Kernel body string (insert inside a ``tilelang_main(...)`` function).
    """
    # parse TileLang function to get the PrimFunc
    primfunc = jit_fn.get_tir(*tir_args, **tir_kwargs)
    # PrimFunc -> TileBinding 
    ctxt     = _make_softhier_ctxt(network_name)
    visitor  = TilelangVisitor(cluster_id=cluster_id, cluster_ids=cluster_ids)
    tilebinding = visitor.visit_bindings(primfunc, ctxt)
    # TileBinding -> raw ExecutionBlock
    raw_eb: ExecutionBlock = tilebinding.bind()
    # Per-op code transformation in each TileBinding
    ctxt, eb = tilebinding.codeTransform(ctxt, raw_eb)
    # code generation: NetworkContext + ExecutionBlock -> C code string
    return eb.generate(ctxt)

def compile_tilelang_to_softhier_parallel(
    jit_fn,
    *tir_args,
    group_registry: ClusterGroupRegistry,
    hw_binding: HardwareBinding,
    backend: Optional[CollectiveBackend] = None,
    cluster_policy: str = "hybrid",
    num_clusters: Optional[int] = None,
    cluster_ids: Optional[List[int]] = None,
    network_name: str = "DeeployNetwork",
    **tir_kwargs,
) -> str:
    """Compile a TileLang jit function with cluster-group collective support.

    Extends ``compile_tilelang_to_softhier`` with group-aware barrier insertion
    and collective lowering.

    Parameters
    ----------
    jit_fn :
        A ``@tilelang.jit``-decorated function with a ``.get_tir()`` method.
    *tir_args :
        Positional arguments forwarded to ``jit_fn.get_tir()``.
    group_registry : ClusterGroupRegistry
        Declared cluster groups.
    hw_binding : HardwareBinding
        Physical cluster-ID mapping for the groups.
    backend : Optional[CollectiveBackend]
        Hardware collective backend.  Defaults to ``SoftHierCollectiveBackend``.
    cluster_policy : str
        Cluster assignment policy for ``TilelangVisitor``.  One of
        ``"explicit_attr"``, ``"block_idx"``, or ``"hybrid"`` (default).
    num_clusters : Optional[int]
        Number of physical clusters for modulo mapping from block id to
        cluster id.  Used when ``cluster_policy`` includes block-index
        inference.
    network_name : str
        Used for C symbol mangling (default ``'DeeployNetwork'``).
    **tir_kwargs :
        Keyword arguments forwarded to ``jit_fn.get_tir()``.

    Returns
    -------
    str
        Kernel body string with group-init, group barriers, and lowered
        collectives.
    """
    if backend is None:
        backend = SoftHierCollectiveBackend()

    if cluster_ids is None and hw_binding is not None:
        all_ids = set()
        for ids in hw_binding.cluster_ids.values():
            all_ids.update(ids)
        cluster_ids = sorted(all_ids)

    primfunc = jit_fn.get_tir(**tir_kwargs)
    ctxt = _make_softhier_ctxt(network_name)
    visitor = TilelangVisitor(
        cluster_policy=cluster_policy,
        num_clusters=num_clusters,
        cluster_ids=cluster_ids,
        group_registry=group_registry,
        hw_binding=hw_binding,
    )
    tilebinding = visitor.visit_bindings(primfunc, ctxt)

    # After visiting, use the registry that the visitor populated from the kernel
    # spec (via T.cluster_group annotations).  When the caller passed group_registry=None,
    # the visitor auto-creates a ClusterGroupRegistry; we pick it up here so that
    # the passes below see the full group geometry without requiring a manual registry
    # in driver code.
    effective_registry = visitor.group_registry if group_registry is None else group_registry

    # Replace default GlobalClusterBarrierPass with group-aware pass;
    # keep SoftwarePipelinePass first to handle T.Pipelined(num_stages=N) loops.
    # Pass group_registry so SoftwarePipelinePass can emit strided K-split loops
    # for multi-cluster TP groups instead of running all K-blocks on every cluster.
    from Deeploy.TileIR.Passes.SpatzVectorization import SpatzVectorizationPass
    tilebinding.binding_passes = [SoftwarePipelinePass(group_registry=effective_registry), HoistAllocFreePass(), SpatzVectorizationPass(), GroupAwareBarrierPass()]
    # Add collective lowering pass
    tilebinding.add_binding_pass(
        CollectiveLoweringPass(
            registry=effective_registry,
            hw_binding=hw_binding,
            backend=backend,
        ))
    # Final cleanup: collapse runs of consecutive intra-cluster syncs.
    tilebinding.add_binding_pass(DedupSyncPass())

    ctxt, eb = tilebinding.codeTransform(ctxt)
    return eb.generate(ctxt)


def write_tilelang_softhier_test_network(
    body: str,
    dumpdir: str,
    func_sig: str = "void tilelang_main(fp16* A, fp16* B, fp16* C)",
    buffer_initialization_code: str = "",
    global_definition_code: str = "",
) -> None:
    """Generate TileLang SoftHier test artifacts (Network.c/.h, inputs/outputs headers)."""
    generateTilelangSoftHierTestNetwork(
        tilelangBody = body,
        dumpdir = dumpdir,
        functionSignature = func_sig,
        bufferInitializationCode = buffer_initialization_code,
        globalDefinitionCode = global_definition_code,
    )
    print(f"[deeployRunner] Wrote test artifacts under {dumpdir}")


def _demo_live() -> None:
    """Demo: compile the built-in GEMV example via the live TVM API."""
    try:
        import tilelang
        import tilelang.language as T
    except ImportError:
        print("[deeployRunner] tilelang not installed — skipping live demo.")
        return

    @tilelang.jit
    def tl_gemv(A, B, BLOCK_M: int, BLOCK_K: int):
        M, K        = T.const("M, K")
        dtype       = T.float16
        accum_dtype = T.float32
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K,),   dtype)
        C = T.empty((M,), dtype)

        with T.Kernel(T.ceildiv(M, BLOCK_M), threads=128) as pid_m:
            A_local = T.alloc_fragment((BLOCK_M, BLOCK_K), dtype)
            B_local = T.alloc_fragment((BLOCK_K,),         dtype)
            C_local = T.alloc_fragment((BLOCK_M,),         accum_dtype)
            AB_temp = T.alloc_fragment((BLOCK_M, BLOCK_K), accum_dtype)

            T.clear(C_local)
            for k in T.Serial(K // BLOCK_K):
                T.copy(A[pid_m * BLOCK_M, k * BLOCK_K], A_local)
                T.copy(B[k * BLOCK_K,],                 B_local)
                for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                    AB_temp[i, j] = (A_local[i, j].astype(accum_dtype) *
                                     B_local[j].astype(accum_dtype))
                T.reduce_sum(AB_temp, C_local, dim=1, clear=False)
            T.copy(C_local, C[pid_m * BLOCK_M,])
        return C

    A = T.empty((1024, 512), T.float16)
    B = T.empty((512,),      T.float16)
    print("[deeployRunner] Compiling GEMV via live TVM API …")
    body = compile_tilelang_to_softhier(
        tl_gemv, A, B, BLOCK_M=128, BLOCK_K=64, cluster_id=0)
    write_tilelang_softhier_test_network(
        body,
        dumpdir = "TEST_SOFTHIER/Tests/Tilelang/live_demo",
        func_sig = "void tilelang_main(fp16* A, fp16* B, fp16* C)",
    )
    print("[deeployRunner] Done. Ready to compile with GCC for SoftHier.")

# TODO: not used for now
def main():
    import argparse
    parser = argparse.ArgumentParser(description="TileLang → SoftHier compiler")
    parser.add_argument("--textfile", metavar="AST_TXT",
                        help="Print a text-format TVM AST dump (inspection only).")
    args = parser.parse_args()

    if args.textfile:
        print(f"[deeployRunner] --- Text-file AST: {args.textfile} ---")
        with open(args.textfile) as f:
            print(f.read())
        print("[deeployRunner] Inspection mode only: use the Python API or --live for code gen.")
        return

    _demo_live()


if __name__ == "__main__":
    main()
