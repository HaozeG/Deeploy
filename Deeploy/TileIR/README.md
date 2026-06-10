# Tilelang for SoftHier based on Deeploy

## DSL Extension

> `TileIR/Frontend/tl_deeploy.py`
> `tilelang/tilelang/language/cluster_group.py`

### Usage example:
```python
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

    with T.Kernel(T.ceildiv(M, GY_ * BM), T.ceildiv(N, GX_ * BN)) as (by, bx):
        with T.cluster_group("summa", x=GX_, y=GY_, num_groups=1,
                                axes=("x", "y")) as (inst_id, local_x, local_y):
            A_local = T.alloc_fragment((BM, BK), dtype)
            B_local = T.alloc_fragment((BK, BN), dtype)
            C_local = T.alloc_fragment((BM, BN), dtype)
            T.clear(C_local)

            for bk in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                if local_x == local_y:
                    T.copy(A[(by * GY_ + local_y) * BM, bk * BK], A_local)
                    T.copy(B[bk * BK, (bx * GX_ + local_x) * BN], B_local)
                D.broadcast(A_local, level="intra_group", axis="x",
                            group="summa", root=local_y)
                D.broadcast(B_local, level="intra_group", axis="y",
                            group="summa", root=local_x)
                T.gemm(A_local, B_local, C_local, clear_accum=False)

            T.copy(C_local, C[(by * GY_ + local_y) * BM, (bx * GX_ + local_x) * BN])


```
`from Deeploy.TileIR.Frontend import tl_deeploy as D`. Use `D.` to differentiate from Tilelang's native primitives.

### Workflow for adding a new op:

1. Register in TVM op registry: `tvm.ir.register_op_attr(_op_name, "TCallEffectKind", tir.CallEffectKind.Opaque)`
2. Frontend validation helpers: `_check_reduce_args`, `_check_broadcast_args`
3. API for programmer to write as `D.func()`: define function, validate inputs, emit TVM op `tir.call_intrin("handle", ...)`
4. Use `jit_func.get_tir()` to get Tilelang AST: see function `compile_tilelang_to_softhier_parallel`

### Ops
Tested ops:

```python
D.reduce(buffer, level, axis, group, root)
D.broadcast()

```

Untested proposed ops:
```python
D.sync_grid() # sync with global barrier
# could use for debugging
D.device_assert()
```

## Compiler

> The definition of IR here is negotiable. Current design might not be clean and extensible

Extend with Deeploy's original workflow. Compilation flow as below:

- Function decorated with `@tilelang.jit`
- Get Tilelang AST with `.get_tir()`
- Create frontend visitor `TilelangVisitor` with hardware infos (lastest version remove this requirement, but rely on `cluster_group` info from Tilelang AST. I think for a complete mapping from logical cluster to physical cluster, here we need to register physical cluster configs with visitor emitted IRs)
- *Frontend*: Visit Tilelang AST with `TilelangVisitor` and emit `TileBindingPipeline` as a list of `TileBinding` (extended from Deeploy's Bindings) and context (extended from Deeploy's `NetworkContext`)
- *Midend*: Apply optimization passes with `tilebinding.codeTransform(ctxt)` to get ordered list of `ExecutionBlock`, update information in `NetworkContext` ctxt
- *Backend*: Generate code from `NetworkContext` and `ExecutionBlock`

Tilelang AST -> TileBinding -> ExecutionBlock + NetworkContext -> code

Core IR is `TileBinding`. Recorded information as below:
- Category of IR: MEMORY/COMPUTE/CONTROL_FLOW/SYNC/COLLECTIVE/MISC. 
    - used for software pipeline pass to recognize COMPUTE/COMMUNICATION.
- OpKind: semantic operation kind
    - for example, alloc, free, load, gemm, for_open, for_close
- Code template `NodeTemplate`
    - reused from Deeploy. Mako templates
- code_transformer `CodeTransformation`
    - reused from Deeploy. as a pass applied to a single `ExecutionBlock`
    - Here only has per-op transformations, inter-op transformations recorded in `TileBindingPipeline`

Special IR for collectives `CollectiveBinding`:
- Consider it heavily relies on `cluster_group` informations, here deals with it separately
    - For other IRs, information like physical cluster assignments can just be a list of cluster ids

> But I think this can be simplified or unified, if not reusing Deeploy's design. Now, only the overall IR design (what is stored in IR) is borrowed from Deeploy, but Frontend, Midend, Backend all did not follow Deeploy's design.

Detailed example as below:
```python
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
```

### Frontend

> `TileIR/Frontend/TilelangVisitor.py`

Visitor: Key entry function is `_visit_stmt`. Based on the type of TVM op, call corresponding `_visit_` function. Such function checks on arguments of TVM op, do initial checks, convert to annotations and call `_emit_binding` to emit one `TileBinding` with information collected.

When adding a new op, here might need a new visitor function.

> `ShardMetadata` now not used. Previously designed to record predefined parallelization strategies information

### Midend

> `TileIR/Passes`

General philosophy:
- Recognize pattern in current ordered list of `TileBinding` 
- Replace it with desired new `TileBinding`s

### Backend

> `TileIR/Backend/Templates`

Basically Mako templates as in Deeploy's design.

