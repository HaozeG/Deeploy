# TileIR — TileLang → SoftHier compiler extension

## What lives where

| Component | File |
|-----------|------|
| DSL ops (`D.*`) | `Deeploy/TileIR/Frontend/tl_deeploy.py` |
| `T.cluster_group` | **external repo** `tilelang/tilelang/language/cluster_group.py` |
| Frontend visitor | `Deeploy/TileIR/Frontend/TilelangVisitor.py` |
| IR primitives | `Deeploy/TileIR/IR/CollectivePrimitives.py`, `HardwareBinding.py` |
| Passes | `Deeploy/TileIR/Passes/` |
| C templates | `Deeploy/TileIR/Backend/Templates/` |

## Quickstart: compiling a kernel

```python
from DeeployTest.deeployRunner_tilelang_softhier import compile_tilelang_to_softhier_parallel
from Deeploy.TileIR.IR import ClusterGroup, ClusterGroupRegistry, HardwareBinding

# Declare logical cluster groups
registry = ClusterGroupRegistry([
    ClusterGroup("summa", group_x=GX, group_y=GY)
])
# Map group name → physical cluster IDs
hw = HardwareBinding({"summa": list(range(GX * GY))})

c_code = compile_tilelang_to_softhier_parallel(
    jit_fn,
    group_registry=registry,
    hw_binding=hw,
    num_clusters=GX * GY,
)
```

## DSL ops

Import with:
```python
from Deeploy.TileIR.Frontend import tl_deeploy as D
```

### `D.broadcast(buf, level, axis, group, root)`

Broadcast `buf` from root rank to all other ranks in the group.

- `level`: `"intra_group"` (within one group instance) or `"inter_group"` (across instances)
- `axis`: axis name to broadcast along (required for 2-D groups; `""` for 1-D)
- `group`: `ClusterGroup` name (string)
- `root`: C expression for source rank (e.g. `"local_y"`, `"0"`)

### `D.reduce(buf, op, level, axis, group, root)`

Reduce `buf` in-place across all ranks.

- `op`: `"sum"` | `"max"` | `"min"` | `"prod"`
- `level`, `axis`, `group`, `root`: same semantics as `D.broadcast`; `root=""` → allreduce

### Example (SUMMA GEMM pattern)

```python
import tilelang
import tilelang.language as T
from Deeploy.TileIR.Frontend import tl_deeploy as D

@tilelang.jit
def summa_gemm(A, B, C, BM: int, BN: int, BK: int, GX_: int, GY_: int):
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

## Required parameters for `compile_tilelang_to_softhier_parallel`

| Parameter | Type | Description |
|-----------|------|-------------|
| `group_registry` | `ClusterGroupRegistry` | Declares cluster groups (shape, axes) |
| `hw_binding` | `HardwareBinding` | Maps group name → physical cluster IDs |
| `num_clusters` | `int` | Total physical clusters on the SoC |

Optional:
- `use_block_idx: bool` — infer cluster from block indices when no explicit `T.attr(..., "cluster_id", ...)` is present (default `False`)
- `backend` — collective backend, defaults to `SoftHierCollectiveBackend`

## Compilation pipeline

Passes run in this order after the visitor:

```
SoftwarePipelinePass → HoistAllocFreePass → SpatzVectorizationPass
→ GroupAwareBarrierPass → CollectiveLoweringPass → DedupSyncPass
```

The visitor (`TilelangVisitor`) emits `TileBinding` objects tagged with `shard_group_id` (a plain string naming the enclosing cluster group). The passes use this tag to insert barriers and lower collectives correctly.

## How to add a new op

1. **Register in TVM op registry** (`tl_deeploy.py`):
   ```python
   tvm.ir.register_op_attr(_op_name, "TCallEffectKind", tir.CallEffectKind.Opaque)
   ```
2. **Add validation helper** (optional): `_check_<op>_args(...)`.
3. **Expose as `D.<op>()`**: validate inputs, emit `tir.call_intrin("handle", ...)`.
4. **Add visitor handler** in `TilelangVisitor`:
   - Register in `self._supported_ops` dict (key = last `.`-segment of intrinsic name).
   - Implement `_handle_<op>(self, args, cluster_id)`.
   - Parse buffers via `self._region_buf_name(args[i])` or `str(args[i]).strip('"')`.
   - Emit via `self._emit_binding(op_kind, Template, rep)` or `_emit_collective_binding(...)`.

For collective ops, also add a strategy in `Backend/CollectiveStrategies.py` implementing `matches()` and `emit()`.
