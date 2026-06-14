# TileIR — TileLang → SoftHier compiler extension

## Environment setup

```bash
# 1. Source SoftHier toolchain paths (adjust to softhier_ if using that clone)
source /home/haoze/softhier/sourceme.sh

# 2. Activate Deeploy conda environment (sets Python paths)
conda activate Deeploy

# 3. Install Deeploy (if not already installed)
pip install -e /home/haoze/Deeploy

# 4. Install TileLang external repo (provides T.cluster_group, T.Pipelined, D.* base)
cd /home/haoze/codebase/tilelang/tilelang && pip install -e .
```

Running e2e tests:
```bash
cd /home/haoze/Deeploy/DeeployTest
pytest test_tilelang_gemm.py -v -s -m "tilelang and softhier" \
  --toolchain=GCC --toolchain-install-dir=$SOFTHIER_INSTALL_DIR/third_party/toolchain/install
pytest test_tilelang_attention.py -v -s -m "tilelang and softhier" \
  --toolchain=GCC --toolchain-install-dir=$SOFTHIER_INSTALL_DIR/third_party/toolchain/install
```

Key external source files (in `/home/haoze/codebase/tilelang/tilelang/tilelang/language/`):
- `cluster_group.py` — `T.cluster_group` context manager
- `collective_op.py` — `T.allreduce`, `T.broadcast` native TileLang primitives

## What lives where

| Component | File |
|-----------|------|
| DSL ops (`D.*`) | `Deeploy/TileIR/Frontend/tl_deeploy.py` |
| `T.cluster_group` | **external repo** `tilelang/tilelang/language/cluster_group.py` |
| Frontend visitor | `Deeploy/TileIR/Frontend/TilelangVisitor.py` |
| IR primitives | `Deeploy/TileIR/IR/CollectivePrimitives.py` — ClusterGroup, CollectiveOpSpec |
| Hardware binding | `Deeploy/TileIR/IR/HardwareBinding.py` — HardwareBinding, SoftHierCollectiveBackend |
| Core IR | `Deeploy/TileIR/IR/TileBinding.py` — TileBinding, TileOpKind, TileOpCategory |
| Collective IR | `Deeploy/TileIR/IR/CollectiveBinding.py` — CollectiveBinding |
| Passes | `Deeploy/TileIR/Passes/` — SoftwarePipeline, Sync, CollectiveLowering, etc. |
| C templates | `Deeploy/TileIR/Backend/Templates/SoftHierTileTemplates.py` |
| Collective templates | `Deeploy/TileIR/Backend/Templates/SoftHierCollectiveTemplates.py` |
| Collective strategies | `Deeploy/TileIR/Backend/CollectiveStrategies.py` |

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

### T.cluster_group (external TileLang, `tilelang/language/cluster_group.py`)

Context manager that declares a logical cluster group for collective ops.

```python
with T.cluster_group("summa", x=GX, y=GY, num_groups=1, axes=("x", "y"),
                     split_axes=(), split_shape=()) as (gid, gid_x, gid_y):
    ...
```

- `x`, `y`: group dimensions (number of clusters per axis)
- `num_groups`: number of group instances (usually 1)
- `axes`: logical axis names, matched by `D.broadcast(axis=...)` / `D.reduce(axis=...)`
- `split_axes`, `split_shape`: K-split partitioning for TP groups (triggers strided K loops)
- Yields `(gid, gid_x, gid_y, *split_ids)` as TIR Vars usable in the kernel body

### Tested ops (e2e verified)

The following are the only fully e2e-verified ops, used by the four reference tests:

- `D.broadcast(buf, level, axis, group, root)` — broadcast from root rank to all others
- `D.reduce(buf, op, level, axis, group, root)` — in-place reduction across all ranks

**`D.broadcast` arguments:**
- `level`: `"intra_group"` (within one group instance) or `"inter_group"` (across instances)
- `axis`: axis name to broadcast along (required for 2-D groups; `""` for 1-D)
- `group`: `ClusterGroup` name (string)
- `root`: C expression for source rank (e.g. `"local_y"`, `"0"`)

**`D.reduce` arguments:**
- `op`: `"sum"` | `"max"` | `"min"` | `"prod"`
- `level`, `axis`, `group`, `root`: same semantics; `root=""` → allreduce

**Reference e2e tests:**
- `DeeployTest/test_tilelang_gemm.py`: `test_summa_gemm_dp_2d_e2e` (uses `D.broadcast`),
  `test_summa_gemm_dp_split_k_e2e`, `test_summa_gemm_dp_split_k2d_e2e` (use `D.reduce`)
- `DeeployTest/test_tilelang_attention.py`: `test_masked_flash_attn_dp_e2e` (uses `D.reduce`)

### Untested / incomplete ops

The following exist in `tl_deeploy.py` with visitor handlers but are **not e2e tested**:
- `D.sync_grid()` — emits global barrier (functional but untested in context)
- `D.device_assert()`, `D.assume()` — compile-only tested
- `T.allreduce`, `T.broadcast`, `T.group_shift`, `T.group_bcast_axis` — native TileLang
  collective ops routed through untested CollectiveStrategies (see Known limitations)

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

### Overview

```
jit_fn.get_tir()              → TVM PrimFunc
→ TilelangVisitor              → TileBindingPipeline (ordered list of TileBindings)
→ TileBindingPipeline.codeTransform(ctxt)   → applies passes in order
→ ExecutionBlock.generate(ctxt)             → C code string
```

The compiler extends Deeploy's original workflow. Only the overall IR structure
(what is stored in each binding) is borrowed from Deeploy — the Frontend, Midend
and Backend are all new.

### Passes (in order)

```
SoftwarePipelinePass → HoistAllocFreePass → SpatzVectorizationPass
→ GroupAwareBarrierPass → CollectiveLoweringPass → DedupSyncPass
```

The visitor (`TilelangVisitor`) emits `TileBinding` objects tagged with `shard_group_id`
(a plain string naming the enclosing cluster group). The passes use this tag to insert
barriers and lower collectives correctly.

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
   - Emit via `self._emit_binding(op_kind, Template, rep)`.

For collective ops, also add a strategy in `Backend/CollectiveStrategies.py`
implementing `matches(spec, group, topology)` and `emit(spec, binding, registry)`.

## Internal IR design

### TileBinding (`IR/TileBinding.py`)

Core IR unit — emitted by the frontend, consumed by passes, rendered to C by templates.

Fields:
- `op_kind`: str — one of `TileOpKind` values, grouped by `TileOpCategory`:
  - MEMORY: `alloc`, `free`, `load`, `store`, `copy`, `fill`, `alloc_reducer`
  - COMPUTE: `reduce`, `gemm`, `eltwise`, `math_preamble`
  - CONTROL_FLOW: `for_open`, `for_close`, `if_open`, `if_close`, `else_open`, `else_close`
  - SYNC: `sync`, `global_barrier`, `group_barrier`
  - COLLECTIVE: `group_collective`
  - MISC: `comment`, `block_preamble`, `runtime_assert`, `runtime_assume`
- `template`: `NodeTemplate` — Mako string; `${key}` substituted from `operator_representation`
- `operator_representation`: `Dict[str, Any]` — template variables; always includes:
  - `"cluster_id"`: `Optional[int]` — `None` = all clusters execute; `int` = cluster-guarded
  - `"shard_group_id"`: `Optional[str]` — enclosing cluster group name (see below)
- `op_name`: str — debug label
- `code_transformer`: optional per-binding `CodeTransformation` (reused from Deeploy)

`TileOpCategory` is used by `SoftwarePipelinePass` to separate COMPUTE from MEMORY/SYNC.

### CollectiveBinding (`IR/CollectiveBinding.py`)

Subclass of `TileBinding` for inter-cluster collectives. `op_kind` is always `"group_collective"`.
Adds `spec: CollectiveOpSpec` — the math-level description of the collective.

Emitted by: `_handle_dp_reduce`, `_handle_dp_broadcast`, `_handle_allreduce`, `_handle_collective`.
Lowered to hardware TileBindings by `CollectiveLoweringPass` via
`SoftHierCollectiveBackend.lower(spec, hw_binding, registry)`.

### shard_group_id convention

`operator_representation["shard_group_id"]` is a plain string naming the enclosing
`T.cluster_group` scope, or `None` for top-level ops. Set automatically by `_emit_binding()`
from `self._current_group_id`. Read by:
- `GroupAwareBarrierPass` — to identify group membership during barrier insertion
- `CollectiveLoweringPass._group_id_of()` — to tag alloc bindings for cluster guards
- `SoftwarePipelinePass._detect_tp_k_split()` — to find TP groups needing K-split loops

### Data model

**ClusterGroup** (`IR/CollectivePrimitives.py`): logical group of clusters.
Fields: `group_id`, `group_x`, `group_y`, `num_groups`, `axis_names`, `root_coord`,
`physical_cluster_ids` (populated by `HardwareBinding.populate_registry`),
`split_axes`, `split_shape`.

**ClusterGroupRegistry**: named registry of `ClusterGroup` objects.
Methods: `get(gid)`, `contains(gid)`, `root_cluster_for(gid)`, `clusters_for(gid)`.

**HardwareBinding** (`IR/HardwareBinding.py`): maps group names to physical cluster IDs.
`cluster_ids: Dict[str, List[int]]`
Key methods: `populate_registry(registry)`, `active_instances(gid, registry)`,
`compute_hw_bitmask(gid)`, `clusters_for(gid)`.

**CollectiveOpSpec** (`IR/CollectivePrimitives.py`): math-level spec for a collective.
Core fields: `op`, `group_id`, `src_buffer`, `dst_buffer`, `reduce_op`, `reduce_axis`.
D.reduce/D.broadcast fields: `level`, `axis`, `root_expr`.
Directional fields: `shift_by`, `from_coord`, `from_coord_expr`.

## Frontend internals

> `TileIR/Frontend/TilelangVisitor.py`

Entry point: `visit_bindings(primfunc, ctxt)` → `TileBindingPipeline`.

`_visit_stmt` is the top-level dispatcher on TVM IR node type:
- `_visit_Block` — handles `alloc_buffers`, emits `alloc`/`free` around the body
- `_visit_AttrStmt` — handles `cluster_id` attr and `cluster_group` attr
  (sets `self._current_group_id` for the scope when entering `T.cluster_group`)
- `_visit_For` — handles `T.Pipelined` and `T.Parallel` loops
- `_visit_Evaluate` → dispatches `tir.call_intrin` nodes by name suffix to `_handle_*`

`_emit_binding(op_kind, template, rep)` appends one `TileBinding`; automatically injects
`"shard_group_id": self._current_group_id` if not already in `rep`.

`_resolve_cluster(explicit_cluster_id)`:
- Explicit `T.attr(..., "cluster_id", N)` always wins.
- If `use_block_idx=True`: falls back to `_infer_cluster_from_block_idx()` which
  linearises `blockIdx` from T.Kernel loop vars via the `cluster_ids` list.
- Otherwise: uses fallback `cluster_id` or `None` (all clusters execute).

## Midend: passes

> `TileIR/Passes/`

Each pass is a `@dataclass` subclassing `TileBindingPass` (Base.py).
Interface: `apply(bindings: List[TileBinding]) -> List[TileBinding]`.

Philosophy: pattern-match the ordered `TileBinding` list and replace matched spans
with new `TileBinding`s.

What each pass does:
1. **SoftwarePipelinePass** — detects `T.Pipelined` loops, emits prefetch + pipelined body
   with double-buffering. For TP K-split groups (`group_x > 1`): rewrites loop to strided
   per-cluster K iteration (detected via `shard_group_id` + `ClusterGroupRegistry`).
2. **HoistAllocFreePass** — moves `alloc`/`free` bindings outside pipeline loop bodies.
3. **SpatzVectorizationPass** — vectorises eligible loops for Spatz SIMD units.
4. **GroupAwareBarrierPass** — inserts `flex_global_barrier_xy()` on `cluster_id` transitions.
5. **CollectiveLoweringPass**:
   - Calls `hw_binding.populate_registry(registry)`.
   - Prepends `global_barrier + group_init + global_barrier + group_context` per group
     (using `TileGroupInitTemplate` / `TileGroupContextTemplate`).
   - Patches `alloc_reducer` bindings with root cluster ID.
   - Lowers `CollectiveBinding` via `backend.lower(spec, hw_binding, registry)`, which
     selects a `CollectiveStrategy` and emits hardware TileBindings.
   - Tags group-member alloc bindings with `cluster_guard_type="group_membership"`.
6. **DedupSyncPass** — collapses consecutive intra-cluster `flex_intra_cluster_sync()` calls.

## Backend: templates

> `TileIR/Backend/Templates/`

Mako strings wrapped in `NodeTemplate(...)`. `${var}` is substituted from
`operator_representation[var]`.

- `SoftHierTileTemplates.py` — alloc/free/load/copy/gemm/reduce/sync/eltwise/barriers
- `SoftHierCollectiveTemplates.py` — group_init/context, collective reduce/broadcast

Collective strategy selection (`Backend/CollectiveStrategies.py`):
`SoftHierCollectiveBackend.lower()` iterates the strategy list and calls
`strategy.matches(spec, group, topology)`. First match wins, then `strategy.emit()`
returns the hardware TileBindings.

**E2e-verified strategies** (used by reference kernels):
- `DpReduceStrategy` — handles `D.reduce` (used by split-K GEMM and attention)
- `DpBroadcastStrategy` — handles `D.broadcast` (used by SUMMA GEMM 2D)

**Untested strategies** (handle `T.allreduce`, `T.broadcast`, directional ops):
`AxisReduceBroadcast`, `AxisBroadcast`, `FullGroupReduceBroadcast`,
`ScatterStrategy`, `GatherStrategy`, `GroupShift`, `GroupBcastAxis`.
These could be removed or isolated to reduce maintenance surface.

## Known limitations / future work

- Mapping from logical cluster group instances to physical cluster IDs is incomplete
  for non-contiguous or heterogeneous topologies (see `HardwareBinding.py`).
- `SoftwarePipelinePass` K-split detection (Strategy 1 via `shard_group_id`) only fires
  when the `cluster_group` AttrStmt wraps the loop body; falls back to registry scan
  when TileLang places the AttrStmt inside the `T.allreduce` call instead.
- Most `CollectiveStrategies` are untested (see above) — the strategy list could be
  pruned to only `DpReduceStrategy` + `DpBroadcastStrategy` once unused ops are dropped.
- IR design is negotiable — the `CollectiveBinding` / `TileBinding` split could be
  unified, and the overall IR simplified further.
