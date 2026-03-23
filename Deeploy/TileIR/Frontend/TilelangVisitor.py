# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileLang → Deeploy ExecutionBlock visitor.

Walks a TVM ``tir.PrimFunc`` produced by a TileLang ``@tilelang.jit``
function (via ``fn.get_tir(...)``) and emits a Deeploy ``ExecutionBlock``
composed of ``(NodeTemplate, OperatorRepresentation)`` pairs.

Usage
-----
**Preferred — full pipeline (allows pass injection):**
::

    import tilelang
    import tilelang.language as T
    from Deeploy.TileIR.Frontend.TilelangVisitor import TilelangVisitor
    from Deeploy.TileIR.Midend import TileBindingPipeline
    from Deeploy.Targets.SoftHier.Platform import SoftHierDynamicBuffer
    from Deeploy.DeeployTypes import NetworkContext

    @tilelang.jit
    def my_kernel(A, B, BLOCK_K: int):
        ...

    primfunc = my_kernel.get_tir(A, B, BLOCK_K=64)

    ctxt = NetworkContext(
        variableBuffer  = SoftHierDynamicBuffer,
        constantBuffer  = SoftHierDynamicBuffer,
        structBuffer    = SoftHierDynamicBuffer,
        transientBuffer = SoftHierDynamicBuffer,
    )

    visitor = TilelangVisitor()
    pipeline = visitor.visit_bindings(primfunc, ctxt)
    # optionally inject additional structural passes:
    # pipeline.add_binding_pass(MyCustomPass())
    ctxt, eb = pipeline.codeTransform(ctxt)
    code = eb.generate(ctxt)

**Shorthand — single-call convenience wrapper:**
::

    visitor = TilelangVisitor()
    eb = visitor.visit(primfunc, ctxt)   # visit_bindings → codeTransform internally
    code = eb.generate(ctxt)

Cluster-ID annotation
---------------------
The visitor propagates a ``cluster_id`` value to every emitted operation.
Priority order (highest to lowest):

1. **TIR annotation** — ``T.attr("anno", "cluster_id", T.int32(N))`` in the
   PrimFunc body overrides the cluster on a per-subtree basis.
2. **PrimFunc attribute** — ``primfunc.attrs["cluster_id"]`` sets the default
   for the whole function.
3. **Constructor default** — ``TilelangVisitor(cluster_id=N)`` provides the
   fallback when neither of the above is present.  ``None`` means no cluster
   guard (all clusters execute).

Each emitted C snippet is wrapped in::

    uint32_t CID = flex_get_cluster_id();
    if (CID == <cluster_id>) { ... }

**Cross-cluster barriers** — when the ``cluster_id`` changes between adjacent
operations, ``GlobalClusterBarrierPass`` (active by default in
``TileBindingPipeline``) inserts an unguarded::

    flex_global_barrier_xy();

to ensure the producer cluster has finished before the consumer cluster
starts.  Requires ``#include "flex_group_barrier_api.h"``.

Memory-space inference
----------------------
+------------------------+----------------+
| TVM buffer scope       | _memoryLevel   |
+========================+================+
| ``""``  (PrimFunc param) | ``'HBM'``     |
| ``"local.fragment"``   | ``'L1'``       |
| ``"shared"``           | ``'L1'``       |
+------------------------+----------------+
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

# TVM imports — kept conditional so the rest of Deeploy works without TVM
try:
    import tilelang
    import tvm
    import tvm.tir as tir
    from tvm.tir import PrimFunc
    _TVM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TVM_AVAILABLE = False
    PrimFunc = object  # type: ignore[assignment,misc]

from Deeploy.DeeployTypes import (
    CodeTransformation,
    ExecutionBlock,
    NetworkContext,
    NodeTemplate,
)
from Deeploy.TileIR.Midend import TileBinding, TileBindingPipeline
from Deeploy.TileIR.IR.CollectivePrimitives import (
    ClusterGroupRegistry,
    CollectiveOpSpec,
    ShardMetadata,
)
from Deeploy.TileIR.IR.ParallelPasses import CollectiveBinding
from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
    TileAllocReducerTemplate,
)
from Deeploy.TileIR.Backend.Templates.SoftHierTileTemplates import (
    ForLoopCloseTemplate,
    ForLoopOpenTemplate,
    TileAllocTemplate,
    TileCopyTemplate,
    TileEltwiseTemplate,
    TileFillTemplate,
    TileFreeTemplate,
    TileLoadTemplate,
    TileReduceTemplate,
    TileStoreTemplate,
    TileSyncTemplate,
    TileGemmTemplate,
)

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

#: TVM dtype string → C type name
_TVM_DTYPE_TO_C: Dict[str, str] = {
    "float16": "fp16",
    "float32": "float32_t",
    "int8":    "int8_t",
    "int16":   "int16_t",
    "uint8":   "uint8_t",
    "uint16":  "uint16_t",
}

#: TVM dtype string → element byte size
_TVM_DTYPE_BYTES: Dict[str, int] = {
    "float16": 2,
    "float32": 4,
    "int8":    1,
    "int16":   2,
    "uint8":   1,
    "uint16":  2,
}


def _c_dtype(tvm_dtype: str) -> str:
    return _TVM_DTYPE_TO_C.get(tvm_dtype, tvm_dtype)


def _dtype_bytes(tvm_dtype: str) -> int:
    return _TVM_DTYPE_BYTES.get(tvm_dtype, 4)


def _prod(shape) -> int:
    """Product of a TVM shape tuple (all elements must be int-convertible)."""
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


# ---------------------------------------------------------------------------
# Memory-level inference
# ---------------------------------------------------------------------------

def _infer_memory_level(buf: "tir.Buffer") -> str:
    """Infer SoftHier memory level from a TVM Buffer's scope string."""
    scope = buf.scope if hasattr(buf, "scope") else ""
    if scope in ("local.fragment", "local", "shared"):
        return "L1"
    return "HBM"


# ---------------------------------------------------------------------------
# Expression stringification for TileEltwise
# ---------------------------------------------------------------------------

class _ExprStringifier:
    """Converts a TVM PrimExpr to a C expression string."""

    def stringify(self, expr) -> str:
        if expr is None:
            return "0"
        cls = type(expr).__name__
        method = getattr(self, f"_str_{cls}", self._str_generic)
        return method(expr)

    def _str_generic(self, expr) -> str:
        return str(expr)

    def _str_BufferLoad(self, expr) -> str:
        buf = expr.buffer.name
        indices = ", ".join(self.stringify(i) for i in expr.indices)
        return f"((fp16*){buf})[{indices}]"

    def _str_Add(self, expr) -> str:
        return f"({self.stringify(expr.a)} + {self.stringify(expr.b)})"

    def _str_Sub(self, expr) -> str:
        return f"({self.stringify(expr.a)} - {self.stringify(expr.b)})"

    def _str_Mul(self, expr) -> str:
        return f"({self.stringify(expr.a)} * {self.stringify(expr.b)})"

    def _str_Div(self, expr) -> str:
        return f"({self.stringify(expr.a)} / {self.stringify(expr.b)})"

    def _str_Cast(self, expr) -> str:
        c_type = _c_dtype(str(expr.dtype))
        return f"(({c_type}){self.stringify(expr.value)})"

    def _str_FloatImm(self, expr) -> str:
        return repr(float(expr.value))

    def _str_IntImm(self, expr) -> str:
        return str(int(expr.value))

    def _str_Var(self, expr) -> str:
        return str(expr.name)

    def _str_Call(self, expr) -> str:
        # T.if_then_else → ternary
        op_name = str(expr.op) if hasattr(expr, "op") else ""
        if "if_then_else" in op_name:
            cond, t, f = [self.stringify(a) for a in expr.args]
            return f"(({cond}) ? {t} : {f})"
        # Fallback
        args = ", ".join(self.stringify(a) for a in expr.args)
        return f"{op_name}({args})"

    def _str_GT(self, expr) -> str:
        return f"({self.stringify(expr.a)} > {self.stringify(expr.b)})"

    def _str_GE(self, expr) -> str:
        return f"({self.stringify(expr.a)} >= {self.stringify(expr.b)})"

    def _str_LT(self, expr) -> str:
        return f"({self.stringify(expr.a)} < {self.stringify(expr.b)})"

    def _str_LE(self, expr) -> str:
        return f"({self.stringify(expr.a)} <= {self.stringify(expr.b)})"

    def _str_EQ(self, expr) -> str:
        return f"({self.stringify(expr.a)} == {self.stringify(expr.b)})"

    def _str_NE(self, expr) -> str:
        return f"({self.stringify(expr.a)} != {self.stringify(expr.b)})"


_STRINGIFIER = _ExprStringifier()


# ---------------------------------------------------------------------------
# Intrinsic call detection helpers
# ---------------------------------------------------------------------------

def _is_call(node, op_name: str) -> bool:
    """Return True if *node* is a TVM Call expression with op matching *op_name*."""
    if not hasattr(node, "op"):
        return False
    op = str(node.op)
    return op_name in op


def _call_args(node) -> list:
    return list(node.args) if hasattr(node, "args") else []


# ---------------------------------------------------------------------------
# Main visitor
# ---------------------------------------------------------------------------

class TilelangVisitor:
    """Walk a TVM ``tir.PrimFunc`` and produce a Deeploy ``ExecutionBlock``.

    Parameters
    ----------
        cluster_id : Optional[int]
                Default cluster ID used as fallback when no explicit or inferred
                cluster is available.
        cluster_policy : str
                Cluster assignment policy.
                - ``"explicit_attr"``: only ``T.attr(..., "cluster_id", ...)`` plus
                    fallback defaults.
                - ``"block_idx"``: infer cluster from block indices (bx/by/bz) when
                    available, then fallback defaults.
                - ``"hybrid"``: explicit attr overrides block-index inference, then
                    fallback defaults.
        num_clusters : Optional[int]
                Number of clusters for modulo mapping from block id to cluster id.
                If omitted, inferred block id is used directly.
    """

    def __init__(
        self,
        cluster_id: Optional[int] = None,
        cluster_policy: str = "hybrid",
        num_clusters: Optional[int] = None,
        cluster_ids: Optional[List[int]] = None,
        group_registry: Optional[ClusterGroupRegistry] = None,
    ):
        if not _TVM_AVAILABLE:
            raise ImportError(
                "TVM is required for TilelangVisitor."
            )
        self.cluster_id: Optional[int] = cluster_id
        self.cluster_policy = cluster_policy
        self.group_registry: Optional[ClusterGroupRegistry] = group_registry

        # Canonical cluster ID list: supports non-contiguous IDs like [0, 2, 5].
        # If num_clusters is set without cluster_ids, derive contiguous list.
        if cluster_ids is not None:
            self.cluster_ids: Optional[List[int]] = list(cluster_ids)
        elif num_clusters is not None:
            self.cluster_ids = list(range(num_clusters))
        else:
            self.cluster_ids = None
        self.num_clusters = len(self.cluster_ids) if self.cluster_ids else num_clusters

        if self.cluster_policy not in ("explicit_attr", "block_idx", "hybrid"):
            raise ValueError(
                f"Unsupported cluster_policy='{self.cluster_policy}'. "
                "Use one of: explicit_attr, block_idx, hybrid."
            )

        # State accumulated during visit
        self._ctxt: Optional[NetworkContext] = None
        self._eb: Optional[ExecutionBlock] = None
        self._bindings: Optional[TileBindingPipeline] = None

        # Buffers encountered in this PrimFunc:
        # name -> tir.Buffer
        self._global_bufs: Dict[str, "tir.Buffer"] = {}  # PrimFunc params (HBM)
        self._local_bufs:  Dict[str, "tir.Buffer"] = {}  # alloc_fragment  (L1)

        # L1 alloc order (for ordered dealloc)
        self._alloc_order: List[str] = []

        # Cluster-fallback context for current PrimFunc
        self._primfunc_cluster_id: Optional[int] = None

        # Block-index context (symbolic expression + extents) for cluster inference
        self._block_axes: Dict[str, str] = {}
        self._block_extents: Dict[str, Union[int, str]] = {}

        # Current cluster_group annotation context (set by _visit_AttrStmt)
        self._current_group_id: Optional[str] = None

        # supported op and intrinsic patterns (extend as needed)
        self._supported_ops = {
            "copy": self._handle_copy,
            "reduce": self._handle_reduce,
            "fill": self._handle_fill,
            "gemm_py": self._handle_gemm_py,
            "collective": self._handle_collective,
            "alloc_reducer": self._handle_alloc_reducer_intrinsic,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # def visit(self, primfunc, ctxt: NetworkContext) -> ExecutionBlock:
    #     """Walk *primfunc* and return a populated ``ExecutionBlock``.

    #     Parameters
    #     ----------
    #     primfunc : tir.PrimFunc
    #         TVM PrimFunc produced by ``jit_fn.get_tir(...)``.
    #     ctxt : NetworkContext
    #         Caller-provided context.  I/O buffers (HBM) are registered in
    #         ``ctxt.globalObjects``; L1 scratch buffers are registered in
    #         ``ctxt.localObjects``.

    #     Returns
    #     -------
    #     ExecutionBlock
    #         An ExecutionBlock whose CodeSnippets, when rendered by
    #         ``eb.generate(ctxt)``, produce the full kernel body.
    #     """
    #     pipeline = self.visit_bindings(primfunc, ctxt)
    #     raw_eb = pipeline.bind()
    #     self._ctxt, self._eb = pipeline.codeTransform(ctxt, raw_eb)
    #     return self._eb

    def visit_bindings(self, primfunc, ctxt: NetworkContext) -> TileBindingPipeline:
        """Walk *primfunc* and return ordered TileBindings.

        The returned pipeline can first materialize an ExecutionBlock via
        ``bind()`` and then run per-op transformations via ``codeTransform()``.
        """
        self._ctxt = ctxt
        self._eb   = ExecutionBlock()
        self._bindings = TileBindingPipeline()
        self._global_bufs = {}
        self._local_bufs  = {}
        self._alloc_order = []
        self._primfunc_cluster_id = None
        self._block_axes = {}
        self._block_extents = {}
        self._current_group_id = None

        # 1. Extract cluster_id from function attrs if present
        self._primfunc_cluster_id = None
        if hasattr(primfunc, "attrs") and primfunc.attrs is not None:
            try:
                self._primfunc_cluster_id = int(primfunc.attrs["cluster_id"])
            except (KeyError, TypeError):
                pass

        # 2. Register PrimFunc parameters as HBM buffers
        self._register_params(primfunc)

        # 3. Walk the body
        self._visit_stmt(primfunc.body, cluster_id=None)

        return self._bindings

    def _emit_binding(self, op_kind: str, template: NodeTemplate, rep: Dict, code_transformer: Optional[CodeTransformation] = None) -> None:
        """Record one TileBinding operation in visit order.

        Automatically injects ``shard_metadata`` (from ``_current_group_id``)
        into *rep* when it is not already present.  This ensures every binding
        emitted inside a ``T.attr("anno", "cluster_group", ...)`` block is
        tagged with the correct group metadata, which ``CollectiveLoweringPass``
        uses to detect which groups need a ``GridSyncGroupInfo`` init prefix.
        """
        if self._bindings is None:
            raise RuntimeError("TileBinding pipeline is not initialized. Call visit_bindings() first.")
        if "shard_metadata" not in rep:
            rep["shard_metadata"] = self._make_shard_metadata()
        self._bindings.add(TileBinding(op_kind=op_kind, template=template, operator_representation=rep, code_transformer=code_transformer))

    def _fallback_cluster(self):
        if self._primfunc_cluster_id is not None:
            return self._primfunc_cluster_id, "primfunc_attr"
        if self.cluster_id is not None:
            return self.cluster_id, "constructor_default"
        return None, "none"

    def _make_shard_metadata(self) -> Optional[ShardMetadata]:
        """Return a ShardMetadata for the current group context, or None."""
        if self._current_group_id is None:
            return None
        strategy_str: Optional[str] = None
        if self.group_registry is not None and self.group_registry.contains(self._current_group_id):
            group = self.group_registry.get(self._current_group_id)
            if group.strategy is not None:
                strategy_str = group.strategy.value
        return ShardMetadata(
            group_id=self._current_group_id,
            parallelism_strategy=strategy_str,
        )

    def _infer_cluster_from_block_idx(self):
        bx = self._block_axes.get("bx")
        by = self._block_axes.get("by")
        bz = self._block_axes.get("bz")

        if bx is None and by is None and bz is None:
            return None, None

        gy = self._block_extents.get("by", 1)

        if bz is not None:
            by_expr = by if by is not None else "0"
            bx_expr = bx if bx is not None else "0"
            block_id = f"((({bz}) * ({gy}) + ({by_expr})) * ({self._block_extents.get('bx', 1)}) + ({bx_expr}))"
        elif bx is not None and by is not None:
            block_id = f"(({bx}) * ({gy}) + ({by}))"
        else:
            block_id = bx or by

        if self.cluster_ids is not None:
            # List-based mapping: cluster_id resolved via lookup table in
            # ClusterGuardTransformationPass.  Return None so no scalar guard
            # is emitted; the cluster_map + block_id_expr in op metadata
            # drive the lookup-table guard instead.
            return None, block_id
        return block_id, block_id

    def _resolve_cluster(self, explicit_cluster_id):
        inferred_cluster_id, block_id_expr = self._infer_cluster_from_block_idx()
        fallback_cluster_id, fallback_source = self._fallback_cluster()

        if self.cluster_policy == "explicit_attr":
            if explicit_cluster_id is not None:
                return explicit_cluster_id, "explicit_attr", block_id_expr
            return fallback_cluster_id, fallback_source, block_id_expr

        if self.cluster_policy == "block_idx":
            if inferred_cluster_id is not None:
                return inferred_cluster_id, "block_idx", block_id_expr
            return fallback_cluster_id, fallback_source, block_id_expr

        # hybrid
        if explicit_cluster_id is not None:
            return explicit_cluster_id, "explicit_attr", block_id_expr
        if inferred_cluster_id is not None:
            return inferred_cluster_id, "block_idx", block_id_expr
        return fallback_cluster_id, fallback_source, block_id_expr

    def _infer_memory_level_from_name(self, buf_name: str) -> str:
        if buf_name in self._global_bufs:
            return "HBM"
        if buf_name in self._local_bufs:
            return "L1"
        buf = self._local_bufs.get(buf_name) or self._global_bufs.get(buf_name)
        if buf is not None:
            return _infer_memory_level(buf)
        return "unknown"

    def _lookup_registered_buffer(self, buf_name: str):
        if self._ctxt is None:
            return None
        try:
            return self._ctxt.lookup(buf_name)
        except Exception:
            return None

    def _buffer_cluster_id(self, buf_name: str):
        registered = self._lookup_registered_buffer(buf_name)
        if registered is not None and hasattr(registered, "cluster_id"):
            return registered.cluster_id

        if buf_name in self._local_bufs:
            local_registered = self._ctxt.localObjects.get(buf_name) if self._ctxt is not None else None
            if local_registered is not None and hasattr(local_registered, "cluster_id"):
                return local_registered.cluster_id

        if buf_name in self._global_bufs:
            global_registered = self._ctxt.globalObjects.get(buf_name) if self._ctxt is not None else None
            if global_registered is not None and hasattr(global_registered, "cluster_id"):
                return global_registered.cluster_id

        return None

    def _region_dims(self, region) -> List[Union[int, str]]:
        # args[1] is the access mask, NOT ndim. Derive ndim from indices length.
        args = list(region.args) if hasattr(region, "args") else []
        if len(args) < 3:
            return []
        ndim = self._region_ndim(region)
        dims = []
        for d in args[2: 2 + ndim]:
            try:
                dims.append(int(d))
            except (TypeError, ValueError):
                dims.append(_STRINGIFIER.stringify(d))
        return dims

    def _region_indices(self, region) -> List[str]:
        args = list(region.args) if hasattr(region, "args") else []
        if not args:
            return []
        first = args[0]
        if hasattr(first, "indices") and first.indices:
            return [_STRINGIFIER.stringify(idx) for idx in first.indices]
        return []

    def _region_metadata(self, region) -> Dict:
        buf_name = self._region_buf_name(region)
        buf = self._local_bufs.get(buf_name) or self._global_bufs.get(buf_name)
        dtype = str(buf.dtype) if buf is not None else "float16"
        shape = [int(d) for d in buf.shape] if buf is not None else []
        cluster_id = self._buffer_cluster_id(buf_name)

        dims = self._region_dims(region)
        return {
            "buffer": buf_name,
            "cluster_id": cluster_id,
            "memory_level": self._infer_memory_level_from_name(buf_name),
            "dtype": dtype,
            "shape": shape,
            "ndim": len(dims),
            "dims": dims,
            "indices": self._region_indices(region),
            "access_mask": self._region_access_mask(region),
            "byte_offset": self._region_byte_offset(region),
            "nbytes": self._region_nbytes(region),
        }

    def _cluster_from_buffers(self, buffer_names: List[str]):
        cluster_ids = []
        for buf_name in buffer_names:
            cluster_id = self._buffer_cluster_id(buf_name)
            if cluster_id is not None:
                cluster_ids.append(cluster_id)

        if not cluster_ids:
            return None, None

        first_cluster_id = cluster_ids[0]
        if all(cluster_id == first_cluster_id for cluster_id in cluster_ids[1:]):
            return first_cluster_id, "buffer_cluster"

        return cluster_ids, "buffer_cluster_mixed"

    def _cluster_from_regions(self, regions: List):
        cluster_ids = []
        for region in regions:
            region_cluster_id = self._region_metadata(region).get("cluster_id")
            if region_cluster_id is not None:
                cluster_ids.append(region_cluster_id)

        if not cluster_ids:
            return None, None

        first_cluster_id = cluster_ids[0]
        if all(cluster_id == first_cluster_id for cluster_id in cluster_ids[1:]):
            return first_cluster_id, "region_cluster"

        return cluster_ids, "region_cluster_mixed"

    def _op_metadata(
        self,
        explicit_cluster_id,
        src_regions: Optional[List] = None,
        dst_regions: Optional[List] = None,
        src_buffers: Optional[List[str]] = None,
        dst_buffers: Optional[List[str]] = None,
    ) -> Tuple[Union[int, str, None], Dict]:
        src_regions = src_regions or []
        dst_regions = dst_regions or []
        src_buffers = src_buffers or []
        dst_buffers = dst_buffers or []
        all_regions = src_regions + dst_regions
        all_buffers = src_buffers + dst_buffers

        region_cluster_id, region_cluster_source = self._cluster_from_regions(all_regions)
        buffer_cluster_id, buffer_cluster_source = self._cluster_from_buffers(all_buffers)

        # Always resolve through the standard priority chain so that an explicit
        # T.attr cluster_id annotation always wins.  Buffer/region cluster IDs
        # are kept as informational metadata only — they must never be used as
        # the execution cluster_id because cross-cluster ops involve buffers from
        # different clusters and would produce an invalid list value.
        cluster_id, cluster_source, block_id_expr = self._resolve_cluster(explicit_cluster_id)

        src_region_metadata = [self._region_metadata(region) for region in src_regions]
        dst_region_metadata = [self._region_metadata(region) for region in dst_regions]

        return cluster_id, {
            "cluster_source": cluster_source,
            "cluster_map": list(self.cluster_ids) if self.cluster_ids else None,
            "block_indices": dict(self._block_axes),
            "block_extents": dict(self._block_extents),
            "block_id_expr": block_id_expr,
            "attached_region_cluster_id": region_cluster_id,
            "attached_buffer_cluster_id": buffer_cluster_id,
            "src_buffers": src_buffers,
            "dst_buffers": dst_buffers,
            "src_buffer_cluster_ids": {name: self._buffer_cluster_id(name) for name in src_buffers},
            "dst_buffer_cluster_ids": {name: self._buffer_cluster_id(name) for name in dst_buffers},
            "src_memory_levels": {name: self._infer_memory_level_from_name(name) for name in src_buffers},
            "dst_memory_levels": {name: self._infer_memory_level_from_name(name) for name in dst_buffers},
            "src_regions": src_region_metadata,
            "dst_regions": dst_region_metadata,
        }

    def _thread_axis_name(self, stmt, loop_var: str) -> Optional[str]:
        if loop_var in ("bx", "by", "bz"):
            return loop_var

        thread_binding = getattr(stmt, "thread_binding", None)
        thread_tag = str(thread_binding) if thread_binding is not None else ""
        if "blockIdx.x" in thread_tag:
            return "bx"
        if "blockIdx.y" in thread_tag:
            return "by"
        if "blockIdx.z" in thread_tag:
            return "bz"
        return None

    # ------------------------------------------------------------------
    # Internal: cluster inference helpers
    # ------------------------------------------------------------------
    def _collect_buffer_cluster_hints(self, stmt, cluster_id, hints: Optional[Dict[str, Union[int, str]]] = None) -> Dict[str, Union[int, str]]:
        if hints is None:
            hints = {}
        if stmt is None:
            return hints

        cls = type(stmt).__name__
        if cls == "AttrStmt":
            next_cluster_id = cluster_id
            if hasattr(stmt, "node") and str(stmt.node) == "anno" and hasattr(stmt, "attr_key") and str(stmt.attr_key) == "cluster_id":
                try:
                    next_cluster_id = int(stmt.value)
                except (KeyError, TypeError, ValueError):
                    next_cluster_id = cluster_id
            return self._collect_buffer_cluster_hints(stmt.body, next_cluster_id, hints)

        if cls == "Evaluate":
            value = stmt.value
            if hasattr(value, "args"):
                for arg in value.args:
                    buf_name = self._region_buf_name(arg)
                    if buf_name != "unknown" and buf_name not in hints and cluster_id is not None:
                        hints[buf_name] = cluster_id
            return hints

        if cls == "BufferStore":
            buf_name = stmt.buffer.name
            if buf_name not in hints and cluster_id is not None:
                hints[buf_name] = cluster_id
            return hints

        if cls == "SeqStmt":
            for child in stmt.seq:
                self._collect_buffer_cluster_hints(child, cluster_id, hints)
            return hints

        if cls == "For":
            return self._collect_buffer_cluster_hints(stmt.body, cluster_id, hints)

        if cls == "BlockRealize":
            return self._collect_buffer_cluster_hints(stmt.block, cluster_id, hints)

        if cls == "Block":
            # Buffer declarations (T.alloc_fragment) in this Block were created
            # under the current cluster_id.  Record them unconditionally so the
            # declaration cluster always wins over any later usage hints.
            for alloc_buf in getattr(stmt, "alloc_buffers", []):
                if cluster_id is not None:
                    hints[alloc_buf.name] = cluster_id
            return self._collect_buffer_cluster_hints(stmt.body, cluster_id, hints)

        for attr in ("body", "then_case", "else_case"):
            child = getattr(stmt, attr, None)
            if child is not None:
                self._collect_buffer_cluster_hints(child, cluster_id, hints)
        return hints

    # ------------------------------------------------------------------
    # Internal: parameter registration
    # ------------------------------------------------------------------

    def _register_params(self, primfunc):
        """Register PrimFunc buffer parameters as HBM VariableBuffers."""
        for param in primfunc.params:
            if param in primfunc.buffer_map:
                buf = primfunc.buffer_map[param]
                self._global_bufs[buf.name] = buf
                # Register in context (global = HBM)
                try:
                    from Deeploy.Targets.SoftHier.Platform import SoftHierDynamicBuffer
                    db = SoftHierDynamicBuffer(
                        name         = buf.name,
                        shape        = [int(d) for d in buf.shape],
                        memory_level = "HBM",
                        cluster_id   = self._primfunc_cluster_id if self._primfunc_cluster_id is not None else self.cluster_id,
                    )
                    # _type/_instance are required by NetworkContext.lookup;
                    # we leave them unset here — code generation accesses
                    # the buffer by name directly via OperatorRepresentation.
                    self._ctxt.globalObjects[buf.name] = db
                except Exception:
                    pass  # context registration is best-effort

    def _register_local_buf(self, buf: "tir.Buffer", cluster_id, group_id: Optional[str] = None):
        """Register an alloc_fragment buffer as an L1 DynamicBuffer."""
        self._local_bufs[buf.name] = buf
        self._alloc_order.append(buf.name)
        try:
            from Deeploy.Targets.SoftHier.Platform import SoftHierDynamicBuffer
            nbytes = _prod(buf.shape) * _dtype_bytes(str(buf.dtype))
            db = SoftHierDynamicBuffer(
                name         = buf.name,
                shape        = [int(d) for d in buf.shape],
                memory_level = "L1",
                cluster_id   = cluster_id,
            )
            # Tag the buffer with group_id for downstream analysis
            if group_id is not None:
                db._group_id = group_id
            self._ctxt.localObjects[buf.name] = db
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal: statement dispatch
    # ------------------------------------------------------------------

    def _visit_stmt(self, stmt, cluster_id):
        """Dispatch a TVM Stmt node to the appropriate visitor method."""
        if stmt is None:
            return
        cls = type(stmt).__name__
        method = getattr(self, f"_visit_{cls}", self._visit_generic)
        method(stmt, cluster_id)

    def _visit_generic(self, stmt, cluster_id):
        """Fallback: recurse into known child fields."""
        for attr in ("body", "then_case", "else_case"):
            child = getattr(stmt, attr, None)
            if child is not None:
                self._visit_stmt(child, cluster_id)

    def _visit_BlockRealize(self, stmt, cluster_id):
        self._visit_stmt(stmt.block, cluster_id)

    def _peek_cluster_group(self, stmt) -> Optional[str]:
        """Scan the top-level AttrStmt chain in *stmt* for a 'cluster_group' annotation.

        TileLang places ``T.attr("anno", "cluster_group", ...)`` as an
        AttrStmt that wraps the Block body.  Since ``alloc_buffers`` in the
        Block are processed before the body is visited, ``_current_group_id``
        is not yet set at alloc-emit time.  This helper pre-scans to detect
        the group so allocs can be tagged correctly.
        """
        node = stmt
        while node is not None and type(node).__name__ == "AttrStmt":
            if (hasattr(node, "node") and str(node.node) == "anno"
                    and hasattr(node, "attr_key")
                    and str(node.attr_key) == "cluster_group"):
                return str(node.value).strip('"').strip("'")
            node = getattr(node, "body", None)
        return None

    def _visit_Block(self, stmt, cluster_id):
        # TODO: consider adding sync at block boundaries if needed (e.g. after a producer block before a consumer block)
        # Allocate local buffers declared in this block
        block_cluster_hints = self._collect_buffer_cluster_hints(stmt.body, cluster_id)
        # Pre-scan body for a cluster_group annotation so allocs are tagged correctly
        # even though the AttrStmt is inside the body (not wrapping the Block itself).
        current_group_id = self._current_group_id or self._peek_cluster_group(stmt.body)
        # Build a ShardMetadata for the alloc/free reps using the pre-scanned group_id
        alloc_shard_meta = (
            self._make_shard_metadata()
            if self._current_group_id is not None
            else (
                ShardMetadata(
                    group_id=current_group_id,
                    # parallelism_strategy=(
                    #     self.group_registry.get(current_group_id).strategy.value
                    #     if (current_group_id is not None
                    #         and self.group_registry is not None
                    #         and self.group_registry.contains(current_group_id))
                    #     else None
                    # ),
                )
                if current_group_id is not None
                else None
            )
        )

        for buf in getattr(stmt, "alloc_buffers", []):
            alloc_cluster_id = block_cluster_hints.get(buf.name, cluster_id)
            self._register_local_buf(buf, alloc_cluster_id, group_id=current_group_id)
            nbytes = _prod(buf.shape) * _dtype_bytes(str(buf.dtype))
            resolved_cluster_id, op_metadata = self._op_metadata(
                alloc_cluster_id,
                dst_buffers=[buf.name],
            )
            rep = {
                "name":           buf.name,
                "dtype":          _c_dtype(str(buf.dtype)),
                "nbytes":         nbytes,
                "cluster_id":     resolved_cluster_id,
                "cluster_map":    op_metadata.get("cluster_map"),
                "block_id_expr":  op_metadata.get("block_id_expr"),
                "metadata":       op_metadata,
                "shard_metadata": alloc_shard_meta,
            }
            self._emit_binding("alloc", TileAllocTemplate, rep)

        self._visit_stmt(stmt.body, cluster_id)

        # Emit frees in reverse alloc order
        for name in reversed(self._alloc_order):
            free_cluster_id = block_cluster_hints.get(name, cluster_id)
            resolved_cluster_id, op_metadata = self._op_metadata(
                free_cluster_id,
                src_buffers=[name],
            )
            rep = {
                "name":           name,
                "cluster_id":     resolved_cluster_id,
                "cluster_map":    op_metadata.get("cluster_map"),
                "block_id_expr":  op_metadata.get("block_id_expr"),
                "metadata":       op_metadata,
                "shard_metadata": alloc_shard_meta,
            }
            self._emit_binding("free", TileFreeTemplate, rep)
        # TODO: consider deallocating local buffers at the end of their block scope, align with cluster id
        # Only do this once per block
        self._alloc_order = []

    def _visit_AttrStmt(self, stmt, cluster_id):
        """Handle TVM AttrStmt nodes.

        Four cases are distinguished:

        * ``node == "anno"`` and ``attr_key == "cluster_id"`` — extracts the
          integer cluster id and propagates it to the subtree.
        * ``node == "anno"`` and ``attr_key == "cluster_group"`` — sets the
          current group context (``_current_group_id``) and propagates it to
          the subtree; all operations emitted within this block are tagged with
          the group's ``ShardMetadata``.
        * ``attr_key == "thread_extent"`` with ``blockIdx.*`` thread tag —
          emits a C for-loop and tracks block axes for cluster inference.
        * All other AttrStmt nodes (e.g. ``threadIdx.*``) — recurse into
          the body unchanged.
        """
        if hasattr(stmt, "node") and str(stmt.node) == "anno" and hasattr(stmt, "attr_key"):
            attr_key = str(stmt.attr_key)
            if attr_key == "cluster_id":
                try:
                    cluster_id = int(stmt.value)
                except (KeyError, TypeError, ValueError):
                    pass
            elif attr_key == "cluster_group":
                # Enter a cluster_group annotation block
                group_id = str(stmt.value).strip('"').strip("'")
                prev_group_id = self._current_group_id
                self._current_group_id = group_id
                self._visit_stmt(stmt.body, cluster_id)
                self._current_group_id = prev_group_id
                return

        # Handle thread_extent AttrStmt for blockIdx axes —
        # emit C for-loops so bx/by/bz variables are declared.
        raw_attr_key = getattr(stmt, "attr_key", None)
        if raw_attr_key is not None and str(raw_attr_key) == "thread_extent":
            node = stmt.node
            if hasattr(node, "var") and hasattr(node, "thread_tag"):
                thread_tag = str(node.thread_tag)
                if "blockIdx" in thread_tag:
                    loop_var = str(node.var.name)
                    extent = int(stmt.value)
                    if "blockIdx.x" in thread_tag:
                        axis_name = "bx"
                    elif "blockIdx.y" in thread_tag:
                        axis_name = "by"
                    else:
                        axis_name = "bz"

                    # Save previous axis state
                    prev_axis_expr = self._block_axes.get(axis_name)
                    prev_axis_extent = self._block_extents.get(axis_name)
                    self._block_axes[axis_name] = loop_var
                    self._block_extents[axis_name] = extent

                    # Emit for-loop (no cluster guard on the brace itself)
                    rep_open = {
                        "loop_var": loop_var,
                        "min_val": 0,
                        "extent": extent,
                        "cluster_id": None,
                    }
                    self._emit_binding("for_open", ForLoopOpenTemplate, rep_open)
                    self._visit_stmt(stmt.body, cluster_id)
                    self._emit_binding("for_close", ForLoopCloseTemplate, {"loop_var": loop_var})

                    # Restore axis state
                    if prev_axis_expr is None:
                        self._block_axes.pop(axis_name, None)
                    else:
                        self._block_axes[axis_name] = prev_axis_expr
                    if prev_axis_extent is None:
                        self._block_extents.pop(axis_name, None)
                    else:
                        self._block_extents[axis_name] = prev_axis_extent
                    return
                # threadIdx: just recurse (hardware threads, not emulated in C)

        self._visit_stmt(stmt.body, cluster_id)

    def _visit_SeqStmt(self, stmt, cluster_id):
        for s in stmt.seq:
            self._visit_stmt(s, cluster_id)

    def _visit_Evaluate(self, stmt, cluster_id):
        # op
        """Dispatch based on the intrinsic call inside."""
        value = stmt.value
        if not hasattr(value, "op"):
            return
        op_str = str(value.op.name).split(".")[-1] if hasattr(value.op, "name") else ""
        args   = list(value.args) if hasattr(value, "args") else []

        handler = self._supported_ops.get(op_str)
        if handler:
            handler(args, cluster_id)
        else:
            # Unknown intrinsic — emit a comment
            print(f"[TilelangVisitor] Warning: unhandled intrinsic call: {op_str}")
            self._emit_binding(
                "comment",
                NodeTemplate(f"// Unhandled intrinsic: {op_str}\n"),
                {
                    "cluster_id": self._resolve_cluster(cluster_id)[0],
                    "metadata": self._op_metadata(cluster_id)[1],
                },
            )

    def _visit_For(self, stmt, cluster_id):
        """Serial or parallel For loop."""
        loop_var = str(stmt.loop_var.name)
        min_val  = int(stmt.min) if hasattr(stmt.min, "__int__") else str(stmt.min)
        extent   = int(stmt.min) + int(stmt.extent) if hasattr(stmt.extent, "__int__") else str(stmt.extent)

        # ForKind: 0=Serial, 1=Parallel, 2=Unroll, 3=Vectorized, 4=ThreadBinding
        # TODO: dealing with pipeline annotations if specified
        for_kind = int(stmt.kind)

        axis_name = self._thread_axis_name(stmt, loop_var)
        prev_axis_expr = self._block_axes.get(axis_name) if axis_name is not None else None
        prev_axis_extent = self._block_extents.get(axis_name) if axis_name is not None else None
        if axis_name is not None:
            self._block_axes[axis_name] = loop_var
            self._block_extents[axis_name] = int(stmt.extent) if hasattr(stmt.extent, "__int__") else str(stmt.extent)

        # Parallel For: map to TileEltwise if body is a BufferStore
        if for_kind == 1:  # Parallel
            self._handle_parallel_for(stmt, loop_var, int(stmt.extent), cluster_id)
        else:
            # Serial: emit open/close brackets and recurse
            rep_open = {
                "loop_var":   loop_var,
                "min_val":    min_val,
                "extent":     int(stmt.extent),
                "cluster_id": None,  # loop brace itself has no cluster guard
            }
            self._emit_binding("for_open", ForLoopOpenTemplate, rep_open)
            self._visit_stmt(stmt.body, cluster_id)
            self._emit_binding("for_close", ForLoopCloseTemplate, {"loop_var": loop_var})

        if axis_name is not None:
            if prev_axis_expr is None:
                self._block_axes.pop(axis_name, None)
            else:
                self._block_axes[axis_name] = prev_axis_expr

            if prev_axis_extent is None:
                self._block_extents.pop(axis_name, None)
            else:
                self._block_extents[axis_name] = prev_axis_extent

    # ------------------------------------------------------------------
    # Intrinsic handlers
    # ------------------------------------------------------------------

    def _handle_copy(self, args: list, cluster_id):
        """T.copy(src_region, dst_region) → TileLoad / TileStore / TileCopy."""
        if len(args) < 2:
            return

        # args[0] = src region Call, args[1] = dst region Call
        src_region = args[0]
        dst_region = args[1]

        src_buf_name = self._region_buf_name(src_region)
        dst_buf_name = self._region_buf_name(dst_region)
        nbytes       = self._region_nbytes(src_region)
        src_offset   = self._region_byte_offset(src_region)
        dst_offset   = self._region_byte_offset(dst_region)

        src_is_hbm = src_buf_name in self._global_bufs
        dst_is_hbm = dst_buf_name in self._global_bufs
        resolved_cluster_id, op_metadata = self._op_metadata(
            cluster_id,
            src_regions=[src_region],
            dst_regions=[dst_region],
            src_buffers=[src_buf_name],
            dst_buffers=[dst_buf_name],
        )

        # check for cluster_id annotation
        if src_is_hbm and not dst_is_hbm:
            # HBM → L1: TileLoad
            rep = {
                "src":        src_buf_name,
                "dst":        dst_buf_name,
                "nbytes":     nbytes,
                "src_offset": src_offset,
                "cluster_id": resolved_cluster_id,
                "metadata":   op_metadata,
            }
            self._emit_binding("load", TileLoadTemplate, rep)

        elif not src_is_hbm and dst_is_hbm:
            # L1 → HBM: TileStore
            rep = {
                "src":        src_buf_name,
                "dst":        dst_buf_name,
                "nbytes":     nbytes,
                "dst_offset": dst_offset,
                "cluster_id": resolved_cluster_id,
                "metadata":   op_metadata,
            }
            self._emit_binding("store", TileStoreTemplate, rep)

        else:
            # L1 → L1: TileCopy
            rep = {
                "src":        src_buf_name,
                "dst":        dst_buf_name,
                "nbytes":     nbytes,
                "cluster_id": resolved_cluster_id,
                "metadata":   op_metadata,
            }
            self._emit_binding("copy", TileCopyTemplate, rep)

        # Insert sync after every data movement
        self._emit_binding("sync", TileSyncTemplate, {
            "cluster_id": resolved_cluster_id,
            "metadata": op_metadata,
        })

    def _handle_reduce(self, args: list, cluster_id):
        """T.reduce(src_slice, dst_slice, op_str, dim, clear_flag) → TileReduce."""
        if len(args) < 3:
            return
        src_region = args[0]
        dst_region = args[1]
        op_str     = str(args[2]).strip('"')  # "sum" / "max" / "min"
        resolved_cluster_id, op_metadata = self._op_metadata(
            cluster_id,
            src_regions=[src_region],
            dst_regions=[dst_region],
            src_buffers=[self._region_buf_name(src_region)],
            dst_buffers=[self._region_buf_name(dst_region)],
        )

        src_buf = self._region_buf_name(src_region)
        dst_buf = self._region_buf_name(dst_region)

        src_tvm_buf = (self._local_bufs.get(src_buf) or
                       self._global_bufs.get(src_buf))
        dst_tvm_buf = (self._local_bufs.get(dst_buf) or
                       self._global_bufs.get(dst_buf))

        if src_tvm_buf is None:
            extent = 1
            outer  = 1
            dtype  = "fp16"
        else:
            shape  = [int(d) for d in src_tvm_buf.shape]
            dtype  = _c_dtype(str(src_tvm_buf.dtype))
            # dim=1 (reduce along last dim) is the common case from test_AST
            outer  = shape[0] if len(shape) >= 2 else 1
            extent = shape[-1]

        rep = {
            "src":        src_buf,
            "dst":        dst_buf,
            "op":         op_str,
            "outer":      outer,
            "extent":     extent,
            "dtype":      dtype,
            "cluster_id": resolved_cluster_id,
            "metadata":   op_metadata,
        }
        self._emit_binding("reduce", TileReduceTemplate, rep)
        self._emit_binding("sync", TileSyncTemplate, {
            "cluster_id": resolved_cluster_id,
            "metadata": op_metadata,
        })

    def _handle_fill(self, args: list, cluster_id):
        """T.fill(region, val) → TileFill."""
        if len(args) < 2:
            return
        region = args[0]
        val    = 0
        try:
            val = int(args[1])
        except (TypeError, ValueError):
            val = str(args[1])

        buf_name = self._region_buf_name(region)
        nbytes   = self._region_nbytes(region)
        resolved_cluster_id, op_metadata = self._op_metadata(
            cluster_id,
            dst_regions=[region],
            dst_buffers=[buf_name],
        )

        rep = {
            "buf":        buf_name,
            "val":        val,
            "nbytes":     nbytes,
            "cluster_id": resolved_cluster_id,
            "metadata":   op_metadata,
        }
        self._emit_binding("fill", TileFillTemplate, rep)
        self._emit_binding("sync", TileSyncTemplate, {
            "cluster_id": resolved_cluster_id,
            "metadata": op_metadata,
        })

    def _handle_gemm_py(self, args: list, cluster_id):
        """T.gemm(A_region, B_region, C_region, M, N, K) → TileGemm."""
        if len(args) < 8:
            return
        A_region = args[0]
        B_region = args[1]
        C_region = args[2]
        M        = int(args[5]) if hasattr(args[5], "__int__") else str(args[5])
        N        = int(args[6]) if hasattr(args[6], "__int__") else str(args[6])
        K        = int(args[7]) if hasattr(args[7], "__int__") else str(args[7])

        A_buf_name = self._region_buf_name(A_region)
        B_buf_name = self._region_buf_name(B_region)
        C_buf_name = self._region_buf_name(C_region)
        resolved_cluster_id, op_metadata = self._op_metadata(
            cluster_id,
            src_regions=[A_region, B_region],
            dst_regions=[C_region],
            src_buffers=[A_buf_name, B_buf_name],
            dst_buffers=[C_buf_name],
        )

        redmule_dtype_mapping = {
            'uint16': 'REDMULE_UINT_16',
            'int16':  'REDMULE_INT_16',
            'float16': 'REDMULE_FP_16',
            'uint8':  'REDMULE_UINT_8',
            'int8':   'REDMULE_INT_8',
        }
        A_buf = self._local_bufs.get(A_buf_name) or self._global_bufs.get(A_buf_name)
        A_dtype = str(A_buf.dtype) if A_buf is not None else "float16"
        redmule_dtype_str = redmule_dtype_mapping.get(A_dtype, "REDMULE_FP_16")

        rep = {
            "A":          A_buf_name,
            "B":          B_buf_name,
            "C":          C_buf_name,
            "M":          M,
            "N":          N,
            "K":          K,
            "cluster_id": resolved_cluster_id,
            "redmule_dtype": redmule_dtype_str,
            "metadata":   op_metadata,
        }
        self._emit_binding("gemm", TileGemmTemplate, rep)
        # TODO: consider adding a pass to fuse unnecessary syncs
        self._emit_binding("sync", TileSyncTemplate, {
            "cluster_id": resolved_cluster_id,
            "metadata": op_metadata,
        })

    def _handle_parallel_for(self, stmt, loop_var: str, extent: int, cluster_id):
        # TODO: consider using vector unit by SIMD ops
        """Parallel For with a BufferStore body → TileEltwise."""
        body = stmt.body
        # Unwrap nested parallel For loops
        while type(body).__name__ == "For" and int(body.kind) == 1:
            inner_body     = body.body
            inner_loop_var = str(body.loop_var.name)
            inner_extent   = int(body.extent)
            body = inner_body

        if type(body).__name__ == "BufferStore":
            dst_name  = body.buffer.name
            src_expr  = _STRINGIFIER.stringify(body.value)
            buf       = (self._local_bufs.get(dst_name) or
                         self._global_bufs.get(dst_name))
            dtype     = _c_dtype(str(buf.dtype)) if buf else "fp16"
            resolved_cluster_id, op_metadata = self._op_metadata(
                cluster_id,
                dst_buffers=[dst_name],
            )
            rep = {
                "dst":        dst_name,
                "src_expr":   src_expr,
                "loop_var":   loop_var,
                "extent":     extent,
                "dtype":      dtype,
                "cluster_id": resolved_cluster_id,
                "metadata":   op_metadata,
            }
            self._emit_binding("eltwise", TileEltwiseTemplate, rep)
            self._emit_binding("sync", TileSyncTemplate, {
                "cluster_id": resolved_cluster_id,
                "metadata": op_metadata,
            })
        else:
            # Fallback: emit a serial loop
            rep_open = {
                "loop_var":   loop_var,
                "min_val":    0,
                "extent":     extent,
                "cluster_id": None,
            }
            self._emit_binding("for_open", ForLoopOpenTemplate, rep_open)
            self._visit_stmt(stmt.body, cluster_id)
            self._emit_binding("for_close", ForLoopCloseTemplate, {"loop_var": loop_var})

    def _visit_BufferStore(self, stmt, cluster_id):
        """Scalar BufferStore outside a parallel loop — emit as TileEltwise(extent=1)."""
        dst_name = stmt.buffer.name
        src_expr = _STRINGIFIER.stringify(stmt.value)
        buf      = (self._local_bufs.get(dst_name) or
                    self._global_bufs.get(dst_name))
        dtype    = _c_dtype(str(buf.dtype)) if buf else "fp16"
        resolved_cluster_id, op_metadata = self._op_metadata(
            cluster_id,
            dst_buffers=[dst_name],
        )
        # Use a constant 0 index for simplicity
        loop_var = "_i_scalar"
        rep = {
            "dst":        dst_name,
            "src_expr":   src_expr,
            "loop_var":   loop_var,
            "extent":     1,
            "dtype":      dtype,
            "cluster_id": resolved_cluster_id,
            "metadata":   op_metadata,
        }
        self._emit_binding("eltwise", TileEltwiseTemplate, rep)

    def _handle_collective(self, args: list, cluster_id):
        """T.collective(src_buf, dst_buf, op, reduce_op, group) → CollectiveBinding.

        Expected argument layout (all as string/IntImm):
          args[0] : src buffer region or var
          args[1] : dst buffer region or var
          args[2] : op string, e.g. "allreduce"
          args[3] : reduce_op string, e.g. "sum"  (optional, default "sum")
          args[4] : group_id string               (optional, falls back to _current_group_id)
        """
        if len(args) < 2:
            return

        src_buf = self._region_buf_name(args[0]) if hasattr(args[0], "args") else str(args[0])
        dst_buf = self._region_buf_name(args[1]) if hasattr(args[1], "args") else str(args[1])
        op_str = str(args[2]).strip('"') if len(args) > 2 else "allreduce"
        reduce_op = str(args[3]).strip('"') if len(args) > 3 else "sum"
        group_id = str(args[4]).strip('"') if len(args) > 4 else self._current_group_id

        if group_id is None:
            print("[TilelangVisitor] Warning: T.collective() outside a cluster_group block; skipping.")
            return

        # Compute nbytes from the source buffer
        src_tvm_buf = self._local_bufs.get(src_buf) or self._global_bufs.get(src_buf)
        if src_tvm_buf is not None:
            nbytes = _prod(src_tvm_buf.shape) * _dtype_bytes(str(src_tvm_buf.dtype))
        else:
            nbytes = 0

        spec = CollectiveOpSpec(
            op=op_str,
            group_id=group_id,
            src_buffer=src_buf,
            dst_buffer=dst_buf,
            reduce_op=reduce_op,
        )
        shard_meta = self._make_shard_metadata()
        rep = {
            "src_name":       src_buf,
            "dst_name":       dst_buf,
            "op":             op_str,
            "group_id":       group_id,
            "nbytes":         nbytes,
            "cluster_id":     None,  # guard is inside template
            "shard_metadata": shard_meta,
        }
        self._emit_binding(
            "group_collective",
            NodeTemplate(f"// T.collective({op_str}) — lowered by CollectiveLoweringPass\n"),
            rep,
            code_transformer=None,
        )
        # Replace the placeholder with a real CollectiveBinding
        if self._bindings is not None:
            last = self._bindings.bindings[-1]
            cb = CollectiveBinding(
                op_kind="group_collective",
                template=last.template,
                operator_representation=rep,
                op_name=f"tile_collective_{op_str}_{group_id}",
                spec=spec,
            )
            self._bindings.bindings[-1] = cb

    def _handle_alloc_reducer_intrinsic(self, args: list, cluster_id):
        """T.alloc_reducer(buf_name_or_region, dtype, nbytes) → alloc_reducer binding.

        This is the intrinsic-call path (T.alloc_reducer used as a function).
        The more common path is via alloc_buffers in a Block — handled in
        _visit_Block when the buffer name ends with the reducer convention.

        Expected argument layout:
          args[0] : buffer name string or region
          args[1] : dtype string
          args[2] : nbytes int
        """
        if len(args) < 1:
            return

        buf_name = (self._region_buf_name(args[0])
                    if hasattr(args[0], "args") else str(args[0]))
        dtype = _c_dtype(str(args[1]).strip('"')) if len(args) > 1 else "fp16"
        nbytes = int(args[2]) if len(args) > 2 and hasattr(args[2], "__int__") else 0

        group_id = self._current_group_id
        shard_meta = self._make_shard_metadata()
        resolved_cluster_id, op_metadata = self._op_metadata(cluster_id)

        rep = {
            "name":             buf_name,
            "dtype":            dtype,
            "nbytes":           nbytes,
            "root_cluster_id":  None,  # resolved by CollectiveLoweringPass
            "group_id":         group_id,
            "cluster_id":       resolved_cluster_id,
            "metadata":         op_metadata,
            "shard_metadata":   shard_meta,
        }
        self._emit_binding("alloc_reducer", TileAllocReducerTemplate, rep)

    # ------------------------------------------------------------------
    # Region helpers
    # TODO: Consider extension here, as it carrys tile information
    # ------------------------------------------------------------------

    def _region_buf_name(self, region) -> str:
        """Extract the buffer name from a T.region(...) call.

        T.region layout (from RegionOp in TileLang lowering):
          args[0] : BufferLoad whose indices are per-axis *minima*.
          args[1] : Integer access mask (1=r, 2=w, 3=rw).
          args[2+i]: Extent of axis i.
        """
        if region is None:
            return "unknown"
        args = list(region.args) if hasattr(region, "args") else []
        if args:
            first = args[0]
            # BufferLoad → buffer.name
            if hasattr(first, "buffer"):
                return first.buffer.name
            # Var → name
            if hasattr(first, "name"):
                return str(first.name)
            # Fallback: stringify and extract identifier
            s = str(first)
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", s)
            if m:
                return m.group(1)
        return "unknown"

    def _region_ndim(self, region) -> int:
        """Number of axes in a T.region call.

        ndim is the length of args[0].indices (per-axis minima),
        NOT args[1] which is the access mask.
        """
        args = list(region.args) if hasattr(region, "args") else []
        if not args:
            return 0
        first = args[0]
        if hasattr(first, "indices") and first.indices:
            return len(first.indices)
        # Fallback: infer from buffer shape when indices are unavailable
        buf_name = self._region_buf_name(region)
        buf = self._local_bufs.get(buf_name) or self._global_bufs.get(buf_name)
        return len(buf.shape) if buf is not None else 0

    def _region_access_mask(self, region) -> int:
        """Access mask from a T.region call (1=read, 2=write, 3=read-write).

        Stored in args[1] of the RegionOp call.
        """
        args = list(region.args) if hasattr(region, "args") else []
        if len(args) < 2:
            return 3  # default to read-write when unknown
        try:
            return int(args[1])
        except (TypeError, ValueError):
            return 3

    def _region_nbytes(self, region) -> int:
        """Compute transfer size in bytes from a T.region(...) call.

        T.region layout:
          args[0] : BufferLoad whose indices are per-axis minima.
          args[1] : access mask (1=r, 2=w, 3=rw) — NOT ndim.
          args[2+i]: Extent of axis i.
        ndim is derived from len(args[0].indices).
        """
        args = list(region.args) if hasattr(region, "args") else []
        if len(args) < 3:
            return 0
        ndim = self._region_ndim(region)
        dims = args[2: 2 + ndim]
        n    = 1
        for d in dims:
            try:
                n *= int(d)
            except (TypeError, ValueError):
                pass

        # Infer dtype from the source buffer
        buf_name = self._region_buf_name(region)
        buf      = (self._local_bufs.get(buf_name) or
                    self._global_bufs.get(buf_name))
        dtype    = str(buf.dtype) if buf else "float16"
        return n * _dtype_bytes(dtype)

    def _dense_row_major_strides(self, shape: List[int]) -> List[int]:
        """Return dense row-major element strides for a shape."""
        if not shape:
            return []
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return strides

    def _region_byte_offset(self, region) -> int | str:
        """Compute byte offset from region indices using dense row-major layout."""
        args = list(region.args) if hasattr(region, "args") else []
        if not args:
            return 0

        # First arg is often a BufferLoad with indices encoding the offset.
        first = args[0]
        if not (hasattr(first, "indices") and first.indices):
            return 0

        indices = list(first.indices)
        if not indices:
            return 0

        buf_name = self._region_buf_name(region)
        buf = self._local_bufs.get(buf_name) or self._global_bufs.get(buf_name)
        elem_bytes = _dtype_bytes(str(buf.dtype)) if buf is not None else _dtype_bytes("float16")

        # Try fully static evaluation first.
        int_indices: List[int] = []
        all_int = True
        for idx in indices:
            try:
                int_indices.append(int(idx))
            except (TypeError, ValueError):
                all_int = False
                break

        if all_int:
            if buf is not None:
                try:
                    shape = [int(d) for d in buf.shape]
                except (TypeError, ValueError):
                    shape = []
                if len(shape) == len(int_indices):
                    strides = self._dense_row_major_strides(shape)
                    elem_offset = sum(i * s for i, s in zip(int_indices, strides))
                else:
                    # Fallback to first-index behavior when rank metadata does not match.
                    elem_offset = int_indices[0]
            else:
                elem_offset = int_indices[0]
            return elem_offset * elem_bytes

        # Symbolic path: build a C expression for flattened element offset.
        if buf is not None:
            try:
                shape = [int(d) for d in buf.shape]
            except (TypeError, ValueError):
                shape = []
        else:
            shape = []

        idx_exprs = [_STRINGIFIER.stringify(idx) for idx in indices]
        if len(shape) == len(idx_exprs) and len(shape) > 1:
            strides = self._dense_row_major_strides(shape)
            terms = []
            for idx_expr, stride in zip(idx_exprs, strides):
                if stride == 1:
                    terms.append(f"({idx_expr})")
                else:
                    terms.append(f"(({idx_expr}) * {stride})")
            elem_expr = " + ".join(terms)
            if len(terms) > 1:
                elem_expr = f"({elem_expr})"
        else:
            # Fallback when rank metadata does not match available indices.
            elem_expr = f"({idx_exprs[0]})"

        if elem_bytes == 1:
            return elem_expr
        return f"(({elem_expr}) * {elem_bytes})"
