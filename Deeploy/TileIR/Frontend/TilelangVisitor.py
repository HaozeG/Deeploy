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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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
        Default cluster ID used as the guard for all generated operations.
        ``None`` means no cluster-ID guard (all clusters execute).
    """

    def __init__(self, cluster_id: Optional[int] = None):
        if not _TVM_AVAILABLE:
            raise ImportError(
                "TVM is required for TilelangVisitor."
            )
        self.cluster_id: Optional[int] = cluster_id

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

        # supported op and intrinsic patterns (extend as needed)
        self._supported_ops = {
            "copy": self._handle_copy,
            "reduce": self._handle_reduce,
            "fill": self._handle_fill,
            "gemm_py": self._handle_gemm_py,
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

        # 1. Register PrimFunc parameters as HBM buffers
        self._register_params(primfunc)

        # 2. Extract cluster_id from function attrs if present
        func_cluster_id = self.cluster_id
        if hasattr(primfunc, "attrs") and primfunc.attrs is not None:
            try:
                func_cluster_id = int(primfunc.attrs["cluster_id"])
            except (KeyError, TypeError):
                pass

        # 3. Walk the body
        self._visit_stmt(primfunc.body, cluster_id=func_cluster_id)

        return self._bindings

    def _emit_binding(self, op_kind: str, template: NodeTemplate, rep: Dict, code_transformer: Optional[CodeTransformation] = None) -> None:
        """Record one TileBinding operation in visit order."""
        if self._bindings is None:
            raise RuntimeError("TileBinding pipeline is not initialized. Call visit_bindings() first.")
        self._bindings.add(TileBinding(op_kind=op_kind, template=template, operator_representation=rep, code_transformer=code_transformer))

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
                        cluster_id   = self.cluster_id,
                    )
                    # _type/_instance are required by NetworkContext.lookup;
                    # we leave them unset here — code generation accesses
                    # the buffer by name directly via OperatorRepresentation.
                    self._ctxt.globalObjects[buf.name] = db
                except Exception:
                    pass  # context registration is best-effort

    def _register_local_buf(self, buf: "tir.Buffer", cluster_id):
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

    def _visit_Block(self, stmt, cluster_id):
        # TODO: consider adding sync at block boundaries if needed (e.g. after a producer block before a consumer block)
        # Allocate local buffers declared in this block
        for buf in getattr(stmt, "alloc_buffers", []):
            self._register_local_buf(buf, cluster_id)
            nbytes = _prod(buf.shape) * _dtype_bytes(str(buf.dtype))
            rep = {
                "name":       buf.name,
                "dtype":      _c_dtype(str(buf.dtype)),
                "nbytes":     nbytes,
                "cluster_id": cluster_id,
            }
            self._emit_binding("alloc", TileAllocTemplate, rep)

        self._visit_stmt(stmt.body, cluster_id)

        # TODO: consider deallocating local buffers at the end of their block scope, align with cluster id
        # Emit frees in reverse alloc order
        for name in reversed(self._alloc_order):
            rep = {"name": name, "cluster_id": cluster_id}
            self._emit_binding("free", TileFreeTemplate, rep)
        # Only do this once per block
        self._alloc_order = []

    def _visit_AttrStmt(self, stmt, cluster_id):
        """Handle TVM AttrStmt nodes.

        Two cases are distinguished:

        * ``node == "anno"`` and ``attr_key == "cluster_id"`` — extracts the
          integer cluster id from ``T.attr("anno", "cluster_id", T.int32(N))``
          and propagates it to the subtree.
        * All other AttrStmt nodes (e.g. ``thread_extent``) — recurse into
          the body unchanged.
        """
        # Extract cluster_id from 'anno' node if present
        if hasattr(stmt, "node") and stmt.node == "anno" and hasattr(stmt, "attr_key") and stmt.attr_key == "cluster_id":
            try:
                cluster_id = int(stmt.value)
            except (KeyError, TypeError):
                pass
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
                {"cluster_id": cluster_id},
            )

    def _visit_For(self, stmt, cluster_id):
        """Serial or parallel For loop."""
        loop_var = str(stmt.loop_var.name)
        min_val  = int(stmt.min) if hasattr(stmt.min, "__int__") else str(stmt.min)
        extent   = int(stmt.min) + int(stmt.extent) if hasattr(stmt.extent, "__int__") else str(stmt.extent)

        # ForKind: 0=Serial, 1=Parallel, 2=Unroll, 3=Vectorized, 4=ThreadBinding
        for_kind = int(stmt.kind)

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

        # check for cluster_id annotation
        if src_is_hbm and not dst_is_hbm:
            # HBM → L1: TileLoad
            rep = {
                "src":        src_buf_name,
                "dst":        dst_buf_name,
                "nbytes":     nbytes,
                "src_offset": src_offset,
                "cluster_id": cluster_id,
            }
            self._emit_binding("load", TileLoadTemplate, rep)

        elif not src_is_hbm and dst_is_hbm:
            # L1 → HBM: TileStore
            rep = {
                "src":        src_buf_name,
                "dst":        dst_buf_name,
                "nbytes":     nbytes,
                "dst_offset": dst_offset,
                "cluster_id": cluster_id,
            }
            self._emit_binding("store", TileStoreTemplate, rep)

        else:
            # L1 → L1: TileCopy
            rep = {
                "src":        src_buf_name,
                "dst":        dst_buf_name,
                "nbytes":     nbytes,
                "cluster_id": cluster_id,
            }
            self._emit_binding("copy", TileCopyTemplate, rep)

        # Insert sync after every data movement
        self._emit_binding("sync", TileSyncTemplate, {"cluster_id": cluster_id})

    def _handle_reduce(self, args: list, cluster_id):
        """T.reduce(src_slice, dst_slice, op_str, dim, clear_flag) → TileReduce."""
        if len(args) < 3:
            return
        src_region = args[0]
        dst_region = args[1]
        op_str     = str(args[2]).strip('"')  # "sum" / "max" / "min"

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
            "cluster_id": cluster_id,
        }
        self._emit_binding("reduce", TileReduceTemplate, rep)
        self._emit_binding("sync", TileSyncTemplate, {"cluster_id": cluster_id})

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

        rep = {
            "buf":        buf_name,
            "val":        val,
            "nbytes":     nbytes,
            "cluster_id": cluster_id,
        }
        self._emit_binding("fill", TileFillTemplate, rep)
        self._emit_binding("sync", TileSyncTemplate, {"cluster_id": cluster_id})

    def _handle_gemm_py(self, args: list, cluster_id):
        """T.gemm(A_region, B_region, C_region, M, N, K) → TileGemm."""
        if len(args) < 6:
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

        redmule_dtype_mapping = {
            'uint16': 'REDMULE_UINT_16',
            'int16':  'REDMULE_INT_16',
            'float16': 'REDMULE_FP_16',
            'uint8':  'REDMULE_UINT_8',
            'int8':   'REDMULE_INT_8',
        }
        redmule_dtype_str = redmule_dtype_mapping.get(str(self._local_bufs.get(A_buf_name).dtype), "REDMULE_FP_16")

        rep = {
            "A":          A_buf_name,
            "B":          B_buf_name,
            "C":          C_buf_name,
            "M":          M,
            "N":          N,
            "K":          K,
            "cluster_id": cluster_id,
            "redmule_dtype": redmule_dtype_str,
        }
        self._emit_binding("gemm", TileGemmTemplate, rep)
        # TODO: consider adding a pass to fuse unnecessary syncs
        self._emit_binding("sync", TileSyncTemplate, {"cluster_id": cluster_id})

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
            rep = {
                "dst":        dst_name,
                "src_expr":   src_expr,
                "loop_var":   loop_var,
                "extent":     extent,
                "dtype":      dtype,
                "cluster_id": cluster_id,
            }
            self._emit_binding("eltwise", TileEltwiseTemplate, rep)
            self._emit_binding("sync", TileSyncTemplate, {"cluster_id": cluster_id})
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
        # Use a constant 0 index for simplicity
        loop_var = "_i_scalar"
        rep = {
            "dst":        dst_name,
            "src_expr":   src_expr,
            "loop_var":   loop_var,
            "extent":     1,
            "dtype":      dtype,
            "cluster_id": cluster_id,
        }
        self._emit_binding("eltwise", TileEltwiseTemplate, rep)

    # ------------------------------------------------------------------
    # Region helpers
    # ------------------------------------------------------------------

    def _region_buf_name(self, region) -> str:
        """Extract the buffer name from a T.region(...) call."""
        if region is None:
            return "unknown"
        # T.region signature: T.region(buf_ptr, ndim, d0, d1, ...)
        # args[0] is a BufferLoad or pointer expression containing the name
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

    def _region_nbytes(self, region) -> int:
        """Compute transfer size in bytes from a T.region(...) call.

        T.region(ptr, ndim, d0, d1, ...) — args[2:] are the dimension sizes.
        """
        args = list(region.args) if hasattr(region, "args") else []
        if len(args) < 3:
            return 0
        # args[1] = ndim; args[2:2+ndim] = dimension extents
        ndim = int(args[1]) if hasattr(args[1], "__int__") else 0
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
        return 0
