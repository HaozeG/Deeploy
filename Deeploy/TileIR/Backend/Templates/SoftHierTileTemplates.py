# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""SoftHier-specific NodeTemplates for TileIR operations.

Each template corresponds to one TVM TIR construct encountered during
TileLang AST lifting.  Templates follow the same Mako-based style as
``Deeploy/Targets/SoftHier/Templates/GemmTemplate.py``.

**Cluster-ID guard**
--------------------
Cluster guarding is applied in the TileIR midend by
``ClusterGuardTransformationPass``. Templates in this module contain only the
operation body.

The operatorRepresentation key ``cluster_id`` is still carried with each
operation and consumed by the transformation pass (``None`` means no guard,
i.e. all clusters execute the operation).

**Memory notation**
-------------------
* ``src_hbm`` / ``dst_hbm`` — full 64-bit HBM address
* ``dst_l1`` / ``src_l1`` — L1 pointer converted to ``uint64_t``.
"""

from Deeploy.DeeployTypes import NodeTemplate

# ---------------------------------------------------------------------------
# TileBlockPreamble — hoisted cluster-map guard variables for a tile iteration.
#
# Emitted ONCE at the start of the innermost kernel block body, before any
# TileAlloc / operation.  Declares ``_bid`` (block-index modulo group size) and
# ``_cluster_active`` (bitmask test result for this cluster) as loop-body-scoped
# locals so every subsequent guard can reuse them without recomputation.
#
# OperatorRepresentation keys:
#   cluster_map    : List[int] — physical cluster IDs, one per block
#   block_id_expr  : str       — C expression for the flat block index
# ---------------------------------------------------------------------------

TileBlockPreambleTemplateStr = r"""<%
_cm = context.get('cluster_map', None)
_bie = context.get('block_id_expr', None)
if _cm is not None and _bie is not None:
    _n = len(_cm)
    _ternary = " : ".join("(_bid == %d) ? (1U << %d)" % (i, c) for i, c in enumerate(_cm))
    _ternary += " : 0U"
%>
% if _cm is not None and _bie is not None:
// cluster-map preamble: compute once per tile iteration
uint32_t _bid = (${_bie}) % ${_n};
uint32_t _cluster_active = (1U << flex_get_cluster_id()) & (${_ternary});
if (!_cluster_active) continue;
% endif
"""

TileBlockPreambleTemplate = NodeTemplate(TileBlockPreambleTemplateStr)

# ---------------------------------------------------------------------------
# TileGroupPreamble — cluster_active guard for multi-cluster groups (TP-style).
#
# Uses cluster_active_${group_id} declared in TileGroupContextTemplate:
#   cluster_active = valid_grid && (this_grid_id < num_active_instances)
# This mirrors SummaGEMM's "this_grid_id < summa_groups" check and correctly
# restricts execution to the intended group instances while letting unused
# instances skip the tile body (and the group barriers inside it).
#
# When num_active_instances > 1 and block_id_expr is set, a per-tile dispatch
# check is appended so each group instance handles a disjoint tile subset:
#   this_grid_id == (tile_block_id % num_active_instances)
#
# OperatorRepresentation keys:
#   group_id             : str
#   block_id_expr        : str or None  — C expression for the flat tile index
#   num_active_instances : int or None  — total active group instances
# ---------------------------------------------------------------------------

TileGroupPreambleTemplateStr = r"""<%
_bie  = context.get('block_id_expr', None)
_n    = context.get('num_active_instances', None)
_tbl  = context.get('tgid_table', None)
_dispatch = _bie is not None and _n is not None and _n > 1
%>
// group-preamble: skip tile if not in an active group instance ('${group_id}')
% if _dispatch and _tbl is not None:
<%
_tbl_str = ", ".join(str(t) for t in _tbl)
%>
static const uint32_t _tgid_table_${group_id}[${_n}] = {${_tbl_str}};
if (!cluster_active_${group_id} || (group_info_${group_id}.this_grid_id != _tgid_table_${group_id}[(${_bie}) % ${_n}]))
% elif _dispatch:
if (!cluster_active_${group_id} || (group_info_${group_id}.this_grid_id != ((${_bie}) % ${_n})))
% else:
if (!cluster_active_${group_id})
% endif
  continue;
"""

TileGroupPreambleTemplate = NodeTemplate(TileGroupPreambleTemplateStr)

# ---------------------------------------------------------------------------
# TileLoad — HBM → L1 DMA transfer (2D strided)
#
# OperatorRepresentation keys:
#   src               : str  — name of the HBM-side TileBuffer (PrimFunc param)
#   dst               : str  — name of the L1-side TileBuffer  (alloc_fragment)
#   nbytes            : int  — total transfer size in bytes
#   row_bytes         : int  — bytes per row of the tile
#   num_rows          : int  — number of rows in the tile
#   src_stride_bytes  : int  — byte stride between rows in the HBM source
#   src_offset        : int  — byte offset into src (0 if no offset)
#   cluster_id        : Optional[int]
# ---------------------------------------------------------------------------

TileLoadTemplateStr = r"""
// TileLoad: HBM -> L1  (${src} -> ${dst}, ${num_rows} x ${row_bytes} bytes)
if (flex_is_dm_core()) {
    uint64_t _src_hbm = (uint64_t)(uintptr_t)((char*)${src} + ${src_offset});
    uint64_t _dst_l1  = (uint64_t)(uintptr_t)${dst};
    flex_dma_sync_2d(_dst_l1, _src_hbm, ${row_bytes}, ${row_bytes}, ${src_stride_bytes}, ${num_rows});
}
"""

TileLoadTemplate = NodeTemplate(TileLoadTemplateStr)

# ---------------------------------------------------------------------------
# TileStore — L1 → HBM DMA transfer (2D strided)
#
# OperatorRepresentation keys:
#   src               : str  — name of L1 buffer
#   dst               : str  — name of HBM buffer
#   dst_offset        : int  — byte offset into dst
#   nbytes            : int  — total transfer size in bytes
#   row_bytes         : int  — bytes per row of the tile
#   num_rows          : int  — number of rows in the tile
#   dst_stride_bytes  : int  — byte stride between rows in the HBM destination
#   cluster_id        : Optional[int]
# ---------------------------------------------------------------------------

TileStoreTemplateStr = r"""
// TileStore: L1 -> HBM  (${src} -> ${dst}, ${num_rows} x ${row_bytes} bytes)
flex_intra_cluster_sync();
if (flex_is_dm_core()) {
    uint64_t _dst_hbm = (uint64_t)(uintptr_t)((char*)${dst} + ${dst_offset});
    uint64_t _src_l1  = (uint64_t)(uintptr_t)${src};
    flex_dma_sync_2d(_dst_hbm, _src_l1, ${row_bytes}, ${dst_stride_bytes}, ${row_bytes}, ${num_rows});
}
"""

TileStoreTemplate = NodeTemplate(TileStoreTemplateStr)

# ---------------------------------------------------------------------------
# TileCopy — L1 → L1 memcpy (same memory space, always contiguous)
#
# OperatorRepresentation keys:
#   src, dst, nbytes, cluster_id
# ---------------------------------------------------------------------------

TileCopyTemplateStr = r"""
// TileCopy: L1 -> L1  (${src} -> ${dst}, ${nbytes} bytes)
if (flex_is_dm_core()) {
    uint64_t _dst_l1 = (uint64_t)(uintptr_t)${dst};
    uint64_t _src_l1 = (uint64_t)(uintptr_t)${src};
    flex_dma_sync_2d(_dst_l1, _src_l1, ${nbytes}, ${nbytes}, ${nbytes}, 1);
}
"""

TileCopyTemplate = NodeTemplate(TileCopyTemplateStr)

# ---------------------------------------------------------------------------
# TileFill — fill an L1 buffer with a scalar value (maps from T.clear)
#
# OperatorRepresentation keys:
#   buf, nbytes, val (numeric literal), cluster_id
# ---------------------------------------------------------------------------

TileFillTemplateStr = r"""
// TileFill: fill ${buf} with ${val}  (${nbytes} bytes) (now only supports zero-fill)
if (flex_is_dm_core()) {
    flex_dma_sync_2d((uint64_t)(uintptr_t)${buf}, zomem(0), ${nbytes}, ${nbytes}, 0, 1);
}
"""

TileFillTemplate = NodeTemplate(TileFillTemplateStr)

# ---------------------------------------------------------------------------
# TileReduce — reduction over an L1 tile into an L1 accumulator
#
# OperatorRepresentation keys:
#   src        : str  — input L1 buffer name
#   dst        : str  — output L1 accumulator name
#   op         : str  — reduction op string, e.g. "sum"
#   extent     : int  — number of elements to reduce (innermost dimension)
#   outer      : int  — number of outer elements (parallelised over cores)
#   dtype      : str  — C element type, e.g. "float32_t" / "fp16"
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

# TODO: use reduce related intrinsics
TileReduceTemplateStr = r"""
// TileReduce: ${op} over ${src} -> ${dst}  (${outer} x ${extent} elements)
{
    uint32_t core_id = flex_get_core_id();
    for (uint32_t _oi = core_id; _oi < ${outer}; _oi += ARCH_NUM_CORE_PER_CLUSTER) {
        ${dtype} _acc = ((${dtype}*)${dst})[_oi];
        for (uint32_t _ii = 0; _ii < ${extent}; _ii++) {
            % if op == "sum":
            _acc += ((${dtype}*)${src})[_oi * ${extent} + _ii];
            % elif op == "max":
            { ${dtype} _v = ((${dtype}*)${src})[_oi * ${extent} + _ii]; if (_v > _acc) _acc = _v; }
            % elif op == "min":
            { ${dtype} _v = ((${dtype}*)${src})[_oi * ${extent} + _ii]; if (_v < _acc) _acc = _v; }
            % endif
        }
        ((${dtype}*)${dst})[_oi] = _acc;
    }
}
"""

TileReduceTemplate = NodeTemplate(TileReduceTemplateStr)

# ---------------------------------------------------------------------------
# TileGemm — matrix multiply using SoftHier RedMule.
#
# OperatorRepresentation keys:
#   A, B, C     : str  — L1 buffer names (x, w, y in RedMule API)
#   M, N, K     : int  — GEMM tile dimensions
#   cluster_id  : Optional[int]
# ---------------------------------------------------------------------------

TileGemmTemplateStr = r"""
// TileGemm: RedMule GEMM  C = A x B (+ C) with M=${M}, N=${N}, K=${K}
if (flex_is_first_core()) {
    flex_redmule_config((uint16_t)${M}, (uint16_t)${K}, (uint16_t)${N});
    flex_redmule_trigger(
        (uint32_t)(uintptr_t)${A},
        (uint32_t)(uintptr_t)${B},
        (uint32_t)(uintptr_t)${C},
        ${redmule_dtype}
    );
    flex_redmule_wait();
}
"""

TileGemmTemplate = NodeTemplate(TileGemmTemplateStr)

# ---------------------------------------------------------------------------
# TileGemmTransposeB — transpose B in-place via flex_transpose_engine, then GEMM.
#
# OperatorRepresentation keys (same as TileGemm, plus):
#   elem_size   : int  — element byte width (1=i8/u8, 2=fp16/i16/u16, 4=fp32)
#
# B is stored as [N×K]; after transpose it becomes [K×N] usable by RedMule.
# ---------------------------------------------------------------------------

TileGemmTransposeBTemplateStr = r"""
// TileGemm: RedMule GEMM  C = A x B^T  with M=${M}, N=${N}, K=${K}
if (flex_is_dm_core()) {
    flex_transpose_engine_config(${N}, ${K}, (uint32_t)(uintptr_t)${B}, (uint32_t)(uintptr_t)${B}, ${elem_size});
    flex_transpose_engine_trigger();
    flex_transpose_engine_wait();
}
flex_intra_cluster_sync();
if (flex_is_first_core()) {
    flex_redmule_config((uint16_t)${M}, (uint16_t)${K}, (uint16_t)${N});
    flex_redmule_trigger(
        (uint32_t)(uintptr_t)${A},
        (uint32_t)(uintptr_t)${B},
        (uint32_t)(uintptr_t)${C},
        ${redmule_dtype}
    );
    flex_redmule_wait();
}
"""

TileGemmTransposeBTemplate = NodeTemplate(TileGemmTransposeBTemplateStr)

# ---------------------------------------------------------------------------
# TileGemmTransposeA — transpose A in-place via flex_transpose_engine, then GEMM.
#
# A is stored as [K×M]; after transpose it becomes [M×K] usable by RedMule.
# ---------------------------------------------------------------------------

TileGemmTransposeATemplateStr = r"""
// TileGemm: RedMule GEMM  C = A^T x B  with M=${M}, N=${N}, K=${K}
if (flex_is_dm_core()) {
    flex_transpose_engine_config(${K}, ${M}, (uint32_t)(uintptr_t)${A}, (uint32_t)(uintptr_t)${A}, ${elem_size});
    flex_transpose_engine_trigger();
    flex_transpose_engine_wait();
}
flex_intra_cluster_sync();
if (flex_is_first_core()) {
    flex_redmule_config((uint16_t)${M}, (uint16_t)${K}, (uint16_t)${N});
    flex_redmule_trigger(
        (uint32_t)(uintptr_t)${A},
        (uint32_t)(uintptr_t)${B},
        (uint32_t)(uintptr_t)${C},
        ${redmule_dtype}
    );
    flex_redmule_wait();
}
"""

TileGemmTransposeATemplate = NodeTemplate(TileGemmTransposeATemplateStr)

# ---------------------------------------------------------------------------
# TileGemmTransposeAB — transpose both A and B in-place, then GEMM.
# ---------------------------------------------------------------------------

TileGemmTransposeABTemplateStr = r"""
// TileGemm: RedMule GEMM  C = A^T x B^T  with M=${M}, N=${N}, K=${K}
if (flex_is_dm_core()) {
    flex_transpose_engine_config(${K}, ${M}, (uint32_t)(uintptr_t)${A}, (uint32_t)(uintptr_t)${A}, ${elem_size});
    flex_transpose_engine_trigger();
    flex_transpose_engine_wait();
    flex_transpose_engine_config(${N}, ${K}, (uint32_t)(uintptr_t)${B}, (uint32_t)(uintptr_t)${B}, ${elem_size});
    flex_transpose_engine_trigger();
    flex_transpose_engine_wait();
}
flex_intra_cluster_sync();
if (flex_is_first_core()) {
    flex_redmule_config((uint16_t)${M}, (uint16_t)${K}, (uint16_t)${N});
    flex_redmule_trigger(
        (uint32_t)(uintptr_t)${A},
        (uint32_t)(uintptr_t)${B},
        (uint32_t)(uintptr_t)${C},
        ${redmule_dtype}
    );
    flex_redmule_wait();
}
"""

TileGemmTransposeABTemplate = NodeTemplate(TileGemmTransposeABTemplateStr)

# ---------------------------------------------------------------------------
# TileEltwise — element-wise operation, typically from a BufferStore in a
# parallel For loop.  Parallelised across cores.
#
# OperatorRepresentation keys:
#   dst        : str  — output L1 buffer name
#   src_expr   : str  — fully inlined C expression for the rhs value
#   loop_var   : str  — loop induction variable name
#   extent     : int  — loop bound
#   dtype      : str  — C element type
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

# TODO: consider vectorized version
TileEltwiseTemplateStr = r"""
// TileEltwise: ${dst}[i] = ${src_expr}  (${extent} elements, parallel)
{
    uint32_t core_id = flex_get_core_id();
    for (uint32_t ${loop_var} = core_id; ${loop_var} < ${extent}; ${loop_var} += ARCH_NUM_CORE_PER_CLUSTER) {
        ((${dtype}*)${dst})[${index_expr}] = (${dtype})(${src_expr});
    }
}
"""

TileEltwiseTemplate = NodeTemplate(TileEltwiseTemplateStr)

# ---------------------------------------------------------------------------
# Parallel-core open/close — bracket an inner eltwise with an outer loop
# that distributes its variable across cores (core_id stride).
#
# Usage: TileParallelCoreOpenTemplate ... TileInnerEltwiseTemplate ...
#        TileParallelCoreCloseTemplate
#
# OperatorRepresentation keys:
#   loop_var : str   — outer induction variable
#   extent   : int   — outer loop bound
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

TileParallelCoreOpenTemplateStr = r"""
// ParallelCore: distribute ${loop_var} across cores
{
    uint32_t core_id = flex_get_core_id();
    for (uint32_t ${loop_var} = core_id; ${loop_var} < ${extent}; ${loop_var} += ARCH_SPATZ_ATTACED_CORES) {
"""

TileParallelCoreOpenTemplate = NodeTemplate(TileParallelCoreOpenTemplateStr)

TileParallelCoreCloseTemplateStr = r"""
    } // end parallel_core ${loop_var}
}
"""

TileParallelCoreCloseTemplate = NodeTemplate(TileParallelCoreCloseTemplateStr)

# ---------------------------------------------------------------------------
# InnerEltwise — lives inside a TileParallelCoreOpen block.
# No core_id distribution; the outer parallel-core loop already partitions
# the work.  Vectorizable by SpatzVectorizationPass → SpatzInnerEltwiseTemplate.
#
# OperatorRepresentation keys (same as TileEltwiseTemplate plus):
#   outer_loop_var : str  — outer parallel induction variable (for Spatz pass)
#   dst_stride     : int  — product of buffer dims after dim-0 (row stride in elems)
# ---------------------------------------------------------------------------

TileInnerEltwiseTemplateStr = r"""
// InnerEltwise: ${dst}[i] = ${src_expr}  (${extent} elements, inner serial)
for (uint32_t ${loop_var} = 0; ${loop_var} < ${extent}; ${loop_var}++) {
    ((${dtype}*)${dst})[${index_expr}] = (${dtype})(${src_expr});
}
"""

TileInnerEltwiseTemplate = NodeTemplate(TileInnerEltwiseTemplateStr)


# SpatzContext: emitted once at function scope (prepended by SpatzVectorizationPass).
# Declares _spatz_attached and _spatz_sid so every SpatzEltwiseTemplate in the
# same function can skip the repeated volatile array reads.
# The volatile array trick prevents GCC from auto-vectorizing the init (which
# would clobber v8/v16 before inline asm uses them).
SpatzContextTemplateStr = r"""
// SpatzContext: read Spatz config once for this core (reused by all SpatzEltwise ops)
volatile uint32_t _spatz_check_ctx[ARCH_NUM_CORE_PER_CLUSTER];
do {
    const uint32_t _tmp[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_CHECK_LIST;
    for (uint32_t _i = 0; _i < ARCH_NUM_CORE_PER_CLUSTER; _i++)
        _spatz_check_ctx[_i] = _tmp[_i];
} while (0);
uint32_t _spatz_attached = _spatz_check_ctx[flex_get_core_id()];
volatile uint32_t _spatz_sids_ctx[ARCH_NUM_CORE_PER_CLUSTER];
do {
    const uint32_t _tmp[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_SID_LIST;
    for (uint32_t _i = 0; _i < ARCH_NUM_CORE_PER_CLUSTER; _i++)
        _spatz_sids_ctx[_i] = _tmp[_i];
} while (0);
uint32_t _spatz_sid = _spatz_attached ? _spatz_sids_ctx[flex_get_core_id()] : 0;
"""

SpatzContextTemplate = NodeTemplate(SpatzContextTemplateStr)

SpatzEltwiseTemplateStr = r"""
// SpatzEltwise: vectorized ${spatz_op} on ${dst} (${extent} elements, ${num_spatz} Spatz cores)
// _spatz_attached and _spatz_sid are declared once by SpatzContextTemplate at function scope.
{
    if (_spatz_attached) {
        uint32_t _vlen = ${extent} / ARCH_SPATZ_ATTACED_CORES;
        uint32_t _addr = (uint32_t)(uintptr_t)${dst} + _spatz_sid * _vlen * sizeof(${dtype});
        ${spatz_setup}
        uint32_t _avl;
        while (_vlen > 0) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(_avl) : "r"(_vlen));
            ${spatz_body}
            _vlen -= _avl;
            _addr += _avl * sizeof(${dtype});
        }
    }
}
"""

SpatzEltwiseTemplate = NodeTemplate(SpatzEltwiseTemplateStr)

# ---------------------------------------------------------------------------
# SpatzInnerEltwise — vectorized inner eltwise nested inside a
# TileParallelCoreOpen block.  _addr and every _addr_src produced by
# SpatzVectorizationPass are shifted by outer_loop_var * dst_stride to
# select the correct row.  The else-branch provides a scalar fallback for
# non-Spatz cores (which still have rows assigned to them by the outer loop).
#
# OperatorRepresentation keys (same as SpatzEltwiseTemplate plus):
#   outer_loop_var : str  — outer induction variable (in scope from caller)
#   dst_stride     : int  — row stride of dst in elements
#   loop_var       : str  — inner induction variable (for fallback loop)
#   index_expr     : str  — full linearized index for fallback (uses both vars)
#   fallback_expr  : str  — scalar rhs expression for fallback
# ---------------------------------------------------------------------------

SpatzInnerEltwiseTemplateStr = r"""
// SpatzInnerEltwise: vectorized ${spatz_op} on ${dst}[${outer_loop_var},:]
// (${extent} elements, ${num_spatz} Spatz cores; row offset = ${outer_loop_var}*${dst_stride})
// _spatz_attached and _spatz_sid are declared once by SpatzContextTemplate.
{
    if (_spatz_attached) {
        uint32_t _vlen = ${extent};
        uint32_t _addr = (uint32_t)(uintptr_t)${dst}
                         + (${outer_loop_var} * ${dst_stride}) * sizeof(${dtype})
                         + _spatz_sid * _vlen * sizeof(${dtype});
        ${spatz_setup}
        uint32_t _avl;
        while (_vlen > 0) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(_avl) : "r"(_vlen));
            ${spatz_body}
            _vlen -= _avl;
            _addr += _avl * sizeof(${dtype});
        }
    }
}
"""

SpatzInnerEltwiseTemplate = NodeTemplate(SpatzInnerEltwiseTemplateStr)


# ---------------------------------------------------------------------------
# ForLoop open / close — serial for-loop brace wrappers.
# addLeft(ForLoopOpenTemplate, rep) and addRight(ForLoopCloseTemplate, rep)
# bracket inner code snippets.
#
# OperatorRepresentation keys:
#   loop_var : str
#   min_val  : int or str expression
#   extent   : int or str expression
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

ForLoopOpenTemplateStr = r"""
for (int ${loop_var} = ${min_val}; ${loop_var} < ${extent}; ${loop_var}++) {
"""

ForLoopOpenStridedTemplateStr = r"""
for (int ${loop_var} = ${min_val}; ${loop_var} < ${extent}; ${loop_var} += ${step}) {
"""

ForLoopCloseTemplateStr = r"""
} // end for ${loop_var}
"""

ForLoopOpenTemplate        = NodeTemplate(ForLoopOpenTemplateStr)
ForLoopOpenStridedTemplate = NodeTemplate(ForLoopOpenStridedTemplateStr)
ForLoopCloseTemplate       = NodeTemplate(ForLoopCloseTemplateStr)

# ---------------------------------------------------------------------------
# If / Else — C conditional block wrappers.
#
# Emitted by TilelangVisitor._visit_IfThenElse when TIR contains an
# IfThenElse node (e.g. from ``if group_id_x == 0:`` in a @tilelang.jit
# kernel).  ``condition`` is a C expression string produced by
# _ExprStringifier with group-variable substitutions applied.
#
# OperatorRepresentation keys:
#   condition  : str             — C boolean expression
#   cluster_id : None            — no cluster guard on structural braces
# ---------------------------------------------------------------------------

IfOpenTemplate  = NodeTemplate("if (${condition}) {\n")
IfCloseTemplate = NodeTemplate("}\n")
ElseOpenTemplate = NodeTemplate("} else {\n")

# ---------------------------------------------------------------------------
# TileSync — software barrier between DM-core-driven DMA and compute cores.
#
# OperatorRepresentation keys:
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

TileSyncTemplateStr = r"""
flex_intra_cluster_sync();
"""

TileSyncTemplate = NodeTemplate(TileSyncTemplateStr)

# ---------------------------------------------------------------------------
# TileMathPreamble — fp16 math helper macros emitted once when math intrinsics
# (exp, sigmoid, sqrt, rsqrt, abs, max, relu) are used in eltwise expressions.
# ---------------------------------------------------------------------------

TileMathPreambleTemplateStr = r"""
// TileLang math helpers (fp16)
#define tile_fp16_abs(x) ((fp16)((x) & 0x7FFF))
#define tile_fp16_relu(x) ({ fp16 _x=(x); (fp16_to_float(_x) > 0.0f) ? _x : ((fp16)0); })
#define tile_fp16_max(a,b) ({ fp16 _a=(a); fp16 _b=(b); (fp16_to_float(_a) > fp16_to_float(_b)) ? _a : _b; })
#define tile_fp16_min(a,b) ({ fp16 _a=(a); fp16 _b=(b); (fp16_to_float(_a) < fp16_to_float(_b)) ? _a : _b; })
#define tile_fp16_exp(x) ({ fp16 _in=(x); fp16 _out; asm_fp16_exp(&_in, &_out); _out; })
#define tile_fp16_sigmoid(x) ({ fp16 _in=(x); fp16 _out; asm_fp16_sigmoid(&_in, &_out); _out; })
#define tile_fp16_sqrt(x) ({ float _f=fp16_to_float(x); float _r; __asm__ __volatile__("fsqrt.s %0, %1, rne" : "=f"(_r) : "f"(_f)); float_to_fp16(_r); })
#define tile_fp16_rsqrt(x) ({ float _f=fp16_to_float(x); float _r; __asm__ __volatile__("fsqrt.s %0, %1, rne" : "=f"(_r) : "f"(_f)); float_to_fp16(1.0f / _r); })
"""

TileMathPreambleTemplate = NodeTemplate(TileMathPreambleTemplateStr)

# ---------------------------------------------------------------------------
# TileAssert — runtime assertion from T.device_assert(cond).
# ---------------------------------------------------------------------------

TileAssertTemplateStr = r"""
if (!(${condition})) {
    printf("[TileLang] Assertion failed: ${message}\n");
}
"""

TileAssertTemplate = NodeTemplate(TileAssertTemplateStr)

# ---------------------------------------------------------------------------
# TileGlobalBarrier — explicit mid-kernel global barrier from T.sync_grid().
# ---------------------------------------------------------------------------

TileGlobalBarrierTemplateStr = r"""
flex_global_barrier_xy();
"""

TileGlobalBarrierTemplate = NodeTemplate(TileGlobalBarrierTemplateStr)

# ---------------------------------------------------------------------------
# TileIntraClusterReduce — reduce across Spatz vector cores within a cluster
# using shared L1 + cluster barrier (no DMA).  SoftHier analogue of GPU
# warp_reduce_sum / warp_reduce_max.
#
# OperatorRepresentation keys:
#   buf      : str  — L1 buffer name (in/out)
#   op       : str  — "sum" | "max" | "min"
#   nbytes   : int  — buffer size in bytes
#   dtype    : str  — C element type, e.g. "fp16"
# ---------------------------------------------------------------------------

TileIntraClusterReduceTemplateStr = r"""
// TileIntraClusterReduce: ${op} across cores on ${buf} (${nbytes} bytes)
{
    uint32_t _core_id = flex_get_core_id();
    uint32_t _n_elems = ${nbytes} / sizeof(${dtype});
    if (_core_id == 0) {
        for (uint32_t _c = 1; _c < ARCH_NUM_CORE_PER_CLUSTER; _c++) {
            ${dtype} *_src = ((${dtype}*)${buf}) + _c * _n_elems;
            ${dtype} *_dst = (${dtype}*)${buf};
            for (uint32_t _i = 0; _i < _n_elems; _i++) {
                % if op == "sum":
                _dst[_i] += _src[_i];
                % elif op == "max":
                if (_src[_i] > _dst[_i]) _dst[_i] = _src[_i];
                % elif op == "min":
                if (_src[_i] < _dst[_i]) _dst[_i] = _src[_i];
                % endif
            }
        }
    }
}
flex_intra_cluster_sync();
"""

TileIntraClusterReduceTemplate = NodeTemplate(TileIntraClusterReduceTemplateStr)

# ---------------------------------------------------------------------------
# TileAlloc — L1 scratch buffer allocation (generated by the visitor for each
# alloc_fragment / alloc_shared buffer in the tilelang_root Block).
#
# OperatorRepresentation keys:
#   name       : str — C variable name
#   dtype      : str — C element type, e.g. "fp16"
#   nbytes     : int — allocation size in bytes
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

TileAllocTemplateStr = r"""<%
_cm = context.get('cluster_map', None)
_bie = context.get('block_id_expr', None)
%>// TileAlloc: ${name}  (${nbytes} bytes in L1)
static volatile uintptr_t _addr_${name} = 0;
% if _cm is not None and _bie is not None:
if (flex_is_first_core()) {
    _addr_${name} = (uintptr_t)flex_l1_malloc(${nbytes});
}
% elif cluster_id is not None:
if (flex_get_cluster_id() == ${cluster_id} && flex_is_first_core()) {
    _addr_${name} = (uintptr_t)flex_l1_malloc(${nbytes});
}
% else:
if (flex_is_first_core()) {
    _addr_${name} = (uintptr_t)flex_l1_malloc(${nbytes});
}
% endif
flex_intra_cluster_sync();
${dtype}* ${name} = (${dtype}*)(uintptr_t)_addr_${name};
"""

TileAllocTemplate = NodeTemplate(TileAllocTemplateStr)

# ---------------------------------------------------------------------------
# TileFree — L1 scratch buffer deallocation (generated in reverse order).
#
# OperatorRepresentation keys:
#   name       : str
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

TileFreeTemplateStr = r"""<%
_cm = context.get('cluster_map', None)
_bie = context.get('block_id_expr', None)
%>// TileFree: ${name}
% if _cm is not None and _bie is not None:
if (flex_is_first_core()) {
    flex_l1_free((void*)(uintptr_t)${name});
}
% elif cluster_id is not None:
if (flex_get_cluster_id() == ${cluster_id} && flex_is_first_core()) {
    flex_l1_free((void*)(uintptr_t)${name});
}
% else:
if (flex_is_first_core()) {
    flex_l1_free((void*)(uintptr_t)${name});
}
% endif
flex_intra_cluster_sync();
"""

TileFreeTemplate = NodeTemplate(TileFreeTemplateStr)
