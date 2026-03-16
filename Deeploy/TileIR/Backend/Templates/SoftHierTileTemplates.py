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
# TileLoad — HBM → L1 DMA transfer
#
# OperatorRepresentation keys:
#   src        : str  — name of the HBM-side TileBuffer (PrimFunc param)
#   dst        : str  — name of the L1-side TileBuffer  (alloc_fragment)
#   nbytes     : int  — transfer size in bytes
#   src_offset : int  — byte offset into src (0 if no offset)
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

TileLoadTemplateStr = r"""
// TileLoad: HBM -> L1  (${src} -> ${dst}, ${nbytes} bytes)
if (flex_is_dm_core()) {
    uint64_t _src_hbm = (uint64_t)(uintptr_t)((char*)${src} + ${src_offset});
    uint64_t _dst_l1  = (uint64_t)(uintptr_t)${dst};
    flex_dma_async_1d(_dst_l1, _src_hbm, ${nbytes});
    flex_dma_async_wait_all();
}
flex_intra_cluster_sync();
"""

TileLoadTemplate = NodeTemplate(TileLoadTemplateStr)

# ---------------------------------------------------------------------------
# TileStore — L1 → HBM DMA transfer
#
# OperatorRepresentation keys:
#   src        : str  — name of L1 buffer
#   dst        : str  — name of HBM buffer
#   dst_offset : int  — byte offset into dst
#   nbytes     : int
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

TileStoreTemplateStr = r"""
// TileStore: L1 -> HBM  (${src} -> ${dst}, ${nbytes} bytes)
if (flex_is_dm_core()) {
    uint64_t _dst_hbm = (uint64_t)(uintptr_t)((char*)${dst} + ${dst_offset});
    uint64_t _src_l1  = (uint64_t)(uintptr_t)${src};
    flex_dma_async_1d(_dst_hbm, _src_l1, ${nbytes});
    flex_dma_async_wait_all();
}
flex_intra_cluster_sync();
"""

TileStoreTemplate = NodeTemplate(TileStoreTemplateStr)

# ---------------------------------------------------------------------------
# TileCopy — L1 → L1 memcpy (same memory space)
#
# OperatorRepresentation keys:
#   src, dst, nbytes, cluster_id
# ---------------------------------------------------------------------------

TileCopyTemplateStr = r"""
// TileCopy: L1 -> L1  (${src} -> ${dst}, ${nbytes} bytes)
if (flex_is_dm_core()) {
    uint64_t _dst_l1 = (uint64_t)(uintptr_t)${dst};
    uint64_t _src_l1 = (uint64_t)(uintptr_t)${src};
    flex_dma_async_1d(_dst_l1, _src_l1, ${nbytes});
    flex_dma_async_wait_all();
}
flex_intra_cluster_sync();
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
    flex_dma_async_1d((uint64_t)(uintptr_t)${buf}, zomem(0), ${nbytes});
    flex_dma_async_wait_all();
}
flex_intra_cluster_sync();
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
flex_intra_cluster_sync();
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
    flex_redmule_config((uint16_t)${M}, (uint16_t)${N}, (uint16_t)${K});
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
        ((${dtype}*)${dst})[${loop_var}] = (${dtype})(${src_expr});
    }
}
flex_intra_cluster_sync();
"""

TileEltwiseTemplate = NodeTemplate(TileEltwiseTemplateStr)

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

ForLoopCloseTemplateStr = r"""
} // end for ${loop_var}
"""

ForLoopOpenTemplate  = NodeTemplate(ForLoopOpenTemplateStr)
ForLoopCloseTemplate = NodeTemplate(ForLoopCloseTemplateStr)

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
# TileAlloc — L1 scratch buffer allocation (generated by the visitor for each
# alloc_fragment / alloc_shared buffer in the tilelang_root Block).
#
# OperatorRepresentation keys:
#   name       : str — C variable name
#   dtype      : str — C element type, e.g. "fp16"
#   nbytes     : int — allocation size in bytes
#   cluster_id : Optional[int]
# ---------------------------------------------------------------------------

TileAllocTemplateStr = r"""
// TileAlloc: ${name}  (${nbytes} bytes in L1)
static volatile uintptr_t _addr_${name} = 0;
if (flex_is_first_core()) {
    _addr_${name} = (uintptr_t)flex_l1_malloc(${nbytes});
}
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

TileFreeTemplateStr = r"""
// TileFree: ${name}
flex_intra_cluster_sync();
if (flex_is_first_core()) {
    flex_l1_free((void*)(uintptr_t)${name});
}
flex_intra_cluster_sync();
"""

TileFreeTemplate = NodeTemplate(TileFreeTemplateStr)
