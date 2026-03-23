# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""SoftHier NodeTemplates for cluster-group collective operations.

Templates
---------
TileGroupInitTemplate
    Called by ALL clusters (no cluster guard).  Initialises a
    ``GridSyncGroupInfo`` for one cluster group.  Must be sandwiched by
    ``flex_global_barrier_xy()`` calls (done by ``CollectiveLoweringPass``).

TileGroupBarrierTemplate
    Group-scoped barrier.  Called by ALL group members (no cluster guard,
    no core guard).  Clusters outside the group have ``valid_grid == 0`` and
    are excluded from the synchronisation automatically by the hardware.

TileAllocReducerTemplate
    Allocates an L1 buffer on the root cluster only; other clusters skip.
    Zero-initialised via DMA.

TileCollectiveReduceTemplate
    Root cluster performs ``flex_dma_async_reduction`` to accumulate each
    member's partial buffer into its own ``dst_buffer``.  All group members
    then synchronise via ``grid_sync_group_barrier_xy``.

TileCollectiveBroadcastTemplate
    Root cluster broadcasts ``src_buffer`` to all members' ``dst_buffer``
    via ``flex_dma_async_broadcast``.  All group members then synchronise.

**SPMD barrier rationale**
After any operation guarded by ``if (CID == root_cluster_id)``, ALL group
members call ``grid_sync_group_barrier_xy`` to rendezvous.  Non-root clusters
reach the barrier immediately (having skipped the guarded work) and wait;
the root reaches it after finishing the DMA.  This matches the pattern in
``hello_world.c::test_group_barrier`` where the group barrier is placed
outside any cluster-guard ``if`` block.
"""

from Deeploy.DeeployTypes import NodeTemplate

# ---------------------------------------------------------------------------
# TileGroupInit — declare GridSyncGroupInfo for one cluster group.
#
# OperatorRepresentation keys:
#   group_id : str   — C-safe identifier string used in the variable name
#   x_dim    : int   — number of clusters in the group (grid x dimension)
#   y_dim    : int   — always 1 for now (grid y dimension)
#   cluster_id : None  — no cluster guard; all clusters execute this
# ---------------------------------------------------------------------------

TileGroupInitTemplateStr = r"""
// GroupInit: ALL clusters call grid_sync_group_init for group '${group_id}'
// Clusters outside the group receive valid_grid == 0 and are excluded.
GridSyncGroupInfo group_info_${group_id} = grid_sync_group_init(${x_dim}, ${y_dim});
"""

TileGroupInitTemplate = NodeTemplate(TileGroupInitTemplateStr)

# ---------------------------------------------------------------------------
# TileGroupBarrier — group-scoped barrier (no cluster/core guard).
#
# OperatorRepresentation keys:
#   group_id : str
#   cluster_id : None
# ---------------------------------------------------------------------------

TileGroupBarrierTemplateStr = r"""
// GroupBarrier: all '${group_id}' members sync
grid_sync_group_barrier_xy(&group_info_${group_id});
"""

TileGroupBarrierTemplate = NodeTemplate(TileGroupBarrierTemplateStr)

# ---------------------------------------------------------------------------
# TileAllocReducer — L1 allocation on root cluster only; zero-initialised.
#
# OperatorRepresentation keys:
#   name             : str  — C variable name
#   dtype            : str  — C element type (e.g. "fp16")
#   nbytes           : int  — allocation size in bytes
#   root_cluster_id  : int  — cluster ID of the root
#   cluster_id       : int  — same as root_cluster_id (set by CollectiveLoweringPass)
# ---------------------------------------------------------------------------

TileAllocReducerTemplateStr = r"""
// TileAllocReducer: ${name} on root cluster ${root_cluster_id} only (${nbytes} bytes)
static volatile uintptr_t _addr_${name} = 0;
if (flex_is_first_core() && flex_get_cluster_id() == ${root_cluster_id}) {
    _addr_${name} = (uintptr_t)flex_l1_malloc(${nbytes});
}
flex_intra_cluster_sync();
${dtype}* ${name} = (${root_cluster_id} == flex_get_cluster_id())
    ? (${dtype}*)(uintptr_t)_addr_${name}
    : (${dtype}*)0;
// Zero-initialise the reducer buffer on root
if (flex_is_dm_core() && flex_get_cluster_id() == ${root_cluster_id}) {
    flex_dma_async_1d((uint64_t)(uintptr_t)${name}, zomem(0), ${nbytes});
    flex_dma_async_wait_all();
}
flex_intra_cluster_sync();
"""

TileAllocReducerTemplate = NodeTemplate(TileAllocReducerTemplateStr)

# ---------------------------------------------------------------------------
# TileCollectiveReduce — root reads each member's src buffer and reduces.
#
# OperatorRepresentation keys:
#   src_name           : str  — source (partial result) buffer name
#   dst_name           : str  — destination (accumulator) buffer name (root-local)
#   root_cluster_id    : int
#   group_id           : str
#   collective_op_kind : str  — C enum, e.g. FLEX_DMA_REDUCTION_SUM
#   src_bitmask        : str  — hex bitmask of source cluster IDs
#   dst_bitmask        : str  — hex bitmask of destination cluster IDs
#   nbytes             : int
#   cluster_id         : None — guard is inside template
# ---------------------------------------------------------------------------

TileCollectiveReduceTemplateStr = r"""
// CollectiveReduce: ${collective_op_kind} over group '${group_id}'
// Root cluster ${root_cluster_id} reads each member's ${src_name} -> ${dst_name}
if (flex_is_dm_core() && flex_get_cluster_id() == ${root_cluster_id}) {
    flex_dma_async_reduction(
        (uint32_t)(uintptr_t)${src_name},
        (uint32_t)(uintptr_t)${dst_name},
        ${nbytes},
        ${collective_op_kind},
        ${src_bitmask},
        ${dst_bitmask}
    );
    flex_dma_async_wait_all();
}
// All group members sync (non-root waits for root to finish)
grid_sync_group_barrier_xy(&group_info_${group_id});
"""

TileCollectiveReduceTemplate = NodeTemplate(TileCollectiveReduceTemplateStr)

# ---------------------------------------------------------------------------
# TileCollectiveBroadcast — root sends its result buffer to all members.
#
# OperatorRepresentation keys:
#   src_name        : str  — source buffer name (on root)
#   dst_name        : str  — destination buffer name (on each member)
#   root_cluster_id : int
#   group_id        : str
#   src_bitmask     : str  — hex bitmask
#   dst_bitmask     : str  — hex bitmask
#   nbytes          : int
#   cluster_id      : None — guard is inside template
# ---------------------------------------------------------------------------

TileCollectiveBroadcastTemplateStr = r"""
// CollectiveBroadcast: root ${root_cluster_id} -> all '${group_id}' members
if (flex_is_dm_core() && flex_get_cluster_id() == ${root_cluster_id}) {
    flex_dma_async_broadcast(
        (uint32_t)(uintptr_t)${src_name},
        (uint32_t)(uintptr_t)${dst_name},
        ${nbytes},
        ${src_bitmask},
        ${dst_bitmask}
    );
    flex_dma_async_wait_all();
}
// All group members sync (non-root waits for root to finish broadcast)
grid_sync_group_barrier_xy(&group_info_${group_id});
"""

TileCollectiveBroadcastTemplate = NodeTemplate(TileCollectiveBroadcastTemplateStr)
