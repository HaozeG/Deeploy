# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""SoftHier NodeTemplates for cluster-group collective operations.

Templates
---------
TileGroupInitTemplate
    Called by ALL clusters.  Initialises a ``GridSyncGroupInfo`` via
    ``grid_sync_group_init(group_x, group_y)``.

TileGroupContextTemplate
    Declares per-cluster rank variables from ``GridSyncGroupInfo``:
    ``cluster_in_group_id_x/y``, ``cluster_for_rowwise``,
    ``cluster_for_colwise``.  Pattern copied from
    SummaGEMM.h::SummaGEMMAnaylze lines 131-140.

TileGroupBarrierTemplate
    Group-scoped barrier (all group members, no cluster/core guard).

TileAllocReducerTemplate
    L1 allocation on root cluster only; zero-initialised.

TileCollectiveReduceTemplate
    Axis-scoped DMA reduction via ``flex_dma_async_reduction`` with
    runtime ``${row_mask}`` / ``${col_mask}`` and ``${edge_flag}``
    edge-cluster gating.  Correct for both 1-D and 2-D groups.

TileCollectiveBroadcastTemplate
    Axis-scoped DMA broadcast via ``flex_dma_async_broadcast`` with
    runtime masks and edge-cluster gating.

TileAllocRootTemplate / TileCollectiveScatterTemplate / TileCollectiveGatherTemplate
    Root-only allocation and scatter/gather via sequential DMA.
"""

from Deeploy.DeeployTypes import NodeTemplate

# ---------------------------------------------------------------------------
# TileGroupInit
# ---------------------------------------------------------------------------

TileGroupInitTemplateStr = r"""
// GroupInit: ALL clusters call grid_sync_group_init for group '${group_id}'
// x_dim=${x_dim}, y_dim=${y_dim}.  Clusters outside the group receive valid_grid==0.
GridSyncGroupInfo group_info_${group_id} = grid_sync_group_init(${x_dim}, ${y_dim});
"""

TileGroupInitTemplate = NodeTemplate(TileGroupInitTemplateStr)

# ---------------------------------------------------------------------------
# TileGroupContext — per-cluster rank variables (pattern from SummaGEMM.h:131-140)
# NOTE: expressions kept on one line to avoid Mako treating % as a control char.
# ---------------------------------------------------------------------------

TileGroupContextTemplateStr = r"""<%
_num_groups = context.get('num_groups', 1)
%>
// GroupContext: per-cluster rank variables for group '${group_id}'
// Pattern copied from SummaGEMM.h::SummaGEMMAnaylze lines 131-140 and FlatAttentionUtil.h lines 206-213.
FlexPosition _pos_${group_id} = get_pos(flex_get_cluster_id());
uint32_t cluster_in_group_id_x_${group_id} = (group_info_${group_id}.valid_grid) ? (_pos_${group_id}.x % group_info_${group_id}.grid_x_dim) : 0;
uint32_t cluster_in_group_id_y_${group_id} = (group_info_${group_id}.valid_grid) ? (_pos_${group_id}.y % group_info_${group_id}.grid_y_dim) : 0;
uint32_t cluster_for_rowwise_${group_id} = (group_info_${group_id}.valid_grid) && ((cluster_in_group_id_x_${group_id} % group_info_${group_id}.grid_y_dim) == (cluster_in_group_id_y_${group_id} % group_info_${group_id}.grid_x_dim)) && (cluster_in_group_id_x_${group_id} == (_pos_${group_id}.y % group_info_${group_id}.grid_x_dim));
uint32_t cluster_for_colwise_${group_id} = (group_info_${group_id}.valid_grid) && ((cluster_in_group_id_x_${group_id} % group_info_${group_id}.grid_y_dim) == (cluster_in_group_id_y_${group_id} % group_info_${group_id}.grid_x_dim)) && (cluster_in_group_id_y_${group_id} == (_pos_${group_id}.x % group_info_${group_id}.grid_y_dim));
uint32_t cluster_active_${group_id} = group_info_${group_id}.valid_grid && (group_info_${group_id}.this_grid_id < ${_num_groups});
//    flex_global_barrier_xy();//Global barrier
//    GridSyncGroupInfo info = group_info_${group_id};
//    for (int cid = 0; cid < ARCH_NUM_CLUSTER; ++cid)
//    {
//        if (flex_get_core_id() == 0 && flex_get_cluster_id() == cid)
//        {
//            printf("[Cluster %3d] All Info: \n", cid);
//            printf("-- valid_grid = %0d \n", info.valid_grid);
//            printf("-- grid_x_dim = %0d \n", info.grid_x_dim);
//            printf("-- grid_y_dim = %0d \n", info.grid_y_dim);
//            printf("-- grid_x_num = %0d \n", info.grid_x_num);
//            printf("-- grid_y_num = %0d \n", info.grid_y_num);
//            printf("-- this_grid_id = %0d \n", info.this_grid_id);
//            printf("-- this_grid_id_x = %0d \n", info.this_grid_id_x);
//            printf("-- this_grid_id_y = %0d \n", info.this_grid_id_y);
//            printf("-- this_grid_left_most = %0d \n", info.this_grid_left_most);
//            printf("-- this_grid_right_most = %0d \n", info.this_grid_right_most);
//            printf("-- this_grid_top_most = %0d \n", info.this_grid_top_most);
//            printf("-- this_grid_bottom_most = %0d \n", info.this_grid_bottom_most);
//            printf("-- this_grid_cluster_num = %0d \n", info.this_grid_cluster_num);
//            printf("-- this_grid_cluster_num_x = %0d \n", info.this_grid_cluster_num_x);
//            printf("-- this_grid_cluster_num_y = %0d \n", info.this_grid_cluster_num_y);
//            printf("-- wakeup_row_mask = 0x%0x \n", info.wakeup_row_mask);
//            printf("-- wakeup_col_mask = 0x%0x \n", info.wakeup_col_mask);
//            printf("-- sync_x_cluster = %0d \n", info.sync_x_cluster);
//            printf("-- sync_y_cluster = %0d \n", info.sync_y_cluster);
//            printf("-- sync_x_point = 0x%0x \n", (uint32_t)info.sync_x_point);
//            printf("-- sync_x_piter = 0x%0x \n", (uint32_t)info.sync_x_piter);
//            printf("-- sync_y_point = 0x%0x \n", (uint32_t)info.sync_y_point);
//            printf("-- sync_y_piter = 0x%0x \n", (uint32_t)info.sync_y_piter);
//            printf("-- cluster_in_group_id_x = %0d \n", cluster_in_group_id_x_${group_id});
//            printf("-- cluster_in_group_id_y = %0d \n", cluster_in_group_id_y_${group_id});
//            printf("-- cluster_for_rowwise = %0d \n", cluster_for_rowwise_${group_id});
//            printf("-- cluster_for_colwise = %0d \n", cluster_for_colwise_${group_id});
//        }
//        flex_global_barrier_xy();//Global barrier
//    }
//    flex_global_barrier_xy();//Global barrier
"""

TileGroupContextTemplate = NodeTemplate(TileGroupContextTemplateStr)

# ---------------------------------------------------------------------------
# TileGroupBarrier
# ---------------------------------------------------------------------------

TileGroupBarrierTemplateStr = r"""
// GroupBarrier: all '${group_id}' members sync
grid_sync_group_barrier_xy(&group_info_${group_id});
"""

TileGroupBarrierTemplate = NodeTemplate(TileGroupBarrierTemplateStr)

# ---------------------------------------------------------------------------
# TileAllocReducer
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
# TileCollectiveReduce — runtime row/col masks, edge-cluster gating.
#
# OperatorRepresentation keys:
#   src_name, dst_name  : str
#   root_cluster_id     : int
#   group_id            : str
#   collective_op_kind  : str  — e.g. COLLECTIVE_REDADD_FP_16
#   row_mask            : str  — C expr, e.g. "group_info_tp.wakeup_row_mask"
#   col_mask            : str  — C expr, e.g. "(ARCH_NUM_CLUSTER_Y - 1)"
#   edge_flag           : str  — C expr, e.g. "cluster_for_rowwise_tp"
#   nbytes              : int
#   cluster_id          : None
# ---------------------------------------------------------------------------

TileCollectiveReduceTemplateStr = r"""
// All '${group_id}' members sync before reduction starts
grid_sync_group_barrier_xy(&group_info_${group_id});
// CollectiveReduce: ${collective_op_kind} over group '${group_id}'
// Edge cluster (${edge_flag}) accumulates ${src_name} -> ${dst_name}
if (flex_is_dm_core() && ${edge_flag}) {
    flex_dma_async_reduction(
        (uint32_t)(uintptr_t)${dst_name},
        (uint32_t)(uintptr_t)${src_name},
        ${nbytes},
        ${collective_op_kind},
        ${row_mask},
        ${col_mask}
    );
    flex_dma_async_wait_all();
}
// All group members sync (non-edge waits for edge to finish)
grid_sync_group_barrier_xy(&group_info_${group_id});
"""

TileCollectiveReduceTemplate = NodeTemplate(TileCollectiveReduceTemplateStr)

# ---------------------------------------------------------------------------
# TileCollectiveBroadcast — runtime row/col masks, edge-cluster gating.
#
# OperatorRepresentation keys:
#   src_name, dst_name  : str
#   root_cluster_id     : int
#   group_id            : str
#   row_mask            : str
#   col_mask            : str
#   edge_flag           : str
#   nbytes              : int
#   cluster_id          : None
# ---------------------------------------------------------------------------

TileCollectiveBroadcastTemplateStr = r"""
// CollectiveBroadcast: edge cluster (${edge_flag}) -> all '${group_id}' members
if (flex_is_dm_core() && ${edge_flag}) {
    flex_dma_async_broadcast(
        (uint32_t)(uintptr_t)${dst_name},
        (uint32_t)(uintptr_t)${src_name},
        ${nbytes},
        ${row_mask},
        ${col_mask}
    );
    flex_dma_async_wait_all();
}
// All group members sync (non-edge waits for edge to finish broadcast)
grid_sync_group_barrier_xy(&group_info_${group_id});
"""

TileCollectiveBroadcastTemplate = NodeTemplate(TileCollectiveBroadcastTemplateStr)

# ---------------------------------------------------------------------------
# TileAllocRoot
# ---------------------------------------------------------------------------

TileAllocRootTemplateStr = r"""
// TileAllocRoot: ${name} allocated on root cluster ${root_cluster_id} only (${nbytes} bytes)
static volatile uintptr_t _addr_${name} = 0;
if (flex_is_first_core() && flex_get_cluster_id() == ${root_cluster_id}) {
    _addr_${name} = (uintptr_t)flex_l1_malloc(${nbytes});
}
flex_intra_cluster_sync();
${dtype}* ${name} = (flex_get_cluster_id() == ${root_cluster_id})
    ? (${dtype}*)(uintptr_t)_addr_${name}
    : (${dtype}*)0;
flex_intra_cluster_sync();
"""

TileAllocRootTemplate = NodeTemplate(TileAllocRootTemplateStr)

# ---------------------------------------------------------------------------
# TileCollectiveScatter
# ---------------------------------------------------------------------------

TileCollectiveScatterTemplateStr = r"""
// CollectiveScatter: root ${root_cluster_id} -> all '${group_id}' members
if (flex_is_dm_core() && flex_get_cluster_id() == ${root_cluster_id}) {
    const uint32_t _scatter_ids[] = ${cluster_ids_c};
    for (int _si = 0; _si < ${num_members}; _si++) {
        flex_dma_async_1d(
            (uint64_t)flex_get_cluster_l1_ptr(_scatter_ids[_si], (void*)${dst_name}),
            (uint64_t)((uintptr_t)${src_name} + (uint64_t)_si * ${chunk_nbytes}),
            ${chunk_nbytes}
        );
    }
    flex_dma_async_wait_all();
}
// All group members sync (non-root waits for root to finish DMA)
grid_sync_group_barrier_xy(&group_info_${group_id});
"""

TileCollectiveScatterTemplate = NodeTemplate(TileCollectiveScatterTemplateStr)

# ---------------------------------------------------------------------------
# TileCollectiveGather
# ---------------------------------------------------------------------------

TileCollectiveGatherTemplateStr = r"""
// CollectiveGather: all '${group_id}' members -> root ${root_cluster_id}
if (flex_is_dm_core() && flex_get_cluster_id() == ${root_cluster_id}) {
    const uint32_t _gather_ids[] = ${cluster_ids_c};
    for (int _gi = 0; _gi < ${num_members}; _gi++) {
        flex_dma_async_1d(
            (uint64_t)((uintptr_t)${dst_name} + (uint64_t)_gi * ${chunk_nbytes}),
            (uint64_t)flex_get_cluster_l1_ptr(_gather_ids[_gi], (void*)${src_name}),
            ${chunk_nbytes}
        );
    }
    flex_dma_async_wait_all();
}
// All group members sync (non-root waits for root to finish DMA)
grid_sync_group_barrier_xy(&group_info_${group_id});
"""

TileCollectiveGatherTemplate = NodeTemplate(TileCollectiveGatherTemplateStr)
