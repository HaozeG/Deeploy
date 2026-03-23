# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR Backend — SoftHier-specific NodeTemplates."""

from Deeploy.TileIR.Backend.Templates.SoftHierTileTemplates import (
    TileLoadTemplate,
    TileStoreTemplate,
    TileCopyTemplate,
    TileFillTemplate,
    TileReduceTemplate,
    TileEltwiseTemplate,
    ForLoopOpenTemplate,
    ForLoopCloseTemplate,
    TileSyncTemplate,
    TileAllocTemplate,
    TileFreeTemplate,
)
from Deeploy.TileIR.Backend.Templates.SoftHierCollectiveTemplates import (
    TileGroupInitTemplate,
    TileGroupBarrierTemplate,
    TileAllocReducerTemplate,
    TileCollectiveReduceTemplate,
    TileCollectiveBroadcastTemplate,
)
from Deeploy.TileIR.Backend.Transformations import (
    ClusterGuardTransformationPass,
    PassThroughTransformationPass,
    get_tile_op_transformer,
    register_tile_op_transformer,
)

__all__ = [
    "TileLoadTemplate",
    "TileStoreTemplate",
    "TileCopyTemplate",
    "TileFillTemplate",
    "TileReduceTemplate",
    "TileEltwiseTemplate",
    "ForLoopOpenTemplate",
    "ForLoopCloseTemplate",
    "TileSyncTemplate",
    "TileAllocTemplate",
    "TileFreeTemplate",
    "TileGroupInitTemplate",
    "TileGroupBarrierTemplate",
    "TileAllocReducerTemplate",
    "TileCollectiveReduceTemplate",
    "TileCollectiveBroadcastTemplate",
    "ClusterGuardTransformationPass",
    "PassThroughTransformationPass",
    "get_tile_op_transformer",
    "register_tile_op_transformer",
]
