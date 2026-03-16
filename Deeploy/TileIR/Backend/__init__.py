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
]
