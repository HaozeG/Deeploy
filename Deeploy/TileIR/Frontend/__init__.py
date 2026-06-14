# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR Frontend — TileLang AST → Deeploy ExecutionBlock."""

from Deeploy.TileIR.Frontend import tl_deeploy
from Deeploy.TileIR.Frontend.TilelangVisitor import TilelangVisitor
from Deeploy.TileIR.IR import (
    ClusterGroup,
    ClusterGroupRegistry,
    CollectiveBinding,
    CollectiveOpSpec,
    HardwareBinding,
    SoftHierCollectiveBackend,
)
from Deeploy.TileIR.Passes import (
    CollectiveLoweringPass,
    GroupAwareBarrierPass,
)

__all__ = [
    "tl_deeploy",
    "TilelangVisitor",
    "ClusterGroup",
    "ClusterGroupRegistry",
    "CollectiveBinding",
    "CollectiveOpSpec",
    "CollectiveLoweringPass",
    "GroupAwareBarrierPass",
    "HardwareBinding",
    "SoftHierCollectiveBackend",
]
