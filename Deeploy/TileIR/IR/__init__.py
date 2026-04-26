# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR IR — cluster group and collective operation primitives."""

from Deeploy.TileIR.IR.CollectivePrimitives import (
    ClusterGroup,
    ClusterGroupRegistry,
    CollectiveOpSpec,
    ShardMetadata,
    TensorLayout,
)
from Deeploy.TileIR.IR.HardwareBinding import (
    CollectiveBackend,
    HardwareBinding,
    HwTopology,
    Placement,
    SoftHierCollectiveBackend,
)
from Deeploy.TileIR.IR.ParallelPasses import (
    CollectiveBinding,
    CollectiveLoweringPass,
    GroupAwareBarrierPass,
)

__all__ = [
    "ClusterGroup",
    "ClusterGroupRegistry",
    "CollectiveOpSpec",
    "ShardMetadata",
    "TensorLayout",
    "HardwareBinding",
    "HwTopology",
    "Placement",
    "CollectiveBackend",
    "SoftHierCollectiveBackend",
    "CollectiveBinding",
    "CollectiveLoweringPass",
    "GroupAwareBarrierPass",
]
