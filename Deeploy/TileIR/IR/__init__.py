# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR IR — cluster group, collective operation, and binding primitives."""

from Deeploy.TileIR.IR.CollectiveBinding import CollectiveBinding
from Deeploy.TileIR.IR.CollectivePrimitives import (
    ClusterGroup,
    ClusterGroupRegistry,
    CollectiveOpSpec,
    TensorLayout,
)
from Deeploy.TileIR.IR.HardwareBinding import (
    CollectiveBackend,
    HardwareBinding,
    HwTopology,
    Placement,
    SoftHierCollectiveBackend,
)
from Deeploy.TileIR.IR.TileBinding import (
    TileBinding,
    TileOpCategory,
    TileOpKind,
)
# Backward compat: pass classes previously exported from IR/__init__.py
from Deeploy.TileIR.Passes.CollectiveLowering import (
    CollectiveLoweringPass,
)
from Deeploy.TileIR.Passes.Sync import (
    GroupAwareBarrierPass,
)

__all__ = [
    "ClusterGroup",
    "ClusterGroupRegistry",
    "CollectiveOpSpec",
    "TensorLayout",
    "HardwareBinding",
    "HwTopology",
    "Placement",
    "CollectiveBackend",
    "SoftHierCollectiveBackend",
    "TileBinding",
    "TileOpKind",
    "TileOpCategory",
    "CollectiveBinding",
    "CollectiveLoweringPass",
    "GroupAwareBarrierPass",
]
