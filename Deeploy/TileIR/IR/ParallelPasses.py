# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible re-exports.

Canonical locations::

    Deeploy.TileIR.IR.CollectiveBinding       (CollectiveBinding)
    Deeploy.TileIR.Passes.Sync                (GroupAwareBarrierPass)
    Deeploy.TileIR.Passes.CollectiveLowering  (CollectiveLoweringPass)
"""

from Deeploy.TileIR.IR.CollectiveBinding import CollectiveBinding as CollectiveBinding
from Deeploy.TileIR.Passes.CollectiveLowering import (
    CollectiveLoweringPass as CollectiveLoweringPass,
)
from Deeploy.TileIR.Passes.Sync import (
    GroupAwareBarrierPass as GroupAwareBarrierPass,
)

__all__ = [
    "CollectiveBinding",
    "GroupAwareBarrierPass",
    "CollectiveLoweringPass",
]
