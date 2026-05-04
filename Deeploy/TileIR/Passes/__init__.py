# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR Passes — structural transformation passes over TileBinding lists."""

from Deeploy.TileIR.Passes.Base import TileBindingPass
from Deeploy.TileIR.Passes.CollectiveLowering import CollectiveLoweringPass
from Deeploy.TileIR.Passes.HoistAllocFree import HoistAllocFreePass
from Deeploy.TileIR.Passes.SoftwarePipeline import (
    ScopePhases,
    SoftwarePipelinePass,
)
from Deeploy.TileIR.Passes.Sync import (
    DedupSyncPass,
    GlobalClusterBarrierPass,
    GroupAwareBarrierPass,
)
from Deeploy.TileIR.Passes.WorkPartitioning import WorkPartitioningPass

__all__ = [
    "TileBindingPass",
    "SoftwarePipelinePass",
    "ScopePhases",
    "GlobalClusterBarrierPass",
    "GroupAwareBarrierPass",
    "DedupSyncPass",
    "HoistAllocFreePass",
    "CollectiveLoweringPass",
    "WorkPartitioningPass",
]
