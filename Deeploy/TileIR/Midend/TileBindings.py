# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible re-exports.

Canonical locations::

    Deeploy.TileIR.IR.TileBinding          (TileBinding, TileOpKind, TileOpCategory)
    Deeploy.TileIR.Passes.Base              (TileBindingPass)
    Deeploy.TileIR.Passes.SoftwarePipeline  (SoftwarePipelinePass, ScopePhases)
    Deeploy.TileIR.Passes.Sync              (GlobalClusterBarrierPass, DedupSyncPass)
    Deeploy.TileIR.Passes.HoistAllocFree    (HoistAllocFreePass)
    Deeploy.TileIR.Midend.Pipeline          (TileBindingPipeline)
"""

from Deeploy.TileIR.IR.TileBinding import TileBinding as TileBinding
from Deeploy.TileIR.IR.TileBinding import TileOpCategory as TileOpCategory
from Deeploy.TileIR.IR.TileBinding import TileOpKind as TileOpKind
from Deeploy.TileIR.Midend.Pipeline import TileBindingPipeline as TileBindingPipeline
from Deeploy.TileIR.Passes.Base import TileBindingPass as TileBindingPass
from Deeploy.TileIR.Passes.HoistAllocFree import HoistAllocFreePass as HoistAllocFreePass
from Deeploy.TileIR.Passes.SoftwarePipeline import (
    ScopePhases as ScopePhases,
)
from Deeploy.TileIR.Passes.SoftwarePipeline import (
    SoftwarePipelinePass as SoftwarePipelinePass,
)
from Deeploy.TileIR.Passes.Sync import DedupSyncPass as DedupSyncPass
from Deeploy.TileIR.Passes.Sync import (
    GlobalClusterBarrierPass as GlobalClusterBarrierPass,
)

__all__ = [
    "TileBinding",
    "TileBindingPass",
    "GlobalClusterBarrierPass",
    "TileBindingPipeline",
    "SoftwarePipelinePass",
    "ScopePhases",
    "HoistAllocFreePass",
    "DedupSyncPass",
    "TileOpKind",
    "TileOpCategory",
]
