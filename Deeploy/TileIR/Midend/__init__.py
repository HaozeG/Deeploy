# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR Midend — pipeline orchestration and backward-compat re-exports."""

from Deeploy.TileIR.IR.TileBinding import TileBinding
from Deeploy.TileIR.IR.TileBinding import TileOpKind
from Deeploy.TileIR.Midend.Pipeline import TileBindingPipeline
from Deeploy.TileIR.Passes.Base import TileBindingPass
from Deeploy.TileIR.Passes.Sync import GlobalClusterBarrierPass

__all__ = [
    "TileBinding",
    "TileBindingPass",
    "GlobalClusterBarrierPass",
    "TileBindingPipeline",
    "TileOpKind",
]
