# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR pass infrastructure — base class and shared templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.TileIR.IR.TileBinding import TileBinding


# Shared template used by SoftwarePipelinePass and DedupSyncPass.
_INTRA_CLUSTER_SYNC_TEMPLATE = NodeTemplate("flex_intra_cluster_sync();\n")

# Global barrier template — used by GroupAwareBarrierPass and CollectiveLoweringPass.
_GLOBAL_BARRIER_TEMPLATE = NodeTemplate("""\
// Global barrier (cross-group or ungrouped cluster switch)
flex_global_barrier_xy();
""")


@dataclass
class TileBindingPass:
    """Base class for structural passes over ordered TileBindings."""

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        return bindings
