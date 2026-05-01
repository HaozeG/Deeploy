# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""CollectiveBinding — a TileBinding with a CollectiveOpSpec for backend lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from Deeploy.TileIR.IR.TileBinding import TileBinding, TileOpKind

if TYPE_CHECKING:
    from Deeploy.TileIR.IR.CollectivePrimitives import CollectiveOpSpec


@dataclass
class CollectiveBinding(TileBinding):
    """A TileBinding that carries a CollectiveOpSpec for backend lowering.

    Parameters
    ----------
    spec : CollectiveOpSpec
        Logical description of the collective operation.  Used by
        ``CollectiveLoweringPass`` to call the hardware backend.
    """

    spec: Optional["CollectiveOpSpec"] = None

    def __post_init__(self) -> None:
        if self.op_kind != "group_collective":
            self.op_kind = TileOpKind.group_collective
        super().__post_init__()
