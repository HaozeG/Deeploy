# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""WorkPartitioningPass — generic 1D data-parallel work division across clusters.

For non-tiled (non-GEMM) kernels, this pass emits per-cluster ``chunk_start``
and ``chunk_end`` variables that partition a linear work range across the
available clusters.  This is the SoftHier analogue of the ``divide_task``
macro pattern used in GPU TileLang kernels.

Pattern
-------
The pass is inserted into a ``TileBindingPipeline`` via
``pipeline.add_binding_pass(WorkPartitioningPass(num_elements=<expr>))``.
It prepends a preamble that computes::

    uint32_t _num_elems = <num_elements>;
    uint32_t _num_clusters = ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y;
    uint32_t _cid = flex_get_cluster_id();
    uint32_t _chunk = _num_elems / _num_clusters;
    uint32_t _chunk_start = _cid * _chunk;
    uint32_t _chunk_end = (_cid == _num_clusters - 1)
                        ? _num_elems
                        : _chunk_start + _chunk;

"""

from __future__ import annotations

from typing import List, Optional

from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.TileIR.IR.TileBinding import TileBinding
from Deeploy.TileIR.Passes.Base import TileBindingPass

_WORK_PARTITION_TEMPLATE = NodeTemplate(r"""// WorkPartition: ${num_elements} elements across ${num_clusters} clusters
uint32_t _num_elems = ${num_elements};
uint32_t _cid = flex_get_cluster_id();
uint32_t _chunk = _num_elems / ${num_clusters};
uint32_t _chunk_start = _cid * _chunk;
uint32_t _chunk_end = (_cid == ${num_clusters} - 1) ? _num_elems : _chunk_start + _chunk;
""")


class WorkPartitioningPass(TileBindingPass):
    """Prepend work-partitioning preamble for 1D data-parallel kernels.

    Parameters
    ----------
    num_elements : str
        C expression for the total number of elements (e.g. ``"M"``,
        ``"num_tokens"``).
    num_clusters : int or str
        Number of clusters to partition across.  Defaults to the
        architecture constant ``ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y``
        which is the full chip.
    """

    def __init__(
        self,
        num_elements: str,
        num_clusters: Optional[str] = None,
    ):
        self._num_elements = num_elements
        self._num_clusters = num_clusters or "ARCH_NUM_CLUSTER_X * ARCH_NUM_CLUSTER_Y"

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        rep = {
            "num_elements": self._num_elements,
            "num_clusters": self._num_clusters,
            "cluster_id": None,
            "shard_metadata": None,
        }
        preamble = TileBinding(
            op_kind="block_preamble",
            template=_WORK_PARTITION_TEMPLATE,
            operator_representation=rep,
            op_name="tile_work_partition",
        )
        return [preamble] + list(bindings)
