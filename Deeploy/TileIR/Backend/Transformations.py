# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""TileIR Midend transformation helpers.

Each TileBinding owns a CodeTransformation object. This module provides a
small registry to construct per-op transformers.

By default, every TileBinding applies ``ClusterGuardTransformationPass``:
if the snippet carries ``cluster_id`` in its operator representation and that
value is not ``None``, the snippet body is wrapped with a
``flex_get_cluster_id()`` guard.

Collective op kinds (``group_collective``, ``group_barrier``,
``alloc_reducer``) embed their own guards inside the template body and
therefore use a pass-through transformer (no additional cluster guard).
"""

from __future__ import annotations

from typing import Dict

from Deeploy.DeeployTypes import (
    CodeGenVerbosity,
    CodeTransformation,
    CodeTransformationPass,
    ExecutionBlock,
    NetworkContext,
    NodeTemplate,
    _NoVerbosity,
)

_CLUSTER_GUARD_OPEN = """\
% if cluster_id is not None:
{
    uint32_t CID = flex_get_cluster_id();
    if (CID == ${cluster_id}) {
% endif
"""

_CLUSTER_GUARD_CLOSE = """\
% if cluster_id is not None:
    } // end if CID == ${cluster_id}
}
% endif
"""

_CLUSTER_MAP_GUARD_OPEN = """\
if (_cluster_active) {
"""

_CLUSTER_MAP_GUARD_CLOSE = """\
} // end cluster-map guard
"""

_GROUP_MEMBERSHIP_GUARD_OPEN = """\
% if cluster_guard_type == "group_membership" and group_cluster_ids:
<%
_grp_bitmask = 0
for _cid in group_cluster_ids:
    _grp_bitmask |= (1 << _cid)
%>{
    uint32_t CID = flex_get_cluster_id();
    if ((1U << CID) & ${hex(_grp_bitmask)}U) {
% endif
"""

_GROUP_MEMBERSHIP_GUARD_CLOSE = """\
% if cluster_guard_type == "group_membership" and group_cluster_ids:
    } // end group membership guard
}
% endif
"""


class ClusterGuardTransformationPass(CodeTransformationPass):
    """Wrap snippets in a cluster guard when ``cluster_id`` is present.

    If ``cluster_guard_type == "group_membership"`` and ``group_cluster_ids``
    is set, a multi-cluster OR guard is emitted instead of a single-cluster
    guard.
    """

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity):
        transformed = ExecutionBlock()

        for snippet in executionBlock.codeSnippets:
            operator_representation = dict(snippet.operatorRepresentation)
            cluster_id = operator_representation.get("cluster_id", None)
            guard_type = operator_representation.get("cluster_guard_type", None)

            # Group-membership guard (alloc inside a cluster group)
            if guard_type == "group_membership":
                group_cluster_ids = operator_representation.get("group_cluster_ids", [])
                if group_cluster_ids:
                    template_source = snippet.template.template._source
                    bitmask = 0
                    for cid in group_cluster_ids:
                        bitmask |= (1 << cid)
                    guard_open = (
                        "{\n"
                        "    uint32_t CID = flex_get_cluster_id();\n"
                        f"    if ((1U << CID) & {hex(bitmask)}U) {{\n"
                    )
                    guard_close = (
                        "    } // end group membership guard\n"
                        "}\n"
                    )
                    guarded_template = NodeTemplate(
                        guard_open + template_source + guard_close)
                    transformed.addRight(guarded_template, operator_representation)
                    continue

            # Cluster-map lookup-table guard (list-based mapping)
            # cluster_map/block_id_expr may be top-level or inside "metadata"
            metadata = operator_representation.get("metadata", {}) or {}
            cluster_map = operator_representation.get("cluster_map", None) or metadata.get("cluster_map", None)
            block_id_expr = operator_representation.get("block_id_expr", None) or metadata.get("block_id_expr", None)
            if cluster_map is not None and block_id_expr is not None:
                # Promote to top-level so Mako templates can access cluster_map/block_id_expr.
                # No per-op _cluster_active guard needed: the block preamble's
                # `if (!_cluster_active) continue;` already excludes inactive clusters.
                operator_representation["cluster_map"] = cluster_map
                operator_representation["block_id_expr"] = block_id_expr
                transformed.addRight(snippet.template, operator_representation)
                continue

            # Standard single-cluster guard
            if cluster_id is None:
                transformed.addRight(snippet.template, operator_representation)
                continue

            template_source = snippet.template.template._source
            guarded_template = NodeTemplate(
                _CLUSTER_GUARD_OPEN + template_source + _CLUSTER_GUARD_CLOSE)
            transformed.addRight(guarded_template, operator_representation)

        return ctxt, transformed


class PassThroughTransformationPass(CodeTransformationPass):
    """No-op transformation pass — used for ops that embed their own guards."""

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity):
        return ctxt, executionBlock


def _default_tile_op_transformer() -> CodeTransformation:
    return CodeTransformation([ClusterGuardTransformationPass()])


def _passthrough_transformer() -> CodeTransformation:
    return CodeTransformation([PassThroughTransformationPass()])


# Per-op transformation registry.
_TILE_OP_TRANSFORMERS: Dict[str, CodeTransformation] = {}


def get_tile_op_transformer(op_kind: str) -> CodeTransformation:
    """Return the CodeTransformation configured for a TileIR op kind."""
    return _TILE_OP_TRANSFORMERS.get(op_kind, _default_tile_op_transformer())


def register_tile_op_transformer(op_kind: str, transformer: CodeTransformation) -> None:
    """Register/override the transformer for one TileIR op kind."""
    _TILE_OP_TRANSFORMERS[op_kind] = transformer


# ---------------------------------------------------------------------------
# Register collective op-kind transformers.
#
# group_collective, group_barrier, alloc_reducer:  templates embed their own
# cluster-guard logic, so a pass-through transformer is correct here.
# ---------------------------------------------------------------------------

register_tile_op_transformer("group_collective", _passthrough_transformer())
register_tile_op_transformer("group_barrier", _passthrough_transformer())
register_tile_op_transformer("alloc_reducer", _default_tile_op_transformer())
# alloc/free embed their own cluster guards in the template body
register_tile_op_transformer("alloc", _passthrough_transformer())
register_tile_op_transformer("free", _passthrough_transformer())
# block_preamble declares _bid/_cluster_active once per tile block; no guard needed
register_tile_op_transformer("block_preamble", _passthrough_transformer())
# group_preamble declares per-group active guard; template embeds own logic
register_tile_op_transformer("group_preamble", _passthrough_transformer())
# sync (flex_intra_cluster_sync) must be called by ALL cores in a cluster —
# wrapping it in a cluster guard would cause a deadlock.
register_tile_op_transformer("sync", _passthrough_transformer())
# global_barrier (flex_global_barrier_xy) is a full-chip sync — no guard.
register_tile_op_transformer("global_barrier", _passthrough_transformer())
# intra_cluster_reduce embeds its own core-dispatch logic; no guard needed.
register_tile_op_transformer("intra_cluster_reduce", _passthrough_transformer())
# pipelined for-loop open/close brackets: no cluster guard needed
register_tile_op_transformer("pipelined_for_open", _passthrough_transformer())
register_tile_op_transformer("pipelined_for_close", _passthrough_transformer())
# math_preamble — emitted once; passthrough
register_tile_op_transformer("math_preamble", _passthrough_transformer())
# runtime_assert / runtime_assume — guards already handled by surrounding context
register_tile_op_transformer("runtime_assert", _default_tile_op_transformer())
register_tile_op_transformer("runtime_assume", _passthrough_transformer())
# cumsum — sequential L1 loop, uses cluster guard
register_tile_op_transformer("cumsum", _default_tile_op_transformer())
