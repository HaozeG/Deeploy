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


class ClusterGuardTransformationPass(CodeTransformationPass):
    """Wrap snippets in a cluster guard when ``cluster_id`` is present."""

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity):
        transformed = ExecutionBlock()

        for snippet in executionBlock.codeSnippets:
            operator_representation = dict(snippet.operatorRepresentation)
            cluster_id = operator_representation.get("cluster_id", None)

            # Keep snippets unchanged when no cluster guard is requested.
            if cluster_id is None:
                transformed.addRight(snippet.template, operator_representation)
                continue

            template_source = snippet.template.template._source
            guarded_template = NodeTemplate(
                _CLUSTER_GUARD_OPEN + template_source + _CLUSTER_GUARD_CLOSE)
            transformed.addRight(guarded_template, operator_representation)

        return ctxt, transformed


def _default_tile_op_transformer() -> CodeTransformation:
    return CodeTransformation([ClusterGuardTransformationPass()])

# Per-op transformation registry. Start with empty pipelines and extend per op
# as dedicated TileIR passes are added.
_TILE_OP_TRANSFORMERS: Dict[str, CodeTransformation] = {}


def get_tile_op_transformer(op_kind: str) -> CodeTransformation:
    """Return the CodeTransformation configured for a TileIR op kind."""
    return _TILE_OP_TRANSFORMERS.get(op_kind, _default_tile_op_transformer())


def register_tile_op_transformer(op_kind: str, transformer: CodeTransformation) -> None:
    """Register/override the transformer for one TileIR op kind."""
    _TILE_OP_TRANSFORMERS[op_kind] = transformer
