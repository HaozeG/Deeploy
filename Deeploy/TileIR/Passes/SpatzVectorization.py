# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Pass that rewrites scalar TileEltwise bindings to Spatz vector instructions.

Inspects the stringified ``src_expr`` of each eltwise binding and replaces
the template with ``SpatzEltwiseTemplate`` when the operation has a known
RVV vector equivalent (ReLU → vfmax.vf, sigmoid → vfexp.vv, etc.).

Usage
-----
    from Deeploy.TileIR.Passes.SpatzVectorization import SpatzVectorizationPass
    bindings = SpatzVectorizationPass().apply(bindings)
"""

from __future__ import annotations

import re
from typing import List

from Deeploy.TileIR.Backend.Templates.SoftHierTileTemplates import (
    SpatzEltwiseTemplate,
    TileEltwiseTemplate,
)
from Deeploy.TileIR.IR.TileBinding import TileBinding
from Deeploy.TileIR.Passes.Base import TileBindingPass


class SpatzVectorizationPass(TileBindingPass):
    """Replace scalar eltwise bindings with Spatz vector equivalents."""

    # Map from regex pattern on src_expr → (spatz_op, spatz_setup, spatz_body)
    # Patterns must match the C expression produced by _ExprStringifier.
    _RULES: List[tuple] = [
        # ── T.max(buf[i], 0) → ReLU via vfmax.vf ──────────────────────
        # Generated C: tile_fp16_max(((fp16*)buf)[i], 0.0)
        (
            re.compile(
                r'tile_fp16_max\(\(\(fp16\*\)(?P<buf>\w+)\)\[[^\]]+\],\s*0\.0\)'
            ),
            "relu",
            'fp16 _zero = 0;\n'
            '    asm volatile("fld fa5, (%0)" ::"r"(&_zero));',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr));\n'
            '    asm volatile("vfmax.vf v8, v8, fa5");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));',
        ),
        # ── T.sigmoid(buf[i]) → sigmoid via vfexp.vv ──────────────────
        # Generated C: tile_fp16_sigmoid(((fp16*)buf)[i])
        (
            re.compile(
                r'tile_fp16_sigmoid\(\(\(fp16\*\)(?P<buf>\w+)\)\[[^\]]+\]\)'
            ),
            "sigmoid",
            'fp16 _minus_one = 0xbc00;\n'
            '    asm volatile("fld fa6, (%0)" ::"r"(&_minus_one));',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr));\n'
            '    asm volatile("vfmul.vf v8, v8, fa6");\n'
            '    asm volatile(".word 0x32041857" /*vfexp.vv v16, v8*/);\n'
            '    asm volatile("vfdiv.vv v0, v16, v16");\n'
            '    asm volatile("vfadd.vv v16, v16, v0");\n'
            '    asm volatile("vfdiv.vv v8, v0, v16");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));',
        ),
    ]

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        for b in bindings:
            if b.op_kind != "eltwise" or b.template is not TileEltwiseTemplate:
                continue
            rep = b.operator_representation
            src_expr = rep.get("src_expr", "")
            for pattern, spatz_op, spatz_setup, spatz_body in self._RULES:
                if pattern.search(src_expr):
                    b.template = SpatzEltwiseTemplate
                    rep["spatz_op"] = spatz_op
                    rep["num_spatz"] = "ARCH_SPATZ_ATTACED_CORES"
                    rep["spatz_setup"] = spatz_setup
                    rep["spatz_body"] = spatz_body
                    # Preserve src_expr as fallback for non-Spatz cores
                    rep["fallback_expr"] = src_expr
                    rep.pop("index_expr", None)
                    break
        return bindings
