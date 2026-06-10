# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Pass that rewrites scalar TileEltwise bindings to Spatz vector instructions.

Inspects the stringified ``src_expr`` of each eltwise binding and replaces
the template with ``SpatzEltwiseTemplate`` when the operation has a known
RVV vector equivalent (ReLU -> vfmax.vf, sigmoid -> vfexp.vv, etc.).

Usage
-----
    from Deeploy.TileIR.Passes.SpatzVectorization import SpatzVectorizationPass
    bindings = SpatzVectorizationPass().apply(bindings)
"""

from __future__ import annotations

import re
from typing import List

from Deeploy.TileIR.Backend.Templates.SoftHierTileTemplates import (
    SpatzContextTemplate,
    SpatzEltwiseTemplate,
    SpatzInnerEltwiseTemplate,
    TileEltwiseTemplate,
    TileInnerEltwiseTemplate,
)
from Deeploy.TileIR.IR.TileBinding import TileBinding, TileOpKind
from Deeploy.TileIR.Passes.Base import TileBindingPass

_OUTER_OFFSET_RE = re.compile(
    r'^\(?\s*(?P<outer>\w+)\s*\*\s*(?P<stride>\d+)\s*\)?\s*\+\s*'
    r'(?P<inner>\w+)$'
)


def _outer_offset(index_expr: str, loop_var: str):
    """Parse a linearised 2-D index ``outer * stride + inner`` where
    ``inner == loop_var``.  Returns ``(outer_var, stride)`` when the
    pattern matches and ``inner`` equals ``loop_var``; otherwise
    ``(None, 1)``.

    This detects the case where a flat ``T.Parallel(inner)`` is nested
    inside a serialised outer loop over ``outer``: the Spatz address must
    include an ``outer * stride`` row offset, otherwise every iteration
    of the outer loop overwrites the same buffer slice.
    """
    m = _OUTER_OFFSET_RE.match(index_expr.strip())
    if m and m.group("inner") == loop_var:
        return m.group("outer"), int(m.group("stride"))
    return None, 1


class SpatzVectorizationPass(TileBindingPass):
    """Replace scalar eltwise bindings with Spatz vector equivalents."""

    # Map from regex pattern on src_expr -> (spatz_op, spatz_setup, spatz_body)
    # Patterns must match the C expression produced by _ExprStringifier.
    #
    # Rule ordering is significant: vv (same-index) rules are tried before vf
    # (scalar) rules so that buf[i] OP buf[i] is never mis-classified as
    # buf[i] OP scalar.  The loop breaks on first match.
    _RULES: List[tuple] = [
        # T.max(buf[i], 0) -> ReLU via vfmax.vf
        # Generated C: tile_fp16_max(((fp16*)buf)[i], 0.0)
        (
            re.compile(
                r'tile_fp16_max\(\(\(fp16\*\)(?P<buf>\w+)\)\[[^\]]+\],\s*0\.0\)'
            ),
            "relu",
            'fp16 _zero = 0;\n'
            '    asm volatile("fld fa5, (%0)" ::"r"(&_zero));\n'
            '    uint32_t _addr_src = (uint32_t)(uintptr_t){buf} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src));\n'
            '    asm volatile("vfmax.vf v8, v8, fa5");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src += _avl * sizeof(fp16);',
        ),
        # T.max(buf1[i], buf2[i]) -> binary element-wise max via vfmax.vv
        # Generated C: tile_fp16_max(((fp16*)buf1)[i], ((fp16*)buf2)[i])
        # {buf1}/{buf2} are Python format placeholders filled from regex groups.
        # Backreference (?P=idx) ensures both accesses use the SAME index so we
        # don't accidentally vectorize serial-reduction patterns like
        # cur_max[_i_scalar] = T.max(cur_max[_i_scalar], x_l[i]) where the
        # dst uses the eltwise var and the src uses an outer serial-loop var.
        (
            re.compile(
                r'tile_fp16_max\(\(\(fp16\*\)(?P<buf1>\w+)\)\[(?P<idx>[^\]]+)\],\s*'
                r'\(\(fp16\*\)(?P<buf2>\w+)\)\[(?P=idx)\]\)'
            ),
            "max",
            # _spatz_sid and _vlen are in scope (set by SpatzEltwiseTemplate)
            'uint32_t _addr_src1 = (uint32_t)(uintptr_t){buf1} + _spatz_sid * _vlen * sizeof(fp16);\n'
            '    uint32_t _addr_src2 = (uint32_t)(uintptr_t){buf2} + _spatz_sid * _vlen * sizeof(fp16);',
            # v8 (m8 group v8-v15) = src1, v16 (m8 group v16-v23) = src2
            # _addr (= dst) is advanced by the template loop; sources are advanced manually.
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src1));\n'
            '    asm volatile("vle16.v v16, (%0)" ::"r"(_addr_src2));\n'
            '    asm volatile("vfmax.vv v8, v8, v16");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src1 += _avl * sizeof(fp16);\n'
            '    _addr_src2 += _avl * sizeof(fp16);',
        ),
        # T.sigmoid(buf[i]) -> sigmoid via vfexp.vv
        # Generated C: tile_fp16_sigmoid(((fp16*)buf)[i])
        (
            re.compile(
                r'tile_fp16_sigmoid\(\(\(fp16\*\)(?P<buf>\w+)\)\[[^\]]+\]\)'
            ),
            "sigmoid",
            'fp16 _minus_one = 0xbc00;\n'
            '    asm volatile("fld fa5, (%0)" ::"r"(&_minus_one));\n'
            '    uint32_t _addr_src = (uint32_t)(uintptr_t){buf} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src));\n'
            '    asm volatile("vfmul.vf v8, v8, fa6");\n'
            '    asm volatile(".word 0x32041857" /*vfexp.vv v16, v8*/);\n'
            '    asm volatile("vfdiv.vv v0, v16, v16");\n'
            '    asm volatile("vfadd.vv v16, v16, v0");\n'
            '    asm volatile("vfdiv.vv v8, v0, v16");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src += _avl * sizeof(fp16);',
        ),
        # T.exp(buf[i]) -> vectorized exp via custom vfexp instruction
        # Generated C: tile_fp16_exp(((fp16*)buf)[i])
        (
            re.compile(
                r'tile_fp16_exp\(\(\(fp16\*\)(?P<buf>\w+)\)\[[^\]]+\]\)'
            ),
            "exp",
            'uint32_t _addr_src = (uint32_t)(uintptr_t){buf} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src));\n'
            '    asm volatile(".word 0x32041857" /*vfexp.vv v16, v8*/);\n'
            '    asm volatile("vse16.v v16, (%0)" ::"r"(_addr));\n'
            '    _addr_src += _avl * sizeof(fp16);',
        ),
        # T.exp(buf1[i] - buf2[j]) -> vfsub.vf + vfexp (core attention softmax op)
        # Generated C: tile_fp16_exp(float_to_fp16(fp16_to_float(((fp16*)buf1)[i])
        #                            - fp16_to_float(((fp16*)buf2)[j])))
        # buf2[j] is a scalar buffer accessed with an outer-loop index j != i.
        # The safety check in apply() rejects this rule when j contains the
        # eltwise loop variable (meaning buf2 is not actually a scalar).
        (
            re.compile(
                r'tile_fp16_exp\(float_to_fp16\('
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf1>\w+)\)\[(?P<idx1>[^\]]+)\]\)'
                r'\s*-\s*'
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf2>\w+)\)\[(?P<idx2>[^\]]+)\]\)'
                r'\)\)'
            ),
            "exp_sub",
            'fp16 _scalar = ((fp16*){buf2})[{idx2}];\n'
            '    asm volatile("fld fa5, (%0)" ::"r"(&_scalar));\n'
            '    uint32_t _addr_src = (uint32_t)(uintptr_t){buf1} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src));\n'
            '    asm volatile("vfsub.vf v8, v8, fa5");\n'
            '    asm volatile(".word 0x32041857" /*vfexp.vv v16, v8*/);\n'
            '    asm volatile("vse16.v v16, (%0)" ::"r"(_addr));\n'
            '    _addr_src += _avl * sizeof(fp16);',
        ),
        # T.add(buf1[i], buf2[i]) -> vfadd.vv (same index, backreference)
        # Generated C: float_to_fp16(fp16_to_float(((fp16*)buf1)[i])
        #              + fp16_to_float(((fp16*)buf2)[i]))
        (
            re.compile(
                r'float_to_fp16\('
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf1>\w+)\)\[(?P<idx>[^\]]+)\]\)'
                r'\s*\+\s*'
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf2>\w+)\)\[(?P=idx)\]\)'
                r'\)'
            ),
            "add_vv",
            'uint32_t _addr_src1 = (uint32_t)(uintptr_t){buf1} + _spatz_sid * _vlen * sizeof(fp16);\n'
            '    uint32_t _addr_src2 = (uint32_t)(uintptr_t){buf2} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src1));\n'
            '    asm volatile("vle16.v v16, (%0)" ::"r"(_addr_src2));\n'
            '    asm volatile("vfadd.vv v8, v8, v16");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src1 += _avl * sizeof(fp16);\n'
            '    _addr_src2 += _avl * sizeof(fp16);',
        ),
        # T.sub(buf1[i], buf2[i]) -> vfsub.vv (same index)
        (
            re.compile(
                r'float_to_fp16\('
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf1>\w+)\)\[(?P<idx>[^\]]+)\]\)'
                r'\s*-\s*'
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf2>\w+)\)\[(?P=idx)\]\)'
                r'\)'
            ),
            "sub_vv",
            'uint32_t _addr_src1 = (uint32_t)(uintptr_t){buf1} + _spatz_sid * _vlen * sizeof(fp16);\n'
            '    uint32_t _addr_src2 = (uint32_t)(uintptr_t){buf2} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src1));\n'
            '    asm volatile("vle16.v v16, (%0)" ::"r"(_addr_src2));\n'
            '    asm volatile("vfsub.vv v8, v8, v16");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src1 += _avl * sizeof(fp16);\n'
            '    _addr_src2 += _avl * sizeof(fp16);',
        ),
        # T.mul(buf1[i], buf2[i]) -> vfmul.vv (same index)
        (
            re.compile(
                r'float_to_fp16\('
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf1>\w+)\)\[(?P<idx>[^\]]+)\]\)'
                r'\s*\*\s*'
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf2>\w+)\)\[(?P=idx)\]\)'
                r'\)'
            ),
            "mul_vv",
            'uint32_t _addr_src1 = (uint32_t)(uintptr_t){buf1} + _spatz_sid * _vlen * sizeof(fp16);\n'
            '    uint32_t _addr_src2 = (uint32_t)(uintptr_t){buf2} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src1));\n'
            '    asm volatile("vle16.v v16, (%0)" ::"r"(_addr_src2));\n'
            '    asm volatile("vfmul.vv v8, v8, v16");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src1 += _avl * sizeof(fp16);\n'
            '    _addr_src2 += _avl * sizeof(fp16);',
        ),
        # T.add(buf1[i], buf2[j]) -> vfadd.vf  (j is scalar/outer index)
        # Tried after add_vv; apply() rejects if idx2 contains loop_var.
        (
            re.compile(
                r'float_to_fp16\('
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf1>\w+)\)\[(?P<idx1>[^\]]+)\]\)'
                r'\s*\+\s*'
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf2>\w+)\)\[(?P<idx2>[^\]]+)\]\)'
                r'\)'
            ),
            "add_vf",
            'fp16 _scalar = ((fp16*){buf2})[{idx2}];\n'
            '    asm volatile("fld fa5, (%0)" ::"r"(&_scalar));\n'
            '    uint32_t _addr_src = (uint32_t)(uintptr_t){buf1} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src));\n'
            '    asm volatile("vfadd.vf v8, v8, fa5");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src += _avl * sizeof(fp16);',
        ),
        # T.sub(buf1[i], buf2[j]) -> vfsub.vf  (j is scalar/outer index)
        (
            re.compile(
                r'float_to_fp16\('
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf1>\w+)\)\[(?P<idx1>[^\]]+)\]\)'
                r'\s*-\s*'
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf2>\w+)\)\[(?P<idx2>[^\]]+)\]\)'
                r'\)'
            ),
            "sub_vf",
            'fp16 _scalar = ((fp16*){buf2})[{idx2}];\n'
            '    asm volatile("fld fa5, (%0)" ::"r"(&_scalar));\n'
            '    uint32_t _addr_src = (uint32_t)(uintptr_t){buf1} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src));\n'
            '    asm volatile("vfsub.vf v8, v8, fa5");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src += _avl * sizeof(fp16);',
        ),
        # T.mul(buf1[i], buf2[j]) -> vfmul.vf  (j is scalar/outer index)
        (
            re.compile(
                r'float_to_fp16\('
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf1>\w+)\)\[(?P<idx1>[^\]]+)\]\)'
                r'\s*\*\s*'
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf2>\w+)\)\[(?P<idx2>[^\]]+)\]\)'
                r'\)'
            ),
            "mul_vf",
            'fp16 _scalar = ((fp16*){buf2})[{idx2}];\n'
            '    asm volatile("fld fa5, (%0)" ::"r"(&_scalar));\n'
            '    uint32_t _addr_src = (uint32_t)(uintptr_t){buf1} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src));\n'
            '    asm volatile("vfmul.vf v8, v8, fa5");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src += _avl * sizeof(fp16);',
        ),
        # T.div(buf1[i], buf2[j]) -> vfdiv.vf  (j is scalar/outer index)
        (
            re.compile(
                r'float_to_fp16\('
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf1>\w+)\)\[(?P<idx1>[^\]]+)\]\)'
                r'\s*/\s*'
                r'fp16_to_float\(\(\(fp16\*\)(?P<buf2>\w+)\)\[(?P<idx2>[^\]]+)\]\)'
                r'\)'
            ),
            "div_vf",
            'fp16 _scalar = ((fp16*){buf2})[{idx2}];\n'
            '    asm volatile("fld fa5, (%0)" ::"r"(&_scalar));\n'
            '    uint32_t _addr_src = (uint32_t)(uintptr_t){buf1} + _spatz_sid * _vlen * sizeof(fp16);',
            'asm volatile("vle16.v v8,  (%0)" ::"r"(_addr_src));\n'
            '    asm volatile("vfdiv.vf v8, v8, fa5");\n'
            '    asm volatile("vse16.v v8, (%0)" ::"r"(_addr));\n'
            '    _addr_src += _avl * sizeof(fp16);',
        ),
    ]

    def apply(self, bindings: List[TileBinding]) -> List[TileBinding]:
        any_vectorized = False
        for b in bindings:
            is_flat  = b.op_kind == "eltwise" and b.template is TileEltwiseTemplate
            is_inner = b.op_kind == "eltwise" and b.template is TileInnerEltwiseTemplate
            if not (is_flat or is_inner):
                continue
            rep = b.operator_representation
            src_expr = rep.get("src_expr", "")
            loop_var = rep.get("loop_var", "")
            for pattern, spatz_op, spatz_setup, spatz_body in self._RULES:
                m = pattern.search(src_expr)
                if not m:
                    continue
                # Safety: the pattern must match at the start of src_expr.
                # Without this, sub-expression matches occur in expressions like
                # "S_loc[i] / exp(m_old[r] - m[r])" where the exp_sub pattern
                # finds tile_fp16_exp(...) inside the denominator, misidentifying
                # m_old (a 64-element scalar-per-row buffer) as a vector to load
                # from and producing out-of-bounds reads / wrong computation.
                if m.start() != 0:
                    continue
                groups = m.groupdict()
                # Safety: vf rules capture idx2 (scalar buffer index).  Skip
                # the rule when idx2 contains the eltwise loop variable -- that
                # means the "scalar" actually varies per element and cannot be
                # hoisted into a scalar register.
                if loop_var and "idx2" in groups and loop_var in groups.get("idx2", ""):
                    continue

                # Skip vectorization when there are fewer elements than Spatz
                # cores: integer division yields _vlen=0 and the while-loop
                # never executes, silently dropping the operation.
                # ARCH_SPATZ_ATTACED_CORES == 4 at runtime.
                extent = rep.get("extent", 0)
                if isinstance(extent, int) and extent < 4:
                    continue

                if is_flat:
                    # Detect when the flat eltwise is actually an inner loop
                    # serialised from an outer T.Parallel that fell back to a
                    # serial for-loop.  index_expr of the form "r * stride + d"
                    # reveals an outer loop variable r that must be used as a
                    # row offset; without it the Spatz address always starts at
                    # the buffer base (row 0) regardless of r, so every r
                    # iteration overwrites the same row.
                    outer_loop_var, dst_stride = _outer_offset(
                        rep.get("index_expr", loop_var) or loop_var, loop_var
                    )
                    if outer_loop_var is not None:
                        # Switch to SpatzInnerEltwiseTemplate so _addr and
                        # _addr_src get the (outer_loop_var * dst_stride) prefix.
                        row_prefix = (
                            f"({outer_loop_var} * {dst_stride}) * sizeof(fp16) + "
                        )
                        spatz_setup = spatz_setup.replace(
                            "_spatz_sid * _vlen * sizeof(fp16)",
                            row_prefix + "_spatz_sid * _vlen * sizeof(fp16)",
                        )
                        rep["outer_loop_var"] = outer_loop_var
                        rep["dst_stride"]     = dst_stride
                        b.template = SpatzInnerEltwiseTemplate
                        # Keep index_expr — used by the fallback else-branch.
                    else:
                        b.template = SpatzEltwiseTemplate
                        rep.pop("index_expr", None)
                else:
                    # Inject row offset into every _addr_src produced by the rule.
                    # All rules use "_spatz_sid * _vlen * sizeof(fp16)" as the
                    # Spatz-slice offset from the buffer base; prefix it with the
                    # outer-loop row offset so each core operates on its own row.
                    outer_loop_var = rep.get("outer_loop_var", "")
                    dst_stride     = rep.get("dst_stride", 1)
                    row_prefix = (
                        f"({outer_loop_var} * {dst_stride} * ARCH_SPATZ_ATTACED_CORES) * sizeof(fp16) + "
                    )
                    spatz_setup = spatz_setup.replace(
                        "_spatz_sid * _vlen * sizeof(fp16)",
                        row_prefix + "_spatz_sid * _vlen * sizeof(fp16)",
                    )
                    b.template = SpatzInnerEltwiseTemplate
                    # Keep index_expr — used by the fallback else-branch.

                rep["spatz_op"]      = spatz_op
                rep["num_spatz"]     = "ARCH_SPATZ_ATTACED_CORES"
                rep["spatz_setup"]   = spatz_setup.format(**groups)
                rep["spatz_body"]    = spatz_body.format(**groups)
                rep["fallback_expr"] = src_expr
                any_vectorized = True
                break

        if any_vectorized:
            # Prepend the SpatzContext binding so _spatz_attached and _spatz_sid
            # are declared once at function scope, before any loop or eltwise op.
            # Insert after the last group_barrier/sync prefix (GroupInit, GroupContext)
            # so the Spatz variables appear alongside the group rank variables.
            insert_pos = 0
            for i, b in enumerate(bindings):
                if b.op_kind in (TileOpKind.group_barrier, TileOpKind.sync):
                    insert_pos = i + 1
                else:
                    break  # stop at first non-prefix op; later syncs are inside loop body
            ctx_binding = TileBinding(
                op_kind=TileOpKind.block_preamble,
                template=SpatzContextTemplate,
                operator_representation={},
                op_name="spatz_context",
            )
            bindings = list(bindings[:insert_pos]) + [ctx_binding] + list(bindings[insert_pos:])

        return bindings
