# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Deeploy-native collective frontend for TileLang kernels.

Defines ``D.reduce`` and ``D.broadcast``: two strict, scope-explicit collective
ops that emit ``tir.call_intrin("tl.deeploy.reduce")`` /
``tir.call_intrin("tl.deeploy.broadcast")`` nodes.  The Deeploy visitor
recognises these and lowers them via ``CollectiveLoweringPass`` and
``CollectiveStrategies``.

Design rules
------------
* All parameters except *buffer* are keyword-only — no positional ambiguity.
* ``level`` and ``group`` are always required.
* ``axis`` is required on ``level="inter_group"`` (no implicit "all instances").
* ``axis`` is required on ``level="intra_group"`` when the group has two axes.
* ``root`` is required for ``broadcast``; optional for ``reduce`` (None = allreduce).
* Validation is performed at call site; axis-name checks against the registry
  happen again in the visitor handler where the registry is available.

Intrinsic argument layout
-------------------------
tl.deeploy.reduce:
  args[0] : region(buffer)   (access="rw")
  args[1] : StringImm(op)    e.g. "sum"
  args[2] : StringImm(level) "intra_group" | "inter_group"
  args[3] : StringImm(axis)  axis name or ""
  args[4] : StringImm(group) cluster-group name
  args[5] : StringImm(root)  "" = allreduce, else C-expression string

tl.deeploy.broadcast:
  args[0] : region(buffer)   (access="rw")
  args[1] : StringImm(level)
  args[2] : StringImm(axis)
  args[3] : StringImm(group)
  args[4] : StringImm(root)  C-expression string (required)

Usage example
-------------
::

    from Deeploy.TileIR.Frontend import tl_deeploy as D

    D.broadcast(A_local, level="intra_group", axis="x", group="summa", root=gy)
    D.broadcast(B_local, level="intra_group", axis="y", group="summa", root=gx)
    D.reduce(C_local, op="sum", level="inter_group", axis="k", group="summa")
"""

from __future__ import annotations

from typing import Optional, Union

try:
    import tvm
    import tvm.ir
    import tvm.tir as tir
    from tilelang.language.frame import get_let_value, has_let_value
    from tilelang.utils.language import get_buffer_region_from_load, to_buffer_region

    # Register Deeploy ops in the TVM op registry (idempotent after first import).
    for _op_name in ("tl.deeploy.reduce", "tl.deeploy.broadcast"):
        try:
            tvm.ir.Op.get(_op_name)
        except Exception:
            tvm.ir.register_op_attr(_op_name, "TCallEffectKind", tir.CallEffectKind.Opaque)

    _TVM_AVAILABLE = True
except ImportError:
    _TVM_AVAILABLE = False
    tir = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_OPS = ("sum", "max", "min", "prod")
VALID_LEVELS = ("intra_group", "inter_group")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_region(buf, access_type: str = "rw"):
    if isinstance(buf, tir.Var) and has_let_value(buf):
        buf = get_let_value(buf)

    if isinstance(buf, tir.Buffer):
        extents = list(buf.shape)
    elif isinstance(buf, tir.BufferRegion):
        extents = [r.extent for r in buf.region]
    elif isinstance(buf, tir.BufferLoad):
        region = get_buffer_region_from_load(buf)
        extents = ([r.extent for r in region.region]
                   if region is not None else [tir.IntImm("int32", 1) for _ in buf.indices])
    else:
        extents = []

    return to_buffer_region(buf, access_type=access_type, extents=extents)


def _root_to_str(root) -> str:
    """Convert a root argument to a C-expression string."""
    if root is None:
        return ""
    if isinstance(root, int):
        return str(root)
    # tir.PrimExpr, tir.Var, or any other TIR node: stringify via TVM printer.
    try:
        return str(int(root))
    except (TypeError, ValueError):
        pass
    return str(root)


# ---------------------------------------------------------------------------
# Frontend validation (call-site checks; registry checks happen in visitor)
# ---------------------------------------------------------------------------

def _check_reduce_args(op: str, level: str, axis: Optional[str], group: str) -> None:
    if op not in VALID_OPS:
        raise ValueError(
            f"D.reduce: op={op!r} is not a valid reduction operator. "
            f"Choose from {VALID_OPS}."
        )
    if level not in VALID_LEVELS:
        raise ValueError(
            f"D.reduce: level={level!r} is not valid. "
            f"Choose from {VALID_LEVELS}."
        )
    if not group:
        raise ValueError("D.reduce: group is required and must be a non-empty string.")


def _check_broadcast_args(level: str, axis: Optional[str], group: str, root) -> None:
    if level not in VALID_LEVELS:
        raise ValueError(
            f"D.broadcast: level={level!r} is not valid. "
            f"Choose from {VALID_LEVELS}."
        )
    if not group:
        raise ValueError("D.broadcast: group is required and must be a non-empty string.")
    if root is None:
        raise ValueError(
            "D.broadcast: root is required. "
            "Provide a static int or a TIR/C expression identifying the source rank."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reduce(
    buffer,
    *,
    op: str,
    level: str,
    axis: Optional[str] = None,
    group: str,
    root: Optional[Union[int, "tir.PrimExpr"]] = None,
) -> "tir.PrimExpr":
    """In-place collective reduce over *buffer* across *level* scope.

    Parameters
    ----------
    buffer : Buffer | BufferRegion | BufferLoad
        Operand, updated in place.  Every participant receives the result
        when *root* is ``None`` (allreduce); otherwise only the root rank
        holds a defined value after the call.
    op : str
        Reduction operator: ``"sum"`` | ``"max"`` | ``"min"`` | ``"prod"``.
    level : str
        ``"intra_group"`` — across the ``group_x × group_y`` members of one
        instance, along *axis* (row/col DMA mask).
        ``"inter_group"`` — across all ``num_groups`` instances, along the
        meta-grid axis *axis* (inverted mask + global barrier).
    axis : str or None
        Name of the axis within *level*.

        * ``level="intra_group"``: one of ``ClusterGroup.axis_names``.
          Required when the group has two axes; omit only for a 1-D group.
        * ``level="inter_group"``: one of ``ClusterGroup.meta_axes``.
          Always required (no implicit "all instances").
    group : str
        Cluster-group name declared via ``T.cluster_group(...)``.  Required.
    root : int | tir.PrimExpr | None
        ``None`` (default) → allreduce: every participant ends with the result.
        static int or TIR expression → reduce-to-root; other ranks undefined.
    """
    _check_reduce_args(op, level, axis, group)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.deeploy.reduce"),
        _to_region(buffer, access_type="rw"),
        tir.StringImm(op),
        tir.StringImm(level),
        tir.StringImm(axis or ""),
        tir.StringImm(group),
        tir.StringImm(_root_to_str(root)),
    )


def broadcast(
    buffer,
    *,
    level: str,
    axis: Optional[str] = None,
    group: str,
    root: Union[int, "tir.PrimExpr"],
) -> "tir.PrimExpr":
    """In-place collective broadcast over *buffer* from *root* to all participants.

    Parameters
    ----------
    buffer : Buffer | BufferRegion | BufferLoad
        Operand, updated in place.  Every participant ends up with the value
        held by *root* before the call.
    level : str
        ``"intra_group"`` or ``"inter_group"`` (see ``D.reduce`` docs).
    axis : str or None
        Axis within the level (same rules as ``D.reduce``).
    group : str
        Cluster-group name.  Required.
    root : int | tir.PrimExpr
        Source rank within the participating slice.

        * Static ``int`` — fixed rank along *axis*.
        * ``tir.PrimExpr`` / TIR ``Var`` — dynamic rank, e.g. the loop
          variable ``gid_y`` that varies per cluster in SUMMA's A-broadcast.
    """
    _check_broadcast_args(level, axis, group, root)
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.deeploy.broadcast"),
        _to_region(buffer, access_type="rw"),
        tir.StringImm(level),
        tir.StringImm(axis or ""),
        tir.StringImm(group),
        tir.StringImm(_root_to_str(root)),
    )
