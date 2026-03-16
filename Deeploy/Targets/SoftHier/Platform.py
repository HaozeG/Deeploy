# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer, ONNXLayer
from ignore.TileLangNodeMapper import TileLangMapper
from Deeploy.Targets.Generic.Bindings import BasicAddBindings

from Deeploy.Targets.Generic.Layers import AddLayer, GEMMLayer
from Deeploy.Targets.Generic.Parsers import AddParser
from Deeploy.Targets.SoftHier.Parsers import SoftHierGEMMParser
from Deeploy.Targets.SoftHier.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.SoftHier.Templates.AllocateTemplate import SoftHierTransientInitTemplate, SoftHierTransientAllocateTemplate, \
    SoftHierDynamicInitTemplate, SoftHierDynamicAllocTemplate
from Deeploy.Targets.SoftHier.Templates.FreeTemplate import SoftHierDynamicFreeTemplate
from Deeploy.Targets.SoftHier.Bindings import SoftHierGemmBindings
# Basic bindings
Add_Mapper = NodeMapper(AddParser(), BasicAddBindings)
Gemm_Mapper = NodeMapper(SoftHierGEMMParser(), SoftHierGemmBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
# DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

SoftHierlMapping = {
    # 'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    # 'Gather': GatherLayer([GatherMapper]),
    # 'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    # 'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    # 'MatMul': MatMulLayer([MatMulMapper]),
    'Gemm': GEMMLayer([Gemm_Mapper]),
    # 'RQGemm': RQGEMMLayer([RqGemmMapper]),
    # 'iSoftmax': SoftmaxLayer([iSoftmaxMapper]),
    # 'Softmax': SoftmaxLayer([SoftmaxMapper]),
    # 'iNoNorm': iNoNormLayer([iNoNormMapper]),
    # 'iLayerNorm': LayerNormLayer([iLayerNormMapper]),
    # 'RequantizedAdd': AddLayer([RQAddMapper]),
    'Add': AddLayer([Add_Mapper]),
}

# ---------------------------------------------------------------------------
# Legacy buffer classes (kept for ONNX-path backward compatibility)
# ---------------------------------------------------------------------------

class SoftHierVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.SoftHierInitTemplate
    allocTemplate = AllocateTemplate.SoftHierAllocateTemplate
    deallocTemplate = FreeTemplate.SoftHierGlobalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": memoryLevel
        }


class SoftHierTransientBuffer(TransientBuffer):

    initTemplate = SoftHierTransientInitTemplate
    allocTemplate = SoftHierTransientAllocateTemplate
    deallocTemplate = FreeTemplate.SoftHierLocalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {"type": self._type, "name": self.name, "size": self.size, "_memoryLevel": memoryLevel}



class SoftHierConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.SoftHierGlobalInitTemplate
    allocTemplate = AllocateTemplate.SoftHierGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.SoftHierGlobalTemplate

    def _bufferRepresentation(self):
        operatorRepresentation = super()._bufferRepresentation()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        operatorRepresentation["_memoryLevel"] = memoryLevel

        return operatorRepresentation


class SoftHierStructBuffer(StructBuffer):

    initTemplate = AllocateTemplate.SoftHierStructInitTemplate
    allocTemplate = AllocateTemplate.SoftHierStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


# ---------------------------------------------------------------------------
# Unified DynamicBuffer for the TileLang → SoftHier path
#
# A single buffer class that selects L1 or HBM allocation/deallocation at
# code-generation time based on the `_memoryLevel` attribute:
#   'L1'  → flex_l1_malloc / flex_l1_free     (fragment / shared buffers)
#   'HBM' → flex_hbm_malloc / flex_hbm_free   (PrimFunc I/O parameters)
#
# The cluster_id attribute (optional, default None) is set by the
# TilelangVisitor from TileLang kernel metadata.  When set, the generated
# alloc/compute code is wrapped in `if (CID == cluster_id) { ... }`.
# ---------------------------------------------------------------------------

class SoftHierDynamicBuffer(VariableBuffer):
    """Unified SoftHier buffer for the TileLang compilation path.

    Attributes
    ----------
    _memoryLevel : str
        'L1' or 'HBM'.  Controls which allocator is used.  Set by
        TilelangVisitor based on TVM buffer scope.
    cluster_id : Optional[int]
        If not None, alloc/compute snippets for this buffer are wrapped in
        ``if (CID == cluster_id) { ... }`` to target a specific cluster.
        Set from the TileLang kernel's `cluster_id` attribute.
    """

    initTemplate    = SoftHierDynamicInitTemplate
    allocTemplate   = SoftHierDynamicAllocTemplate
    deallocTemplate = SoftHierDynamicFreeTemplate

    def __init__(self, name: str = '', shape=None, aliases=None,
                 memory_level: str = 'HBM', cluster_id=None):
        super().__init__(name, shape if shape is not None else [1], aliases)
        self._memoryLevel: str = memory_level  # 'L1' or 'HBM'
        self.cluster_id = cluster_id           # None or int cluster index

    def _bufferRepresentation(self):
        return {
            "type":          self._instance,
            "name":          self.name,
            "size":          int(np.prod(self.shape)),
            "_memoryLevel":  self._memoryLevel,
            "cluster_id":    self.cluster_id,
        }

from Deeploy.DeeployTypes import TopologyOptimizer

SoftHierOptimizer = TopologyOptimizer([], name="SoftHierOptimizer")
includeList = ["flex_alloc_api.h", "flex_runtime_api.h", "flex_redmule_api.h", "flex_dma_api.h", "flex_types.h", "flex_printf_api.h","DeeploySoftHierMath.h"]


class SoftHierEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = SoftHierlMapping, initCode: str = "", includeList = includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SoftHierPlatform(DeploymentPlatform):
    """SoftHier deployment platform.

    For the TileLang path, all four buffer type arguments accept
    `SoftHierDynamicBuffer` — a single class whose `_memoryLevel` attribute
    ('L1' / 'HBM') drives the correct allocator selection at code-gen time.
    """

    def __init__(self,
                 engines = [SoftHierEngine("SoftHier")],
                 variableBuffer = SoftHierVariableBuffer,
                 constantBuffer = SoftHierConstantBuffer,
                 structBuffer = SoftHierStructBuffer,
                 transientBuffer = SoftHierTransientBuffer):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
