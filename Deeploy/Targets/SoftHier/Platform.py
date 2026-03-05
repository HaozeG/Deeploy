# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicAddBindings

from Deeploy.Targets.Generic.Layers import AddLayer, GEMMLayer
from Deeploy.Targets.Generic.Parsers import AddParser
from Deeploy.Targets.SoftHier.Parsers import SoftHierGEMMParser
from Deeploy.Targets.SoftHier.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.SoftHier.Templates.AllocateTemplate import SoftHierTransientInitTemplate, SoftHierTransientAllocateTemplate
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
    'Add': AddLayer([Add_Mapper])
}

# TODO: check all buffer's init, alloc, dealloc implementations
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


SoftHierOptimizer = TopologyOptimizer([], name = "SoftHierOptimizer")
includeList = ["flex_alloc_api.h", "flex_runtime_api.h", "flex_redmule_api.h", "flex_dma_api.h", "flex_types.h", "flex_printf_api.h","DeeploySoftHierMath.h"]


class SoftHierEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = SoftHierlMapping, initCode: str = "", includeList = includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SoftHierPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [SoftHierEngine("SoftHier")],
                 variableBuffer = SoftHierVariableBuffer,
                 constantBuffer = SoftHierConstantBuffer,
                 structBuffer = SoftHierStructBuffer,
                 transientBuffer = SoftHierTransientBuffer):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
