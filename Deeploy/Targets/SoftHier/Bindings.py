from functools import partial

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration, MemoryAwareClosureGeneration
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.CommonExtensions.DataTypes import int8_t, uint8_t, float16_t, int16_t, uint16_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.Templates import iNoNormTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, GEMMChecker, RQAddChecker, SoftmaxChecker, iNoNormChecker
from Deeploy.Targets.SoftHier.Templates.GemmTemplate import SoftHierGemm_Template

# TODO: the bindings and templates not completed yet, just placeholders for now to be able to test the mapping and code generation flow
BasicTransformer = CodeTransformation(
    [ArgumentStructGeneration(),
     MemoryManagementGeneration(),
     FutureGeneration()])

TiledTransformer = CodeTransformation([
    ArgumentStructGeneration(),
    MemoryManagementGeneration(),
    FutureGeneration()
])

# TODO: type checker not completed yet
SoftHierGemmBindings = [
    NodeBinding(GEMMChecker([PointerClass(_type), PointerClass(_type), PointerClass(_type)], [PointerClass(_type)]), SoftHierGemm_Template, TiledTransformer) for _type in [int8_t, uint8_t, int16_t, uint16_t, float16_t]
]