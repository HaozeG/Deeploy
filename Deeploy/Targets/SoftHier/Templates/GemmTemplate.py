from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class SoftHier_GemmTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Map Deeploy data-type names to RedMule format enums and byte sizes.
        # Note: float32_t is not natively supported by RedMule; fp16 is used as a
        # fallback because main() converts float32 → fp16 before calling RunNetwork().
        dtype_mapping = {
            'REDMULE_UINT_16': 'uint16_t*',
            'REDMULE_INT_16':  'int16_t*',
            'REDMULE_FP_16':   'fp16*',
            'REDMULE_UINT_8':  'uint8_t*',
            'REDMULE_INT_8':   'int8_t*',
            'REDMULE_FP_8':    'fp8_t*'
        }
        redmule_dtype_mapping = {
            'uint16_t*': 'REDMULE_UINT_16',
            'int16_t*':  'REDMULE_INT_16',
            'float16_t*': 'REDMULE_FP_16',
            'uint8_t*':  'REDMULE_UINT_8',
            'int8_t*':   'REDMULE_INT_8',
            'fp8_t*':    'REDMULE_FP_8',
            'float32_t*': 'REDMULE_FP_16',  # converted to fp16 in main() before RunNetwork()
        }
        redmule_dtype_size_mapping = {
            'REDMULE_UINT_16': 2,
            'REDMULE_INT_16':  2,
            'REDMULE_FP_16':   2,
            'REDMULE_UINT_8':  1,
            'REDMULE_INT_8':   1,
            'REDMULE_FP_8':    1,
        }

        try:
            operatorRepresentation['redmule_dtype'] = redmule_dtype_mapping[operatorRepresentation['data_out_type'].typeName]
            operatorRepresentation['dtype'] = dtype_mapping[operatorRepresentation['redmule_dtype']]
            operatorRepresentation['dtype_size'] = redmule_dtype_size_mapping[operatorRepresentation['redmule_dtype']]
        except KeyError:
            raise ValueError(
                f"Data type {operatorRepresentation['data_out_type'].typeName} not supported by RedMule! "
                f"Supported types: {list(redmule_dtype_mapping.keys())}")

        return ctxt, operatorRepresentation, []

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        """Keep GEMM operands in their original memory level.

        SoftHier GEMM expects A/B/C/Y pointers to be HBM-resident and performs
        tile movement internally. Do not hoist operand buffers into L1 here.
        """

        operatorRepresentation['ctxtBuffer_A'] = operatorRepresentation['A']
        operatorRepresentation['ctxtBuffer_B'] = operatorRepresentation['B']
        operatorRepresentation['ctxtBuffer_C'] = operatorRepresentation['C']

        return ctxt, operatorRepresentation, []


# Template string for SoftHier GEMM.
#
# Assumptions:
#   - A/B/C/Y pointers passed to gemm() are HBM-resident.
#   - gemm() performs HBM ↔ L1 tile movement internally.
#   - float32 models use fp16 data at runtime for RedMule, after conversion in
#     main() before RunNetwork().
#
# All runtime primitives (flex_is_dm_core, flex_intra_cluster_sync) match those used in the
# verified TargetLibraries/SoftHier/src/Gemm.c.
SoftHierGemmTemplateStr = r"""
// GEMM (Name: ${nodeName}, Op: ${nodeOp})
// Data is assumed to be in fp16 in HBM (float32->fp16 conversion done once in main()).

// Set up typed pointers for the resolved (L1 or HBM) operand addresses
${dtype} ref_${nodeName}_A   = (${dtype})(${ctxtBuffer_A});
${dtype} ref_${nodeName}_B   = (${dtype})(${ctxtBuffer_B});
${dtype} ref_${nodeName}_C   = (${dtype})(${ctxtBuffer_C});
${dtype} ref_${nodeName}_Y = (${dtype})(${data_out});

// Check for pointer addresses
#ifdef LOG_ENABLE
if (flex_is_dm_core()) {
    printf("Address of A: 0x%08x, B: 0x%08x, C: 0x%08x, Output: 0x%08x\n",
           (uint32_t)(uintptr_t)ref_${nodeName}_A,
           (uint32_t)(uintptr_t)ref_${nodeName}_B,
           (uint32_t)(uintptr_t)ref_${nodeName}_C,
           (uint32_t)(uintptr_t)ref_${nodeName}_Y);
}
#endif

// Execute tiled GEMM via RedMule (tile dimensions == matrix dimensions → single tile)
gemm(${data_out_shape[0]}, ${data_out_shape[1]}, ${A_shape[1]},
     ${data_out_shape[0]}, ${data_out_shape[1]}, ${A_shape[1]},
     (uint64_t)(uintptr_t)ref_${nodeName}_A,
     (uint64_t)(uintptr_t)ref_${nodeName}_B,
     (uint64_t)(uintptr_t)ref_${nodeName}_C,
     (uint64_t)(uintptr_t)ref_${nodeName}_Y,
     ${redmule_dtype}, ${dtype_size});
// test_group_barrier();
// test_dma();
// test_dma_collectives();
// test_FP16();
// test_redmule();
// test_spatz();
"""

SoftHierGemm_Template = SoftHier_GemmTemplate(SoftHierGemmTemplateStr)
