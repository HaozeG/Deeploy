#include "DeeploySoftHierMath.h"
#include "Redmule_Gemm_fp32.h"

void gemm_fp32_transB_opt(uint32_t M, uint32_t N, uint32_t K, float32_t *A,
                          uint32_t ldA, float32_t *B, uint32_t ldB,
                          float32_t *C, uint32_t ldC, float32_t *Y,
                          uint32_t BETA, uint32_t setup_SSR) {

// TODO: hardcode for now


}