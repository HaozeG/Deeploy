/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_GEMMFP32_KERNEL_HEADER_
#define __DEEPLOY_MATH_GEMMFP32_KERNEL_HEADER_

#include "DeeploySoftHierMath.h"

// TODO: transpose support
/*
 *
 * GEMM with the following format:
 * A is an M x K matrix, B is a K x N matrix, and C is a M x N matrix
 *
 * A' = transpose(A) if transA else A
 * B' = transpose(B) if transB else B
 *
 * Y =  A' * B' + C
 *
 */
void gemm(uint32_t dim_M, uint32_t dim_N, uint32_t dim_K,
    uint16_t dim_tile_M, uint16_t dim_tile_N, uint16_t dim_tile_K,
    uint64_t A_addr, uint64_t B_addr, uint64_t C_addr, uint64_t Y_addr, 
    redmule_compute_format_t REDMULE_OP, uint32_t DATA_TYPE_SIZE);

void summa_gemm(uint64_t                    X_address,
    uint64_t                    W_address,
    uint64_t                    Z_address,
    uint32_t                    M_size,
    uint32_t                    N_size,
    uint32_t                    K_size, /*shared dimension*/
    uint32_t                    M_tile,
    uint32_t                    N_tile,
    uint32_t                    K_tile,
    uint32_t                    group_x,
    uint32_t                    group_y,
    uint32_t                    num_group,
    uint32_t                    group_reduction,
    uint32_t                    group_splitK,
    uint32_t                    group_splitN,
    uint32_t                    X_address_group_gap,
    uint32_t                    W_address_group_gap,
    uint32_t                    Z_address_group_gap,
    // data type related parameters
    uint32_t                   DATA_TYPE_BYTE);

#endif //__DEEPLOY_MATH_GEMMFP32_KERNEL_HEADER_