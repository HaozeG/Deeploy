/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_SOFTHIER_MATH_HEADER_
#define __DEEPLOY_SOFTHIER_MATH_HEADER_

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "types.h"

#include "DeeployBasicMath.h"

#include "flex_cluster_arch.h"
#include "flex_dma_api.h"
#include "flex_redmule_api.h"
#include "flex_group_barrier_api.h"
#include "flex_types.h"

#define LOG_ENABLE
#include "kernel/Gemm.h"
#include "kernel/hello_world.h"

#endif // __DEEPLOY_SOFTHIER_MATH_HEADER_
