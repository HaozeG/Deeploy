/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "flex_cluster_arch.h"
#include "flex_alloc.h"
#include "flex_runtime.h"
#include "flex_dma_pattern.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_group_barrier.h"
#include "flex_libfp16.h"

// float32_t is defined in Generic types.h
typedef float float32_t;

// Deeploy-generated
#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"

int main() {
	uint32_t eoc_val = 0;
	flex_barrier_xy_init();
	flex_global_barrier_xy();
	flex_alloc_init();
	flex_intra_cluster_sync();
	flex_global_barrier_xy();
	flex_intra_cluster_sync();
	/**************************************/
	/*  Program Execution Region -- Start */
	/**************************************/
	uint32_t CID = flex_get_cluster_id(); // Get cluster ID
	uint32_t core_id = flex_get_core_id();

	if (flex_is_first_core() && CID == 0) { // only allow core 0 in cluster 0 to print
		printf("[main.c] >>> Initializing network...\n\n");
	}

	flex_global_barrier_xy();

	InitNetwork(core_id, ARCH_NUM_CORE_PER_CLUSTER);

	flex_global_barrier_xy(); // Ensure InitNetwork completes before RunNetwork

	if (CID == 0) {
		// For non-float32 inputs: assume input datatype is supported by SoftHier components
		if (!ISFLOAT32) {
			if (flex_is_dm_core()) { // allow dm core to init network and dma
				for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
					printf("[main.c] >>> DMAing input buffer from original address 0x%08x to network input buffer at address 0x%08x...\n\n", (uint32_t)(uintptr_t)testInputVector[buf], (uint32_t)(uintptr_t)DeeployNetwork_inputs[buf]);
					// original data in HBM (placed by loader)
					void *ori_addr = testInputVector[buf];

					if ((uint64_t)DeeployNetwork_inputs[buf] <
						(uint64_t)ARCH_HBM_START_BASE) {
					// Trigger DMA transaction: move from HBM to L1
					uint64_t mask = 0x00000000ffffffff;
					uint64_t masked_addr = (uint64_t)ori_addr & mask;
					flex_dma_async_1d(DeeployNetwork_inputs[buf], masked_addr,
										DeeployNetwork_inputs_bytes[buf]);
					// Wait all DMA transaction done
					flex_dma_async_wait_all();
					} else {
					uint64_t *dst_addr = DeeployNetwork_inputs[buf];
					// perform mem_copy with a single core
					for (uint32_t i = 0; i < (DeeployNetwork_inputs_bytes[buf] + 7) / 8;
							i++) {
						uint64_t data = ((uint64_t *)ori_addr)[i];
						dst_addr[i] = data;
					}
					}
				}
			}
			flex_intra_cluster_sync(); // Cluster barrier
			// check first few elements after DMA
			if (flex_is_first_core()) {
				for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
					for (uint32_t i = 0; i < 4; i++) {
						OUTPUTTYPE val = ((OUTPUTTYPE *)DeeployNetwork_inputs[buf])[i];
						printf("[main.c] >>> After DMA, first few elements of input buffer %lu: %f\r\n", buf, val);
					}
				}
			}
		}

		// For float32_t inputs: convert float32 source → fp16 into the network
		// input buffer.  Mirrors the non-fp32 path: if the source data is already
		// in L1 (low address), convert directly; if it is in HBM, DMA it into a
		// temporary L1 buffer first, then convert.
		if (ISFLOAT32) {
			for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
			// DeeployNetwork_inputs_bytes is the fp16 buffer size (2 bytes/element)
			uint32_t num_elements = DeeployNetwork_inputs_bytes[buf] / sizeof(fp16);
			float32_t *src_f32 = (float32_t *)testInputVector[buf];
			fp16      *dst     = (fp16 *)DeeployNetwork_inputs[buf];

			if ((uint64_t)(uintptr_t)src_f32 < (uint64_t)ARCH_HBM_START_BASE) {
				// Source is already in L1 – convert directly, no DMA needed
				if (flex_is_first_core()) {
					for (uint32_t i = 0; i < num_elements; i++) {
						dst[i] = float_to_fp16(src_f32[i]);
					}
				}
				flex_intra_cluster_sync();
			} else {
				// Source is in HBM – DMA float32 data into a temp L1 buffer first
				static volatile uint32_t tmp_l1_addr = 0;

				if (flex_is_first_core()) {
				tmp_l1_addr = (uint32_t)(uintptr_t)flex_l1_malloc(num_elements * sizeof(float32_t));
				printf("[main.c] >>> Allocated temporary L1 buffer at address 0x%08x for DMA and conversion of input buffer %lu...\n\n", tmp_l1_addr, buf);
				}
				flex_intra_cluster_sync();

				if (flex_is_dm_core()) {
				printf("[main.c] >>> Source HBM address: 0x%08x, Temporary L1 address: 0x%08x\n\n", (uint32_t)(uintptr_t)src_f32, tmp_l1_addr);
				uint64_t mask    = 0x00000000ffffffff;
				uint64_t src_hbm = (uint64_t)(uintptr_t)src_f32 & mask;
				flex_dma_async_1d((void *)(uintptr_t)tmp_l1_addr, src_hbm,
									num_elements * sizeof(float32_t));
				flex_dma_async_wait_all();
				}
				flex_intra_cluster_sync();

				if (flex_is_dm_core()) {
				printf("[main.c] >>> Converting input buffer %lu from FP32 to FP16 in L1...\n\n", buf);
				float32_t *src_l1 = (float32_t *)(uintptr_t)tmp_l1_addr;
				for (uint32_t i = 0; i < num_elements; i++) {
					// printf("Converting element %lu: %f to %f\n", i, src_l1[i], fp16_to_float(float_to_fp16(src_l1[i])));
					// printf("dst addr: 0x%08x\n", (uint32_t)(uintptr_t)&dst[i]);
					dst[i] = float_to_fp16(src_l1[i]);
				}
				flex_l1_free((void *)(uintptr_t)tmp_l1_addr);
				}
				flex_intra_cluster_sync();
			}
		}
		flex_intra_cluster_sync();
		if (flex_is_first_core()) {
			for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
				for (uint32_t i = 0; i < 4; i++) {
					float val = fp16_to_float(((fp16 *)DeeployNetwork_inputs[buf])[i]);
					printf("[main.c] >>> After DMA, first few elements of input buffer %lu: %f\r\n", buf, val);
				}
			}
		}
		}
		flex_intra_cluster_sync(); // Cluster barrier

		if (flex_is_first_core()) { // allow core 0 to compute
			printf("[main.c] >>> Running network...\n\n");
		}
		flex_intra_cluster_sync(); // Cluster barrier
	}

	flex_global_barrier_xy(); // Ensure InitNetwork completes before RunNetwork
	
	RunNetwork(core_id, ARCH_NUM_CORE_PER_CLUSTER);
	
	flex_global_barrier_xy(); 

if (CID == 0) { // only allow cluster 0 to work
	flex_intra_cluster_sync(); // Cluster barrier  
	// verification
	if (flex_is_first_core()) { 
		printf("[main.c] >>> Verifying outputs...\n\n");
	}
	
	int32_t tot_err = 0;
	uint32_t tot = 0;
	float diff;
	float expected, actual;

	if (flex_is_first_core()) {
	  for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
		printf("[main.c] >>> Verifying output buffer %lu...\n\n", buf);
		tot += DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE);
		for (uint32_t i = 0;
			 i < DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE); i++) {
		  // for float32_t, converted to fp16 during computation
		  // use customized functions to interpret the bits and compute the difference
		  if (ISFLOAT32) {
			expected = ((float32_t *)testOutputVector[buf])[i];
			actual = fp16_to_float(((fp16 *)DeeployNetwork_outputs[buf])[i]);
			diff = expected - actual;
			if (diff > 0.01f || diff < -0.01f) { // use a threshold for float comparison
			  tot_err += 1;
			  printf("Expected: %f  ", expected);
			  printf("Actual: %f  ", actual);
			  printf("Diff: %f at Index %12lu in Output %lu\r\n", diff, i, buf);
			}
		  } else {
			expected = fp16_to_float(((fp16 *)testOutputVector[buf])[i]);
			actual = fp16_to_float(((fp16 *)DeeployNetwork_outputs[buf])[i]);
			diff = expected - actual;
			if (diff > 0.01f || diff < -0.01f) { // use a threshold for non-float comparison as well, just in case	
			  tot_err += 1;
			  printf("Expected: %4f  ", expected);
			  printf("Actual: %4f  ", actual);
			  printf("Diff: %4f at Index %12lu in Output %u\r\n", diff, i, buf);
			}
		  }
		}
	  }
	  printf("Errors: %ld out of %ld \r\n", tot_err, tot);
	}
	flex_intra_cluster_sync(); // Cluster barrier
  }

  /**************************************/
  /*  Program Execution Region -- Stop  */
  /**************************************/
  flex_global_barrier_xy();
  flex_eoc(eoc_val);
  return 0;
}