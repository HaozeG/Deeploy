#ifndef HELLO_WORLD_H_
#define HELLO_WORLD_H_

#include "DeeploySoftHierMath.h"

void hello_world_core0(void);
void hello_world_all_cluster(void);
void hello_world_all_core(void);
void test_global_barrier(void);
void test_global_barrier_polling(void);
void compare_hw_sync_and_polling_sync(void);
void test_group_barrier(void);
void test_dma(void);
void test_redmule(void);
void check_hbm_preload(void);
void zero_mem_test(void);
void test_synchronization(void);
void test_dma_2d(void);
void test_dma_collectives(void);
void test_HBM_interleaving(void);
void test_FP16(void);
void test_malloc_single_cluster(void);
void test_nm_config(void);
void test_spatz(void);
#endif //HELLO_WORLD_H_