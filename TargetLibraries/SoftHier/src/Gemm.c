#include "Gemm.h"

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
    redmule_compute_format_t REDMULE_OP, uint32_t DATA_TYPE_SIZE) {                                
    uint32_t CID = flex_get_cluster_id();//Get cluster ID

#ifdef LOG_ENABLE
    if (CID == 0) {
        if (flex_is_dm_core()) {
            printf("Initializing RedMule for GEMM with tile_M: %d, tile_N: %d, tile_K: %d\n", dim_tile_M, dim_tile_N, dim_tile_K);
        }
        flex_intra_cluster_sync();//Cluster barrier
    }
#endif


    //Initialize RedMule Paramters
    if ((CID == 0)) {
        if (flex_is_first_core())
        {
            //Configure M-N-K of tile that RedMule will accelerate
            flex_redmule_config(dim_tile_M, dim_tile_N, dim_tile_K);
        }
        if (flex_is_dm_core()) {
            // check for input addresses
            printf("[gemm.c] >>> Address of A: 0x%08x, B: 0x%08x, C: 0x%08x, Output: 0x%08x\n", (uint32_t)(uintptr_t)A_addr, (uint32_t)(uintptr_t)B_addr,
                   (uint32_t)(uintptr_t)C_addr, (uint32_t)(uintptr_t)Y_addr);
        }
        flex_intra_cluster_sync();//Cluster barrier
    }
#ifdef LOG_ENABLE
    if (CID == 0) { // only allow cluster 0 to work
        // Initialize L1 buffer offsets for double buffering
        static volatile uint32_t ptr_CY_L1, ptr_A_L1_1, ptr_B_L1_1, ptr_A_L1_2, ptr_B_L1_2;
        if (flex_is_first_core()) {
            ptr_CY_L1 = (uint32_t)flex_l1_malloc(dim_tile_M * dim_tile_N * sizeof(fp16));
            ptr_A_L1_1 = (uint32_t)flex_l1_malloc(dim_tile_M * dim_tile_K * sizeof(fp16));
            ptr_B_L1_1 = (uint32_t)flex_l1_malloc(dim_tile_K * dim_tile_N * sizeof(fp16));
            ptr_A_L1_2 = (uint32_t)flex_l1_malloc(dim_tile_M * dim_tile_K * sizeof(fp16));
            ptr_B_L1_2 = (uint32_t)flex_l1_malloc(dim_tile_K * dim_tile_N * sizeof(fp16));
        }
        flex_intra_cluster_sync();//Cluster barrier
        // Calculate number of tiles per dimension, assuming dimensions are perfectly divisible by tile sizes
        uint32_t TILES_PER_M = dim_M / dim_tile_M;
        uint32_t TILES_PER_N = dim_N / dim_tile_N;
        uint32_t TILES_PER_K = dim_K / dim_tile_K;
        //Iterate over every Z tiles
        for (int row = 0; row < TILES_PER_M; ++row) {
            for (int col = 0; col < TILES_PER_N; ++col) {
                //Start address of tiles
                uint32_t A_hbm_addr = (uint32_t)A_addr + row * dim_tile_M * dim_K * sizeof(fp16);
                uint32_t B_hbm_addr = (uint32_t)B_addr + col * dim_tile_N * sizeof(fp16);
                uint32_t C_hbm_addr = (uint32_t)C_addr + row * dim_tile_M * dim_N * sizeof(fp16) + col * dim_tile_N * sizeof(fp16);
                uint32_t Y_hbm_addr = (uint32_t)Y_addr + row * dim_tile_M * dim_N * sizeof(fp16) + col * dim_tile_N * sizeof(fp16);
                
                //Preload C,A,B tiles
                if (flex_is_dm_core()) {//Use DM core to trigger DMA transcations
                    //Trigger DMA transaction: move C tile from HBM to L1
                    flex_dma_async_1d((ptr_CY_L1),(C_hbm_addr), DATA_TYPE_SIZE * dim_tile_M * dim_tile_N);
                    
                    //Trigger DMA transaction: move A tile from HBM to L1
                    flex_dma_async_1d((ptr_A_L1_1),(A_hbm_addr), DATA_TYPE_SIZE * dim_tile_M * dim_tile_K);
                    
                    //Trigger DMA transaction: move B tile from HBM to L1
                    flex_dma_async_1d((ptr_B_L1_1),(B_hbm_addr), DATA_TYPE_SIZE * dim_tile_K * dim_tile_N);
                    
                    //Wait all DMA transaction done
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();//Cluster barrier

                //Compute A and B to YZ in L1 + preload next A and B
                int i = 0;
                for (; i < (TILES_PER_K-1); ++i)
                {
                    //Calculate next A and B tile address in HBM
                    A_hbm_addr += DATA_TYPE_SIZE * dim_tile_M * dim_tile_K;
                    B_hbm_addr += TILES_PER_N * DATA_TYPE_SIZE * dim_tile_K * dim_tile_N;
                    
                    //Ping-Pong operations on Double-Buffering A and B
                    if (i%2 == 0)
                    {
                        if (flex_is_dm_core()){//Use DM core to trigger DMA transcations
                        //Trigger DMA transaction: move A tile from HBM to L1
                                flex_dma_async_1d(ptr_A_L1_2,hbm_addr(A_hbm_addr), DATA_TYPE_SIZE * dim_tile_M * dim_tile_K);
                                
                                //Trigger DMA transaction: move B tile from HBM to L1
                                flex_dma_async_1d(ptr_B_L1_2,hbm_addr(B_hbm_addr), DATA_TYPE_SIZE * dim_tile_K * dim_tile_N);
                                
                                //Wait all DMA transaction done
                                flex_dma_async_wait_all();
                            }
                            
                            if (flex_is_first_core())//Use the first core in specified cluster to configure and trigger RedMule
                            {
                                //Configure tile address in L1 and run RedMule acceleration
                                flex_redmule_trigger(ptr_A_L1_1, ptr_B_L1_1, ptr_CY_L1, REDMULE_OP);
                                
                                //Wait RedMule Done
                                flex_redmule_wait();
                            }
                        } else {
                            if (flex_is_dm_core()){//Use DM core to trigger DMA transcations
                            //Trigger DMA transaction: move A tile from HBM to L1
                            flex_dma_async_1d(local(ptr_A_L1_1),(A_hbm_addr), DATA_TYPE_SIZE * dim_tile_M * dim_tile_K);
                            
                            //Trigger DMA transaction: move B tile from HBM to L1
                            flex_dma_async_1d(local(ptr_B_L1_1),(B_hbm_addr), DATA_TYPE_SIZE * dim_tile_K * dim_tile_N);
                            
                            //Wait all DMA transaction done
                            flex_dma_async_wait_all();
                        }
                        
                        if (flex_is_first_core())//Use the first core in specified cluster to configure and trigger RedMule
                        {
                            //Configure tile address in L1 and run RedMule acceleration
                            flex_redmule_trigger(ptr_A_L1_2, ptr_B_L1_2, ptr_CY_L1, REDMULE_OP);
                            
                            //Wait RedMule Done
                            flex_redmule_wait();
                        }
                    }
                    flex_intra_cluster_sync();//Cluster barrier
                }
            
                //Last Computation
                if (flex_is_first_core())
                {
                    if (i%2 == 0)
                    {
                        //Configure tile address in L1 and run RedMule acceleration
                            flex_redmule_trigger(ptr_A_L1_1, ptr_B_L1_1, ptr_CY_L1, REDMULE_OP);
                        } else {
                            //Configure tile address in L1 and run RedMule acceleration
                        flex_redmule_trigger(ptr_A_L1_2, ptr_B_L1_2, ptr_CY_L1, REDMULE_OP);
                    }
                }
                flex_intra_cluster_sync();//Cluster barrier

                //Store Z tile
                if (flex_is_dm_core()){
                    //Trigger DMA transaction: move Z tile from L1 to HBM
                    flex_dma_async_1d((Y_hbm_addr),(ptr_CY_L1), DATA_TYPE_SIZE * dim_tile_M * dim_tile_N);
                    
                    //Wait all DMA transaction done
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();//Cluster barrier
            }
        }
        // Free L1 buffers after all tiles are done
        if (flex_is_first_core()) {
            flex_l1_free((void*)ptr_CY_L1);
            flex_l1_free((void*)ptr_A_L1_1);
            flex_l1_free((void*)ptr_B_L1_1);
            flex_l1_free((void*)ptr_A_L1_2);
            flex_l1_free((void*)ptr_B_L1_2);
        }
        flex_intra_cluster_sync();//Cluster barrier
    }
#endif
#ifdef LOG_ENABLE
    if (CID == 0) {
        if (flex_is_dm_core()) {
            printf("GEMM Done for tile_M: %d, tile_N: %d, tile_K: %d\n", dim_tile_M, dim_tile_N, dim_tile_K);
        }
        flex_intra_cluster_sync();//Cluster barrier
    }
#endif
}
