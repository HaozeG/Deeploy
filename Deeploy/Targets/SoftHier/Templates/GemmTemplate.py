from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.CommonExtensions.DataTypes import int8_t, uint8_t, float16_t, int16_t, uint16_t

class SoftHier_GemmTemplate(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # TODO: convert data type to REDMULE specific format
        redmule_dtype_mapping = {
            'uint16_t*': 'REDMULE_UINT_16',
            'int16_t*': 'REDMULE_INT_16',
            'float16_t*': 'REDMULE_FP_16',
            'uint8_t*': 'REDMULE_UINT_8',
            'int8_t*': 'REDMULE_INT_8',
            'fp8_t*': 'REDMULE_FP_8'
        }
        redmule_dtype_size_mapping = {
            'REDMULE_UINT_16': 2,
            'REDMULE_INT_16': 2,
            'REDMULE_FP_16': 2,
            'REDMULE_UINT_8': 1,
            'REDMULE_INT_8': 1,
            'REDMULE_FP_8': 1
        }
        # set dtype attribute for RedMule code generation
        # log error if data type is not supported by RedMule
        try:            
            operatorRepresentation['dtype'] = redmule_dtype_mapping[operatorRepresentation['data_out_type'].typeName]
            operatorRepresentation['dtype_size'] = redmule_dtype_size_mapping[operatorRepresentation['dtype']]
        except KeyError:
            raise ValueError(f"Data type {operatorRepresentation['data_out_type'].typeName} not supported by RedMule! Supported types: {list(redmule_dtype_mapping.keys())}")
        # infer data type from input tensors, assuming all inputs have the same data type

        return ctxt, operatorRepresentation, []
    
SoftHierGemmTemplateStr = r"""
    #define ELEM_SIZE      ${dtype_size}
    #define GEMM_DIMENSION 32
    #define GEMM_SIZE_BYTE (GEMM_DIMENSION * GEMM_DIMENSION * ELEM_SIZE)
    #define TILE_DIMENSION 32
    #define TILE_SIZE_BYTE (TILE_DIMENSION * TILE_DIMENSION * ELEM_SIZE)
    #define TILES_PER_DIM  (GEMM_DIMENSION/TILE_DIMENSION)

    #define X_HBM_OFFSET   ${A}
    #define W_HBM_OFFSET   ${B}
    #define Y_HBM_OFFSET   ${C}
    #define Z_HBM_OFFSET   ${data_out}

    #define X_L1_OFFSET1   0
    #define W_L1_OFFSET1   (X_L1_OFFSET1 + TILE_SIZE_BYTE)
    #define X_L1_OFFSET2   (W_L1_OFFSET1 + TILE_SIZE_BYTE)
    #define W_L1_OFFSET2   (X_L1_OFFSET2 + TILE_SIZE_BYTE)
    #define YZ_L1_OFFSET   (W_L1_OFFSET2 + TILE_SIZE_BYTE)
                                 
    uint32_t CID = flex_get_cluster_id();//Get cluster ID

    //Initialize RedMule Paramters
    if ((CID == ${cluster_id}))//Use the first core in specified cluster to configure RedMule
    {
        if (flex_is_first_core())
        {
            //Configure M-N-K of tile that RedMule will accelerate
            flex_redmule_config(TILE_DIMENSION, TILE_DIMENSION, TILE_DIMENSION);
        }
        flex_intra_cluster_sync();//Cluster barrier
    }


    if (CID == ${cluster_id})//Only let specified cluster to work
    {
        //Iterate over every Z tiles
        for (int row = 0; row < TILES_PER_DIM; ++row)
        {
            for (int col = 0; col < TILES_PER_DIM; ++col)
            {
                //Start address of tiles
                uint32_t X_hbm_addr = X_HBM_OFFSET + row * TILES_PER_DIM * TILE_SIZE_BYTE;
                uint32_t W_hbm_addr = W_HBM_OFFSET + col * TILE_SIZE_BYTE;
                uint32_t Y_hbm_addr = Y_HBM_OFFSET + row * TILES_PER_DIM * TILE_SIZE_BYTE + col * TILE_SIZE_BYTE;
                uint32_t Z_hbm_addr = Z_HBM_OFFSET + row * TILES_PER_DIM * TILE_SIZE_BYTE + col * TILE_SIZE_BYTE;


                //Preload Y,X,W tiles
                if (flex_is_dm_core()){//Use DM core to trigger DMA transcations
                    //Trigger DMA transaction: move Y tile from HBM to L1
                    flex_dma_async_1d(local(YZ_L1_OFFSET),hbm_addr(Y_hbm_addr), TILE_SIZE_BYTE);

                    //Trigger DMA transaction: move X tile from HBM to L1
                    flex_dma_async_1d(local(X_L1_OFFSET1),hbm_addr(X_hbm_addr), TILE_SIZE_BYTE);

                    //Trigger DMA transaction: move W tile from HBM to L1
                    flex_dma_async_1d(local(W_L1_OFFSET1),hbm_addr(W_hbm_addr), TILE_SIZE_BYTE);

                    //Wait all DMA transaction done
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();//Cluster barrier

                //Compute X and W to YZ in L1 + preload next X and W
                for (int i = 0; i < (TILES_PER_DIM-1); ++i)
                {
                    //Calculate next X and W tile address in HBM
                    X_hbm_addr += TILE_SIZE_BYTE;
                    W_hbm_addr += TILES_PER_DIM * TILE_SIZE_BYTE;

                    //Ping-Pong operations on Double-Buffering X and W
                    if (i%2 == 0)
                    {
                        if (flex_is_dm_core()){//Use DM core to trigger DMA transcations
                            //Trigger DMA transaction: move X tile from HBM to L1
                            flex_dma_async_1d(local(X_L1_OFFSET2),hbm_addr(X_hbm_addr), TILE_SIZE_BYTE);

                            //Trigger DMA transaction: move W tile from HBM to L1
                            flex_dma_async_1d(local(W_L1_OFFSET2),hbm_addr(W_hbm_addr), TILE_SIZE_BYTE);

                            //Wait all DMA transaction done
                            flex_dma_async_wait_all();
                        }

                        if (flex_is_first_core())//Use the first core in specified cluster to configure and trigger RedMule
                        {
                            //Configure tile address in L1 and run RedMule acceleration
                            flex_redmule_trigger(X_L1_OFFSET1, W_L1_OFFSET1, YZ_L1_OFFSET, ${dtype});

                            //Wait RedMule Done
                            flex_redmule_wait();
                        }
                    } else {
                        if (flex_is_dm_core()){//Use DM core to trigger DMA transcations
                            //Trigger DMA transaction: move X tile from HBM to L1
                            flex_dma_async_1d(local(X_L1_OFFSET1),hbm_addr(X_hbm_addr), TILE_SIZE_BYTE);

                            //Trigger DMA transaction: move W tile from HBM to L1
                            flex_dma_async_1d(local(W_L1_OFFSET1),hbm_addr(W_hbm_addr), TILE_SIZE_BYTE);

                            //Wait all DMA transaction done
                            flex_dma_async_wait_all();
                        }

                        if (flex_is_first_core())//Use the first core in specified cluster to configure and trigger RedMule
                        {
                            //Configure tile address in L1 and run RedMule acceleration
                            flex_redmule_trigger(X_L1_OFFSET2, W_L1_OFFSET2, YZ_L1_OFFSET, ${dtype});

                            //Wait RedMule Done
                            flex_redmule_wait();
                        }
                    }
                    flex_intra_cluster_sync();//Cluster barrier
                }

                //Last Computation
                if (flex_is_first_core())
                {
                    flex_redmule_trigger(X_L1_OFFSET2, W_L1_OFFSET2, YZ_L1_OFFSET, ${dtype});
                    flex_redmule_wait();
                }
                flex_intra_cluster_sync();//Cluster barrier

                //Store Z tile
                if (flex_is_dm_core()){
                    //Trigger DMA transaction: move Z tile from L1 to HBM
                    flex_dma_async_1d(hbm_addr(${data_out}),local(YZ_L1_OFFSET), TILE_SIZE_BYTE);

                    //Wait all DMA transaction done
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();//Cluster barrier
            }
        }
    }

"""

SoftHierGemm_Template = SoftHier_GemmTemplate(SoftHierGemmTemplateStr)