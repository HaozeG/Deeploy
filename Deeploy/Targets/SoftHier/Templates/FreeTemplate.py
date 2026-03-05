# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

SoftHierLocalTemplate = NodeTemplate("""
flex_intra_cluster_sync();                                     
if (flex_is_dm_core()) {
    flex_l1_free(${name});
}
""")

SoftHierGlobalTemplate = NodeTemplate("""
flex_intra_cluster_sync();                                      
if (flex_is_dm_core()) {
    flex_hbm_free(${name});
}
""")
