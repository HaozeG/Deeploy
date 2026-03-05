# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.DeeployTypes import DeploymentPlatform, StructBuffer, TopologyOptimizer, VariableBuffer


class SoftHierDeployer(SignPropDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 inputOffsets: Dict[str, int] = {}):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir)

        self.inputOffsets = inputOffsets

        self.loweringOptimizer.passes += []

    def generateBufferAllocationCode(self) -> str:
        """Generate SoftHier global buffer allocation code with synchronized IO publication.

        SoftHier allocators are DM-core-only. Publish `DeeployNetwork_inputs[]` /
        `DeeployNetwork_outputs[]` from DM core only, with barriers before/after
        publication so all cores observe initialized pointers.
        """
        if not self.parsed or not self.bound:
            raise RuntimeError('You need to parse and bind the network before generating code!')

        ctxt = self.ctxt.copy()

        inputs = self.inputs()
        outputs = self.outputs()
        callStack = ''

        for node in ctxt.globalObjects.values():
            if isinstance(node, VariableBuffer) and not isinstance(node, StructBuffer):
                assert issubclass(node._type, Pointer), f"Global VariableBuffer {node.name} is not a Pointer!"
                if node._deploy:
                    name = node.name
                    node.name = ctxt._mangle(node.name)
                    callStack += node.alloc()
                    node.name = name

        for node in ctxt.globalObjects.values():
            if isinstance(node, StructBuffer):

                if node._deploy:
                    name = node.name
                    node.name = ctxt._mangle(node.name)
                    callStack += node.alloc()
                    node.name = name

        callStack += "flex_intra_cluster_sync();"
        callStack += "if (flex_is_dm_core()) {"
        for idx, i in enumerate(inputs):
            callStack += ctxt._mangle("inputs") + f"[{idx}] = (void*) {ctxt._mangle(i.name)};"
        for idx, i in enumerate(outputs):
            callStack += ctxt._mangle("outputs") + f"[{idx}] = (void*) {ctxt._mangle(i.name)};"
        callStack += "}"
        callStack += "flex_intra_cluster_sync();"

        return callStack
