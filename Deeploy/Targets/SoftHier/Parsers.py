from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import GEMMParser, RQGEMMParser

from typing import Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.Targets.Generic.Parsers import GEMMParser, RQGEMMParser



class SoftHierGEMMParser(GEMMParser):

    def parseNode(self, node: gs.Node) -> bool:
        ret = super().parseNode(node)
        
        # TODO: temporary workaround to be able to use the cluster id for mapping, could be replaced by a more generic solution in the future
        # read cluster id from attributes and store it in operator representation for later use in the mapping phase
        if 'cluster_id' in node.attrs:
            self.operatorRepresentation['cluster_id'] = node.attrs['cluster_id']

        if not ret:
            return False

        if not all([
                self.operatorRepresentation['transA'] == 0,
        ]):
            return False

        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        newCtxt, ret = super().parseNodeCtxt(ctxt, node, channels_first)

        if not ret:
            return ctxt, False

        if not all([
                self.operatorRepresentation['batch'] == 1,
        ]):
            return ctxt, False

        return newCtxt, True