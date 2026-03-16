from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, NodeTemplate, _NoVerbosity

_clusterSynchTemplate = NodeTemplate("""
        flex_intra_cluster_sync();
""")

class SoftHierClusterSynch(CodeTransformationPass):
    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock, name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        executionBlock.addLeft(_clusterSynchTemplate, {})
        executionBlock.addRight(_clusterSynchTemplate, {})
        return ctxt, executionBlock
