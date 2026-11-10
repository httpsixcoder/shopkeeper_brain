"""
    入口节点
"""
from processor.import_process.state import ImportGraphState
from processor.import_process.base import BaseNode, T


class BgeEmbeddingNode(BaseNode):
    name: str = "beg_embedding_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        return {}
