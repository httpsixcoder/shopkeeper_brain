"""
    入口节点
"""
from processor.import_process.state import ImportGraphState
from processor.import_process.base import BaseNode, T


class ImportMilvusNode(BaseNode):
    name: str = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        return {}
