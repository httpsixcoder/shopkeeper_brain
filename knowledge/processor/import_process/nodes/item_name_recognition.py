"""
    入口节点
"""
from processor.import_process.state import ImportGraphState
from processor.import_process.base import BaseNode, T


class ItemNameRecognitionNode(BaseNode):
    name: str = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        return {}
