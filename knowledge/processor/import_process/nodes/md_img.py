"""
    入口节点
"""
from processor.import_process.base import BaseNode, T
from processor.import_process.state import ImportGraphState


class MdImgNode(BaseNode):
    name: str = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        return {}
