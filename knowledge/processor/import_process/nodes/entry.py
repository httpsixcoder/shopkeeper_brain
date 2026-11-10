"""
入口节点

检测文件类型并设置处理标志
"""

from processor.import_process.base import BaseNode
from processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """
    入口节点

    根据输入文件的扩展名设置相应的处理标志，
    决定后续流程走 PDF 转换分支还是直接处理 MD 分支。
    """

    name = "entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        处理文件类型的检测

        Args:
            state: ImportGraphState 该节点处理之前的节点状态

        Returns:
            ImportGraphState：该节点处理之后的节点状态
        """
        pass

