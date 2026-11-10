"""
    入口节点
"""
from processor.import_process.base import BaseNode, T
from processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name: str = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        return {}

        # 步骤1：校验PDF路径和输出目录

        # 步骤2：上传PDF到MinerU并轮询解析结果

        # 步骤3：下载ZIP包并提取MD文件

        # 步骤4：读取md的内容

        # 步骤5：更新state状态




















