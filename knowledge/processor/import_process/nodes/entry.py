"""
入口节点

检测文件类型并设置处理标志
"""
from pathlib import Path

from modelscope.fileio.format.jsonplus import json_dumps

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import ValidationError
from processor.import_process.state import ImportGraphState, create_default_state


class EntryNode(BaseNode):
    """
    根据输入文件的扩展名设置相应的处理标志，
    决定后续流程走 PDF 转换分支还是直接处理 MD 分支。
    """

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        处理文件类型的检测
        Args:
            state: ImportGraphState 该节点处理之前的节点状态
        Returns:
            ImportGraphState：该节点处理之后的节点状态
        """

        import_file_path = state.get("import_file_path")
        export_file_path = state.get("file_dir")
        # 校验数据

        if not import_file_path:
            raise ValidationError(message=f"上传文件的路径为空", node_name=self.name)
        if not export_file_path:
            # 降级处理
            state["file_dir"] = str(Path(import_file_path).parent)
        import_file_path_obj = Path(import_file_path)
        if import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = str(import_file_path_obj)
        elif import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = str(import_file_path_obj)
        else:
            ValidationError(message=f"传入的文件不符合规定，要求md或者pdf，当前为：{import_file_path_obj.suffix}",
                            node_name=self.name)
        state["file_title"] = import_file_path_obj.stem

        return state


if __name__ == "__main__":
    setup_logging()
    state = create_default_state(
        import_file_path=r"G:\coder\git\project\shopkeeper_brain\test\pdf\万用表RS-12的使用.pdf",
        file_dir=r"G:\coder\git\project\shopkeeper_brain\test\temp_dir"

    )
    entry_node = EntryNode()
    test_state = entry_node(state)
    print(json_dumps(test_state, indent=4, ensure_ascii=False))
