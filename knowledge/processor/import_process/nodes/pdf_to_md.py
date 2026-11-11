"""
    入口节点
"""
import subprocess
import time
from pathlib import Path

from processor.import_process.base import BaseNode
from processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """PDF 转 Markdown 节点"""
    name: str = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        将PDF文件通过mineru转换成MD文件，更细state
        Args:
            state: ImportGraphState 该节点处理之前的节点状态
        Returns:
            ImportGraphState：该节点处理之后的节点状态
        """
        pdf_path = state.get("pdf_path")
        file_dir = state.get("file_dir")
        # 1. 参数校验
        pdf_path_obj, file_dir_obj = self._validate_state_inputs_path(pdf_path, file_dir)
        # 2.mineru命令，pdf转换成md
        code = self._execute_mineru(pdf_path_obj, file_dir_obj)
        if code != 0:
            raise PdfConversionError(message=f"mineru命令实行失败，状态码：{code}", node_name=self.name)
        # 3.获取MD信息，更新state
        md_path = self._get_md_paths(pdf_path_obj, file_dir_obj)
        state["md_path"] = md_path

        return state

    def _validate_state_inputs_path(self, import_pdf_path: str | None, export_dir: str | None) -> tuple[Path, Path]:
        """
        校验传入的路径参数
        Args:
            import_pdf_path:str 导入的文件路径
            export_dir:str 导出的路径参数
        Returns:
            pdf_path_obj:Path    导入的文件路径
            file_dir_obj:Path    导出的文件路径
        """
        pdf_path = import_pdf_path
        if not pdf_path:
            raise ValidationError(message=f"上传的文件路径为空", node_name=self.name)
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"上传的文件路径不存在，{pdf_path}：无效或不存在", node_name=self.name)
        file_dir = export_dir
        if not file_dir:
            # 降级处理
            file_dir = pdf_path_obj.parent
        file_path_obj = Path(file_dir)
        # 确保输出目录存在，如果是多级目录会自动创建
        file_path_obj.mkdir(parents=True, exist_ok=True)
        return pdf_path_obj, file_path_obj

    def _execute_mineru(self, pdf_path_obj: Path, file_dir_obj: Path) -> int:
        """
        执行mineru命令，将PDF转换成MD
        Args:
            pdf_path_obj:Path PDF文件路径
            file_dir_obj:Path 转换后的路径
        Returns:
            state_code:状态码，（0:成功，非0:失败）
        """
        start_time = time.time()
        try:
            process = subprocess.Popen(
                args=["mineru", "-p", str(pdf_path_obj.resolve()), "-o", str(file_dir_obj.resolve()), "--backend",
                      "pipeline"],
                stdout=subprocess.PIPE,  # 捕获标准输出 子进程通过管道连通主进程，输出标准日志
                stderr=subprocess.STDOUT,  # 合并错误到标准输出
                encoding="utf-8",
                bufsize=1,  # 行缓冲，即实时输出
                errors="replace",  # 遇到乱码时替换
                text=True,  # 输出文本信息
            )
            # 日志输出
            if process.stdout:
                for line in process.stdout:
                    print(f"mineru命令输出的日志：{line.rstrip()}")
            state_code = process.wait()
            end_time = time.time()
            if state_code != 0:
                self.logger.error(f"mineru命令执行失败，退出码：{state_code}")
            else:
                self.logger.info(f"mineru命令执行成功，退出码：{state_code},耗时：{end_time - start_time:.2f}秒")
            return state_code
        except Exception as e:
            self.logger.error(f"执行 mineru 发生未知异常: {e}")
            return 1

    def _get_md_paths(self, pdf_path_obj: Path, file_dir_obj: Path) -> str:
        """
         返回生成后的MD文档的信息和路径
        Args:
            pdf_path_obj (Path): PDF文档路径
            file_dir_obj (Path): MD文档导出路径
         Returns:
            str: MD文档路径
        """
        file_stem = pdf_path_obj.stem
        md_path = str(file_dir_obj / file_stem / "auto" / f"{file_stem}.md")
        if not Path(md_path).exists():
            raise FileProcessingError(message=f"mineru解析失败或未生成MD文件，目标路径：{md_path}", node_name=self.name)
        return md_path
