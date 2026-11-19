"""
    入口节点
"""
import json
import os
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import ValidationError
from processor.import_process.state import ImportGraphState
from utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name: str = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """文档切片节点
        bug：长切节点【图片和代码未处理】
        Args:
            state:ImportGraphState
        Returns:
            state:ImportGraphState
        """
        self.log_step("step0", "开始进行文档切分...")
        # 1.数据校验
        md_content, file_title, max_content_length, min_content_length = self._validation_data(state)
        # 2.标题切分（初次）
        title_chunks: list[dict[str, Any]] = self.file_title_chunks(md_content, file_title)
        # 3.二次切分（长切短合）
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length
        max_title_length = self.config.max_title_length
        legal_chunks = self._split_and_merge(title_chunks, max_content_length, min_content_length, max_title_length)
        # 4.组装切片
        final_chunks = self._assemble_chunks(legal_chunks)
        # 5.数据回填
        self._backfill_data(state, final_chunks)
        # 6.日志
        self._output_log(md_content, final_chunks, max_content_length)
        # 7.备份
        self._backup(state, final_chunks)

        return state

    def _validation_data(self, state: ImportGraphState) -> tuple[str, str, int, int]:
        """数据校验
        Args:
            state: ImportGraphState
        Returns:
            md_content: str：
            file_title: str
            max_content_length: int
            min_content_length: int
        """
        self.log_step("step1", "切分文档的参数校验以及获取...")
        md_content = state.get("md_content")
        if not md_content:
            raise ValidationError(message="文档不存在", node_name=self.name)
        # 统一缩进格式
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n") if md_content else ""
        file_title = state.get("file_title")
        if not file_title:
            raise ValidationError(message="文件标题不存在", node_name=self.name)
        max_content_length = self.config.max_content_length
        min_content_length = self.config.min_content_length
        if not max_content_length or not min_content_length or max_content_length < 0 or min_content_length < 0 or max_content_length < min_content_length:
            raise ValidationError("max_content_length 和 min_content_length 参数不合法")
        return md_content, file_title, max_content_length, min_content_length

    def file_title_chunks(self, md_content: str, file_title: str) -> list[dict[str, Any]]:
        """标题切分:注意空行，代码围挡，副标题
        Args:
            md_content: str
            file_title:str
        Returns:
            title_chunks:list[dict[str, Any]]
        """
        self.log_step("step2", "根据标题进行切分...")
        title_chunks_list = []
        # 1.数据初步处理，生成行内容
        lines = md_content.split("\n")
        if not lines:
            self.logger.warning(msg="该文档内容空")
            return [{
                "title": file_title,
                "parent_title": file_title,
                "body": "",
                "file_title": file_title,
            }]
        # 2.遍历循环行，找出标题行信息
        title_dict_list = []
        in_code_block_flag = False
        pattern_title = re.compile(r"^\s*(#{1,6})\s*(.*)")
        for index, line in enumerate(lines):
            line = line.strip()
            if line.startswith("~~~") or line.startswith("```"):
                in_code_block_flag = not in_code_block_flag
            if not in_code_block_flag and pattern_title.search(line):
                title_dict = {
                    "title_level": len(pattern_title.search(line).group(1)),
                    "title_content": line,
                    "title_index": index
                }
                title_dict_list.append(title_dict)
        # 3.遍历标题行，封装数据
        if not title_dict_list:
            self.logger.warning(msg="该文档为无标题文档", node_name=self.name)
            return [{
                "title": file_title,
                "parent_title": file_title,
                "body": md_content,
                "file_title": file_title,
            }]
        title_list = [""] * 7
        title_level = title_dict_list[0].get("title_level")
        title_list[title_level] = title_dict_list[0].get("title_content")
        for parent_index in range(title_level + 1, 7):
            title_list[parent_index] = ""

        for end in range(1, len(title_dict_list)):
            # 记录各级标题
            title = title_dict_list[end - 1].get("title_content")
            title_level = title_dict_list[end - 1].get("title_level")
            title_list[title_level] = title
            # 父亲标题
            parent_title = file_title
            for parent_index in range(title_level - 1, 0, -1):
                if title_list[parent_index]:
                    parent_title = title_list[parent_index]
                    break
            # 数据清理
            for parent_index in range(title_level + 1, 7):
                title_list[parent_index] = ""
            start_index = title_dict_list[end - 1].get("title_index")
            end_index = title_dict_list[end].get("title_index")
            body = "\n".join(lines[start_index:end_index])
            title_chunks_list.append({
                "title": title,
                "parent_title": parent_title,
                "body": body,
                "file_title": file_title,
            })

        # 孤儿数据
        last_title_level = title_dict_list[-1].get("title_level")
        parent_title = file_title
        for parent_index in range(last_title_level - 1, 0, -1):
            if title_list[parent_index]:
                parent_title = title_list[parent_index]
                break
        title_chunks_list.append({
            "title": title_dict_list[-1].get("title_content"),
            "parent_title": parent_title,
            "body": "\n".join(lines[title_dict_list[-1].get("title_index"):]),
            "file_title": file_title,
        })

        return title_chunks_list

    def _split_and_merge(self, title_chunks: list[dict[str, Any]], max_content_length: int, min_content_length: int,
                         max_title_length: int) -> list[dict[str, Any]]:
        """
        二次切分（长切短合）
        Args:
            title_chunks: list[dict[str,str]]
            max_content_length:int
            min_content_length: int
            max_title_length:int
        Returns:
            final_chunks: list[dict[str,str]]
        """
        final_chunks: list[dict[str, Any]] = []
        # 1.长切
        self.log_step("step3", "进行长内容的切分")
        for chunk in title_chunks:
            legal_long_chunks = self.split_long_chunks(chunk, max_content_length, max_title_length)
            final_chunks.extend(legal_long_chunks)
        # 2.短合
        self.log_step("step3", "进行短切的合并")
        final_chunks = self.merge_short_chunks(final_chunks, min_content_length)
        return final_chunks

    def split_long_chunks(self, temp_chunk: dict[str, Any], max_content_length: int, max_title_length: int) -> list[
        dict[str, Any]]:
        """
        长切,注意：表格，标题处理【图片和代码未处理】
        Args:
            temp_chunk: dict[str,Any]
            max_content_length: int
            max_title_length: int
        Returns:
            final_chunks:dict[str,Any]
        """
        # 1.获取数据
        title = temp_chunk.get("title")
        parent_title = temp_chunk.get("parent_title")
        body = temp_chunk.get("body", "")
        file_title = temp_chunk.get("file_title")
        # 2.表格处理
        if '<table>' in body:
            body = MarkdownTableLinearizer.process(body)  # 表格降维处理，表格消失。
            temp_chunk['body'] = body
        # 3.标题处理
        title_prefix = f"{title}\n\n"
        if len(title_prefix) > max_title_length:
            self.logger.warning(msg=f"检测文件{file_title}对应的{title}长度过长...")
            title_prefix = f"{title[:max_title_length]}\n\n"
        # 4.判断chunk切分
        if len(body) + len(title_prefix) < max_content_length:
            return [temp_chunk]
        else:
            spliter = RecursiveCharacterTextSplitter(
                chunk_size=max_content_length - len(title_prefix),  # chunk最大长度
                separators=["\n\n", "\n", "。", "！", "？", "，", ""],
                keep_separator=False,
                chunk_overlap=0,
            )
            spliter_list = spliter.split_text(body)
            if len(spliter_list) <= 1:
                return [temp_chunk]
            return [{
                "title": title,
                "parent_title": parent_title,
                "body": text,
                "file_title": file_title,
                "part": index + 1}
                for index, text in enumerate(spliter_list)
            ]

    def merge_short_chunks(self, final_chunks: list[dict[str, Any]], min_content_length: int) -> list[dict[str, Any]]:
        """
        短合，
        Args:
            final_chunks:list[dict[str, Any]]
            min_content_length:int
        Returns:
            final_chunks:list[dict[str, Any]]
        """
        # 临时数据
        result_chunks: list[dict[str, Any]] = []
        temp_chunks = final_chunks[0]
        for chunks in final_chunks[1:]:
            is_same_parent = chunks.get("parent_title") == temp_chunks.get("parent_title")
            if is_same_parent and len(temp_chunks.get("body", "")) < min_content_length:
                temp_chunks["body"] = temp_chunks.get("body", "").lstrip() + "\n\n" + chunks.get("body", "").rstrip()
                temp_chunks["title"] = temp_chunks.get("parent_title")
                temp_chunks["part"] = 0
            else:
                result_chunks.append(temp_chunks)
                temp_chunks = chunks
        result_chunks.append(temp_chunks)
        # part序号处理
        temp_title_list:dict[str, int] = {}
        for index, result_chunk in enumerate(result_chunks):
            if "part" in result_chunk.keys():
                parent_title = result_chunk.get("parent_title","")
                temp_title_list[parent_title] = temp_title_list.get(parent_title,0) + 1
                new_part = temp_title_list.get(parent_title)
                result_chunks[index]["title"] = f"{result_chunk.get('title')} - {new_part}"
                result_chunks[index]["part"] = new_part

        return result_chunks

    def _assemble_chunks(self, legal_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        组装切片
        Args:
            legal_chunks:list[dict[str, Any]]
        Returns:
            final_chunks:list[dict[str, Any]]
        """
        self.log_step("step4", "组装最终的切片信息...")
        final_chunks: list[dict[str, Any]] = []
        for temp_chunk in legal_chunks:
            new_chunk = {
                "title": temp_chunk.get("title"),
                "parent_title": temp_chunk.get("parent_title"),
                "file_title": temp_chunk.get("file_title"),
                "content": temp_chunk.get("title", "").rstrip() + "\n\n" + temp_chunk.get("body", "").lstrip()
            }
            if "part" in temp_chunk.keys():
                new_chunk["part"] = temp_chunk.get("part")
            final_chunks.append(new_chunk)
        return final_chunks

    def _backfill_data(self, state: ImportGraphState, final_chunks: list[dict[str, Any]]) -> None:
        """
        回填数据
        Args:
            state:ImportGraphState
            final_chunks:list[dict[str, Any]]
        Returns:None
        """

        self.log_step("step5", "回填数据")
        state['chunks'] = final_chunks

    def _output_log(self, md_content: str, final_chunks: list[dict[str, Any]], max_content_length: int) -> None:
        """
        输出日志
        Args:
            md_content:str
            final_chunks: list[dict[str, Any]]
            max_content_length:int
        Returns:None
        """
        self.log_step("step6", "输出日志")
        lines_count = md_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(final_chunks)}")
        self.logger.info(f"最大切片长度: {max_content_length}")
        # 防御性输出
        print_max_count_size = 5
        print_max_title = 30
        if final_chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(final_chunks[:print_max_count_size]):
                title = sec.get("title", "")[:print_max_title]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(final_chunks) > print_max_count_size:
                self.logger.info(f"  ... 还有 {len(final_chunks) - print_max_count_size} 个章节")

    def _backup(self, state: ImportGraphState, final_chunks: list[dict[str, Any]]) -> None:
        """
        备份切片数据到JSON文件
        Args:
            state:ImportGraphState
            final_chunks:list[dict[str, Any]]
        Returns:None
        """
        self.log_step("step7", "备份切片数据到JSON文件")
        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.warning(msg="未设置 file_dir，跳过备份")
            return
        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
            self.logger.info(msg=f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(msg=f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()
    # 构造状态字典
    # file_path = r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\万用表RS-12的使用\auto\万用表RS-12的使用_new.md"
    file_path = r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\auto\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    state = {
        # "file_title": "万用表的使用",
        "file_title": "Aolynk CB304n Cable网桥 用户手册-5W100-整本手册",
        "md_content": content,
        # "file_dir": r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\万用表RS-12的使用\auto"
        "file_dir": r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\auto"
    }

    final_state = document_node(state)
    print(json.dumps(final_state, ensure_ascii=False, indent=4))
