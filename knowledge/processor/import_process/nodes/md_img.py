"""
    入口节点
"""
import base64
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Deque

from openai import OpenAI
from minio import Minio

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import StateFieldError, FileProcessingError
from processor.import_process.state import ImportGraphState, create_default_state
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients


@dataclass
class ImageContent:
    """封装图片上下文信息的类"""
    heading: str  # 标题
    pre_text: str  # 上文
    post_text: str  # 下文


@dataclass
class ImageInfo:
    """封装图片信息的类"""
    name: str  # 文件名
    path: str  # 文件路径
    content: ImageContent  # 图片上下文信息


# --------------------------1.读取MD处理类--------------------------
class MdFileHandler:

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self, state: ImportGraphState) -> tuple[str, Path, Path]:

        """
        读取MD文件
        :param
            state:上一个节点传入的状态
        :return:
             md_content：    MD文档内容
             md_path_obj,   MD文档路径
             md_images_path_obj MD文档中的图片的路径
        """
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name="md_content", expected_type=str,
                                  message="参数md_content 缺失或无效")
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"{md_path}文件不存在", node_name=self.node_name)
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()
        md_images_path_obj = md_path_obj.parent / "images"
        return md_content, md_path_obj, md_images_path_obj


# --------------------------2.图片扫描器--------------------------
class ImageScanner:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, md_images_path_obj: Path, md_content: str, image_extensions: set[str],
                     img_content_length: int) -> list[ImageInfo]:
        """
        按照图片扫描MD文档，生成图片信息
        :param
            md_images_path_obj:Path     图片路径
            md_content:str     MD文档内容
            image_extensions:str   图片后缀列表
            img_content_length:int 图片上下文长度
        :return:
            image_info_list:list[ImageInfo]    图片信息列表
        """
        image_info_list: list[ImageInfo] = []
        for img in md_images_path_obj.iterdir():
            if (img.suffix not in image_extensions) or (img.is_dir()):
                continue
            md_images_content_obj: ImageContent | None = self._find_context(img.name, md_content, img_content_length)
            if not md_images_content_obj:
                self.logger.warning(
                    f"在MD中没有找到图片的引用，上下文为None【可能时表格图片】-{str(md_images_content_obj)}")
                continue
            image_info = ImageInfo(
                name=img.name,
                path=str(img),
                content=md_images_content_obj
            )
            image_info_list.append(image_info)
        return image_info_list

    def _find_context(self, img_name: str, md_content: str, img_content_length: int) -> ImageContent | None:
        """
        查找MD文档中图片信息，拼接成图片上下文返回
        Args:
            img_name:str 图片名称
            md_content:str MD文档内容
            img_content_length:int 图片上下文长度
        Returns:
            md_images_content_obj:ImageContent|None 图片上下文
        """

        lines = md_content.split("\n")
        pattern_img = re.compile(r"!\[.*?]\(.*?" + re.escape(img_name) + r"\)")
        for index, line in enumerate(lines):
            if pattern_img.search(line):
                # 1.正则定位图片位置
                md_img_index = index
                # 2.向上找标题
                md_title_above, md_title_above_index = self._find_heading_above(lines[:md_img_index])
                # 3.向下找标题
                md_title_below_index = self._find_heading_below(lines[:], md_img_index)
                # 4.段落填装
                md_img_context_above = "" if md_title_above_index == -1 else self._extract_limited_context(
                    lines[md_title_above_index:md_img_index], img_content_length)
                md_img_context_below = "" if md_title_below_index == -1 else self._extract_limited_context(
                    lines[md_img_index:md_title_below_index], img_content_length)
                image_content = ImageContent(
                    heading=md_title_above,
                    pre_text=md_img_context_above,
                    post_text=md_img_context_below
                )
                return image_content
        return None

    def _find_heading_above(self, lines: list) -> tuple[str, int]:
        """
        查找距离图片最近的上一个标题，以及标题索引
        Args:
             lines:list MD文档内容列表
        Returns:
            md_title_above:str 最近上一个标题
            md_title_above_index:int 最近上一个标题索引
        """
        for index in range(len(lines) - 1, -1, -1):
            if re.compile(r"^#{1,6}\s.+").search(lines[index]):
                return lines[index], index
        self.logger.info(msg="无上文标题")
        return "", -1

    def _find_heading_below(self, lines: list, md_img_index: int) -> int:
        """
        查找距离图片最近的下一个标题索引
        Args:
            lines: lines:list MD文档内容列表
            md_img_index:int 图片索引
        Returns:
            md_title_below_index:int 最近下一个标题索引
        """
        for index in range(md_img_index, len(lines)):
            if re.compile(r"^#{1,6}\s.+").search(lines[index]):
                return index
        self.logger.info(msg="无下文标题")
        return -1

    def _extract_limited_context(self, lines: list[str], img_content_length: int) -> str:
        """
        返回规定上下文：
            判断列表第一个是否是图片从而决定是上文还是下文，是否反转
        Args:
            lines:list[str] 完整上下文
            img_content_length:int 规定上下长度
        Returns:
            md_img_context:str 返回的上文or下文
        """
        if not lines:
            self.logger.info(msg="无上下文内容")
            return ""
        # 1.标识上文还是下文 0:下文(正常)，1:上文(两次反转)
        flag = 0 if re.match(r"!\[.*?]\(.*?\)", lines[0].strip()) else 1
        # 2.找到有效段落
        temp_line: list[str] = []
        final_line: list[str] = []
        for line in lines:
            if not line.strip() or re.match(r"!\[.*?]\(.*?\)", line):
                if temp_line:
                    final_line.append("\n".join(temp_line))
                    temp_line = []  # 清空
                continue
            temp_line.append(line)
        if temp_line:
            final_line.append("\n".join(temp_line))
        # 3.贪心累加
        final_line = final_line[::-1] if flag == 1 else final_line
        final_content: list[str] = []
        total = 0
        for line in final_line:
            if len(line) + total > img_content_length:
                break
            final_content.append(line)
            total += len(line)
        final_content = final_content[::-1] if flag == 1 else final_content
        return "\n\n".join(final_content)


# --------------------------3.VLM摘要器--------------------------
class VLMSummarizer:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def summarize_all(self, image_info_list: list[ImageInfo], md_file_title: str, vl_model: str,
                      requests_per_minute: int) -> dict[str, str]:
        """
        根据图片信息，调用VML大模型生成摘要
        Args:
            image_info_list:list[ImageInfo] 图片信息列表
            md_file_title:str   文档标题名
            vl_model:str  模型名称
            requests_per_minute:int 每分钟最大请求数
        Returns:
            image_summary:dict[str,str] 图片摘要
        """
        image_summary: dict[str, str] = {}
        # 1.获取客户端，失败降级处理
        try:
            vml_client = AIClients.get_openai()
        except Exception as e:
            self.logger.warning(msg=f"客户端获取失败，使用默认摘要信息,{e}")
            for image_info in image_info_list:
                image_summary[image_info.name] = "图片描述"
            return image_summary
        # 2.滑动窗口限流，
        request_timestamps: Deque[float] = deque()
        for image_info in image_info_list:
            summary_context = self._enforce_rate_limit(request_timestamps, requests_per_minute, image_info,
                                                       md_file_title, vl_model, vml_client)
            image_summary[image_info.name] = summary_context
        # 3.获取上下文
        return image_summary

    def _enforce_rate_limit(self, request_timestamps: Deque[float], requests_per_minute: int, image_info: ImageInfo,
                            md_file_title: str, vl_model: str, vml_client: OpenAI, window: int = 60) -> str:
        """
        滑动窗口限流：60秒内最多10次请求【特殊情况报错，可后续完善】
        BUG：请求时间过长导致真是请求数大于10，请求时间过短导致无效等待
        Args:
            request_timestamps: Deque[float] = deque()时间戳队列
            requests_per_minute:int 每分钟最大请求数
            image_info:ImageInfo 图片信息
            md_file_title:str 文档标题
            vl_model:str 模型名称
            vml_client:OpenAI 客户端
            window:int 窗口长度，过期时间，默认60秒

        Returns:
            summary_context:str 摘要
        """
        time_now = time.time()
        while request_timestamps and time_now - request_timestamps[0] > window:
            request_timestamps.popleft()
        if len(request_timestamps) >= requests_per_minute:
            wait_time = window - (time_now - request_timestamps[0])
            if wait_time > 0:
                time.sleep(wait_time)
            time_now = time.time()
            while request_timestamps and time_now - request_timestamps[0] > window:
                request_timestamps.popleft()
        request_timestamps.append(time_now)
        summary = self._summarize_one(image_info, md_file_title, vl_model, vml_client)
        return summary

    def _summarize_one(self, image_info: ImageInfo, md_file_title: str, vl_model: str, vml_client: OpenAI):
        """
        调用模型返回摘要
        Args:
            image_info:ImageInfo 图片信息
            md_file_title:str 文档标题
            vl_model:str 模型名称
            vml_client:OpenAI 客户端
        Returns:
            summary_context:str 摘要
        """
        try:
            image_context = image_info.content.heading + "\n" + image_info.content.pre_text + "\n" + image_info.content.post_text
            with open(image_info.path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")  # Base64字符串
            response = vml_client.chat.completions.create(
                model=vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",  # 告诉 API：这是一段文字
                                "text": (
                                    f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                    f"背景信息：\n"
                                    f"  1. 所属文档标题：\"{md_file_title}\"\n"
                                    f"  2. 图片上下文：{image_context}\n"
                                    f"请结合图片内容和上述上下文信息，"
                                    f"用中文简要总结这张图片的内容，"
                                    f"生成一个精准的中文标题（不要包含图片二字）。"
                                )
                            },
                            {
                                "type": "image_url",  # 告诉 API：这是一张图片
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],  # type: ignore  <-- 这里加注释，解决警告
                max_tokens=100,
                stream=False
            )
            summary_context = response.choices[0].message.content   # type: ignore <-- 解决警告
            return summary_context
        except Exception as e:
            self.logger.warning(msg=f"调用VLM生成摘要失败，图片名称：{image_info.name}-原因：{e}")
            return "图片描述"


# --------------------------4.图片上传器--------------------------
class ImageUploader:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self, md_content: str, image_info_list: list[ImageInfo], image_summary: dict[str, str],
                           file_title: str, minio_bucket: str, minio_endpoint: str) -> str:
        """
        上传图片到MinIO，替换图片的引用摘要和路径
        Args:
            md_content:str  MD文档内容
            image_info_list:list[ImageInFo] 图片信息列表
            image_summary:dict[str,str] 图片摘要信息字典
            file_title:str    文档标题
            minio_bucket:str 桶名称 shopkeeper-brain
            minio_endpoint:str MinIO服务器端点
        Returns:
            md_content_new:str 更新后的MD文件
        """
        remote_urls: dict[str, str] = {}
        try:
            minio_client: Minio = StorageClients.get_minio_client()
        except Exception as e:
            self.logger.warning(msg=f"获取MinIO客户端失败：{e}")
            for image_info in image_info_list:  # 全局降级处理
                remote_urls[image_info.name] = f"images/{image_info.path}"
        else:
            for image_info in image_info_list:
                urls = self._upload_all(minio_client, image_info, file_title, minio_bucket, minio_endpoint)
                remote_urls[image_info.name] = urls
        finally:
            md_content_new = self._replace_in_md(md_content, remote_urls, image_summary)
            return md_content_new

    def _upload_all(self, minio_client: Minio, image_info: ImageInfo, file_title: str, minio_bucket: str,
                    minio_endpoint: str) -> str:
        """
        山上传所有的图片，生成每个图片对应的远程路径的字典
        Args:
            minio_client:Minio 连接对象
            image_info: ImageInFo 图片信息
            file_title:str 文档标题
            minio_bucket:str 桶名
            minio_endpoint:str MinIO服务器端点
        Returns:
            urls:str 远端地址
        """
        try:
            file_object = f"{file_title}/{image_info.name}"
            minio_client.fput_object(bucket_name=minio_bucket, object_name=file_object, file_path=image_info.path)
            return f"{minio_endpoint}/{minio_bucket}/{file_object}"
        except Exception as e:
            self.logger.warning(f"上传图片{image_info.name} 失败：{e}")
            return f"images/{image_info.name}"  # 局部降级处理

    def _replace_in_md(self, md_content: str, remote_urls: dict[str, str], image_summary: dict[str, str]) -> str:
        """
       替换MD文档中图片的引用摘要和路径
       Args:
           md_content:str MD文档
           remote_urls:dict[str,str] 每个图片对应的远程路径 字典数据
           image_summary:dict[str,str] 每个图片对应的摘要 字典数据
       Returns:
           md_content_new:str 替换后的MD文档
       """
        # (.*?)  匹配组，有两个匹配组，分别用来匹配摘要group(1)和匹配路径group(2)。group(0) 返回原值
        # NB代码，
        pattern = re.compile(r"!\[(.*?)]\((.*?)\)")

        def replacer(match):
            origin_image_path = match.group(2)
            origin_image_name = Path(origin_image_path).name
            for name, summary in image_summary.items():
                if name in origin_image_name:
                    return f"![{summary}]({remote_urls[name]})"
            self.logger.warning(msg="MD文档替换内容出错，使用默认值")
            return match.group(0)

        # 利用正则表达式从md_content中匹配，每次回调replacer函数
        return pattern.sub(replacer, md_content)


# --------------------------主节点--------------------------
class MdImgNode(BaseNode):
    """MD中土拍你处理任务的主节点"""
    name: str = "md_img_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(self.logger, self.name)
        self.image_scanner = ImageScanner(self.logger, self.name)
        self.vlm_summarizer = VLMSummarizer(self.logger, self.name)
        self.image_uploader = ImageUploader(self.logger, self.name)

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """处理MD文件中的图片,生成摘要,上传图片到MinIO,更新MD中图片的路径和摘要"""

        config = self.config
        # 1.读取上一个节点中的MD文件，返回文件内容，文件路径，文件中图片的路径
        md_content, md_path_obj, md_images_path_obj = self.md_file_handler.read_md(state)
        if not md_images_path_obj.exists():
            self.logger.info(msg=f"{md_path_obj}没有返回任何图片，无序执行后续业务流程")
            state["md_content"] = md_content
            return state
        # 2.图片扫描器，获取上下文信息，返回每一个完整的图片信息列表
        image_info_list: list[ImageInfo] = self.image_scanner.scan_img_dir(md_images_path_obj, md_content,
                                                                           config.image_extensions,
                                                                           config.img_content_length)
        # 3.摘要生成器，调用VML模型生成摘要，返回图片摘要信息字典
        image_summary: dict[str, str] = self.vlm_summarizer.summarize_all(image_info_list, md_path_obj.stem,
                                                                          config.vl_model, config.requests_per_minute)

        # 4.上传图片到MinIO服务器上，同时更新MD文档中的图片的摘要和路径
        md_content_new = self.image_uploader.upload_and_replace(md_content, image_info_list, image_summary,
                                                                md_path_obj.stem,
                                                                config.minio_bucket,
                                                                config.get_minio_base_url())
        print(md_content_new)
        # 5.备份旧的MD文档
        return state


if __name__ == "__main__":
    setup_logging()
    test_state = create_default_state(
        md_path=r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\万用表RS-12的使用\auto\万用表RS-12的使用.md",
    )
    md_img_node = MdImgNode()
    final_state = md_img_node(test_state)
    print(json.dumps(final_state, indent=4, ensure_ascii=False))
