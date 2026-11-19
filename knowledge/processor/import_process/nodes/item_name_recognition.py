"""
    入口节点
"""
import json
import os
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import StateFieldError, ValidationError
from processor.import_process.state import ImportGraphState
from prompt.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from schema.item_name_schema import ItemNameSchemaCreate
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name: str = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        识别商品名
        Args:
            state: ImportGraphState
        Returns:
            state: ImportGraphState
        """

        # 1.校验数据
        chunks_k = self.config.item_name_chunk_k
        chunks_size = self.config.item_name_chunk_size
        file_title, chunks = self._validate_data(state, chunks_k, chunks_size)
        # 2.构建上下文
        context = self._build_context(chunks, chunks_k, chunks_size)
        # 3.调用LLM识别商品名字
        item_name = self._discern_item_name(file_title, context)
        # 4.调用BGE-M3向量化商品名
        dense_vector, sparse_vector = self._vectorization_item_name(item_name)
        # 5.向量持久化到Milvus数据库
        item_name_collection = self.config.item_name_collection
        self._save_vector_to_milvus(file_title, item_name, dense_vector, sparse_vector, item_name_collection)
        # 6.回填数据
        self._backfill_data(state, file_title)
        # 7.返回state
        self._backup_chunks(state, chunks)
        return state

    def _validate_data(self, state: ImportGraphState, chunks_k: int, chunks_size: int) -> tuple[
        str, list[dict[str, Any]]]:
        """
        校验数据：
        Args:
            state:ImportGraphState
            chunks_k:int
            chunks_size:int
        Returns:
            file_title:str
            chunks:list[dict[str,Any]]
        """
        file_title = state.get("file_title")
        chunks = state.get("chunks")
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str,
                                  message="文件标题不存在")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list,
                                  message="文件切片不存在")
        if not self.config.item_name_chunk_k or self.config.item_name_chunk_k <= 0:
            raise ValidationError(message="item_name_chunk_k不能为空", node_name=self.name)
        if not self.config.item_name_chunk_size or self.config.item_name_chunk_size <= 0:
            raise ValidationError(message="item_name_chunk_size不能为空", node_name=self.name)
        return file_title, chunks

    def _build_context(self, chunks: list[dict[str, Any]], chunks_k: int, chunks_size: int) -> str:
        """
        构建上下文
        Args:
            chunks:list[dict[str,Any]]
            chunks_k:int
            chunks_size:int
        Returns:
            context:str
        """
        context_list: list[str] = [chunk.get("context", "") for chunk in chunks[:chunks_k]]
        final_context: list[str] = []
        for chunk in context_list:
            if len(chunk) + len(final_context) <= chunks_size:
                final_context.append(chunk)
        return "\n".join(final_context)

    def _discern_item_name(self, file_title: str, context: str) -> str:
        """
        调用LLM识别商品名
        Args:
            file_title:
            context:
        Returns:
            item_name:str
        """
        try:
            llm_client = AIClients.get_llm_openai()
            prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=context)
            response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            result = response.content.strip()
            if not result or response.content == "UNKNOWN":
                return file_title
            return json.loads(result)["item_name"]
        except Exception as e:
            self.logger.warning(msg=f"LLM调用失败，原因：{e}")
            return file_title

    def _vectorization_item_name(self, item_name: str) -> tuple[list[float], list[float]]:
        """
        调用BGE-M3向量化商品名
        Args:
            item_name: str
        Returns:
             dense_vector:list[float]
             sparse_vector:list[float]
        """
        try:
            bge_client = AIClients.get_bge_m3_client()
            vector_result = bge_client.encode_documents(documents=[item_name])
            dense_vector: list[float] = vector_result.get("dense")[0].tolist()
            start_index = vector_result.get("sparse").indptr[0]
            end_index = vector_result.get("sparse").indptr[1]
            token_ids = vector_result.get("sparse").indices[start_index:end_index].tolist()
            token_weight = vector_result.get("sparse").data[start_index:end_index].tolist()
            sparse_vector = list(zip(token_ids, token_weight))
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.warning(msg=f"商品名向量化失败，原因：{e}")
            return [], []

    def _save_vector_to_milvus(self, file_title: str, item_name: str, dense_vector: list[float],
                               sparse_vector: list[float], item_name_collection: str) -> None:
        """
        持久化商品名信息到Milvus数据库
        Args:
            file_title:
            item_name:
            dense_vector:
            sparse_vector:
        Returns:None
        """
        try:
            milvus_client = StorageClients.get_milvus_client()
            if not milvus_client.has_collection(item_name_collection):
                builder = ItemNameSchemaCreate()
                builder_schema = builder.build_schema(milvus_client=milvus_client)
                builder_index = builder.build_index_params(milvus_client=milvus_client)
                milvus_client.create_collection(
                    collection_name=item_name_collection,
                    schema=builder_schema,
                    index_params=builder_index
                )
                self.logger.info(f"创建集合 {item_name_collection} 并构建索引")
            milvus_data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
            result = milvus_client.insert(collection_name=item_name_collection, data=[milvus_data])
            self.logger.info(msg=f"Milvus数据库存储数据成功，{result.get('ids', ['1'])[0]}")
        except Exception as e:
            self.logger.warning(msg=f"Milvus数据库存储数据失败，{e}")

    def _backfill_data(self, state: ImportGraphState, file_title: str) -> None:
        """
        回填数据，商品名，以及chunks中的商品名
        Args:
            state: ImportGraphState
            file_title:str
        Returns:None
        """
        state["file_title"] = file_title
        for chunk in state["chunks"]:
            chunk["item_name"] = file_title

        pass

    def _backup_chunks(self, state: ImportGraphState, chunks: list[dict[str, Any]]) -> None:
        """
        备份chunks数据，作为下一个节点测试使用
        Args:
            state:ImportGraphState
            chunks:list[dict[str, Any]]
        Returns:None
        """
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        os.makedirs(local_dir, exist_ok=True)  # exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks_item_name.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")


if __name__ == "__main__":
    setup_logging()
    temp_content = ""
    # file_path = r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\万用表RS-12的使用\auto\chunks.json"
    file_path = r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\auto\chunks.json"
    with open(file_path, "r", encoding="utf-8") as f:
        temp_content = json.load(f)

    state = {
        # "file_title": "万用表RS-12的使用",
        "file_title": "Aolynk CB304n Cable网桥 用户手册-5W100-整本手册",
        "chunks": temp_content,
        # "file_dir": r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\万用表RS-12的使用\auto"
        "file_dir": r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\Aolynk CB304n Cable网桥 用户手册-5W100-整本手册\auto"
    }
    node = ItemNameRecognitionNode()
    final_state = node(state)
    print(json.dumps(final_state, ensure_ascii=False, indent=4))
