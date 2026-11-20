"""
    入口节点
"""
import json
import os
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.exceptions import ValidationError
from processor.import_process.state import ImportGraphState
from schema.chunks_schema import ChunksSchemaCreate
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients


class BgeEmbeddingNode(BaseNode):
    """内容切片向量化"""
    name: str = "bge_embedding_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        内容切片向量化
        Args:
            state:ImportGraphState
        Returns:
            state:ImportGraphState
        """

        # 1.数据校验
        chunks, batch_size = self._validation_data(state)
        # 2.获取BGE-M3模型客户端和Milus客户端
        bge_m3_client, milvus_client = self._get_clients()
        # 4.判断并创建数据模型类
        self._create_schema(milvus_client)
        # 5.分批处理
        final_chunks = self._batch_data(chunks, batch_size, bge_m3_client)
        # 6.数据回填
        state["chunks"] = final_chunks
        # 7.备份
        self._backup_data(state)
        # 8.返回
        return state

    def _validation_data(self, validation_state: ImportGraphState) -> tuple[list[dict[str, Any]], int]:
        """
        数据校验
        Args:
            validation_state:ImportGraphState
        Returns:
            chunks:list[dict[str, Any]]
            embedding_batch_size:int
        """
        chunks = validation_state.get("chunks", [])
        embedding_batch_size = self.config.embedding_batch_size
        if not chunks:
            self.logger.warning(msg="数据切片内容为空")
        if not embedding_batch_size or embedding_batch_size <= 0:
            self.logger.warning(msg="分批处理的批次容量错误")
        return chunks, embedding_batch_size

    def _get_clients(self) -> tuple[BGEM3EmbeddingFunction, MilvusClient]:
        """
        获取BGE-M3模型客户端和Milus客户端
        Returns:
            bge_m3_client:BGEM3EmbeddingFunction
            milvus_client:MilvusClient
        """
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.warning(msg=f"BGE_M3模型客户端获取失败，原因：{e}")
            raise ValidationError(message="BGE客户端初始化失败", node_name=self.name)
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.warning(msg=f"Milvus客户端获取失败，原因：{e}")
            raise ValidationError(message="Milvus客户端初始化失败", node_name=self.name)
        return bge_m3_client, milvus_client

    def _create_schema(self, milvus_client: MilvusClient) -> None:
        """
        判断是否创建
        Args:
            milvus_client:MilvusClient
        Returns:None
        """
        collection_name = self.config.chunks_collection
        if not milvus_client.has_collection(collection_name=collection_name):
            builder = ChunksSchemaCreate()
            milvus_client.create_collection(
                collection_name=collection_name,
                schema=builder.build_schema(milvus_client),
                index_params=builder.build_index_params(milvus_client)
            )
            self.logger.info(msg=f"已成功创建{collection_name}集合")

    def _batch_data(self, chunks: list[dict[str, Any]], batch_size: int,
                    bge_m3_client: BGEM3EmbeddingFunction) -> list[dict[str, Any]]:
        """
        分批处理
        Args:
            chunks:list[dict[str, Any]]
            batch_size:int
            bge_m3_client:BGEM3EmbeddingFunction
        Returns:
            final_chunks:list[dict[str, Any]]
        """
        final_chunks: list[dict[str, Any]] = []
        for batch_index in range(0, len(chunks), batch_size):
            batch = chunks[batch_index:batch_index + batch_size]
            try:
                batch = self.embed_temp_chunks(batch, bge_m3_client)
            except Exception as e:
                self.logger.warning(f"批次 {batch_index} 向量化失败，保留原始切片，err:{e}")
            final_chunks.extend(batch)
        return final_chunks

    def embed_temp_chunks(self, chunks: list[dict[str, Any]],
                          bge_m3_client: BGEM3EmbeddingFunction) -> list[dict[str, Any]]:

        """
        批次处理
        Args:
            chunks:list[dict[str, Any]]
            bge_m3_client:BGEM3EmbeddingFunction
        Returns:
            final_chunk:list[dict[str, Any]]
        """
        contents = [f"{chunk.get('item_name')}\n{chunk.get("content")}" for chunk in chunks]
        try:
            vector_result = bge_m3_client.encode_documents(contents)
        except Exception as e:
            self.logger.warning(msg=f"向量化失败，原因：{e}")
            return chunks

        csr = vector_result["sparse"]
        dense_list = vector_result["dense"]
        for index, chunk in enumerate(chunks):
            dense_vector = dense_list[index].tolist()
            start_index = csr.indptr[index]
            end_index = csr.indptr[index + 1]
            sparse_indices = csr.indices[start_index:end_index]
            sparse_data = csr.data[start_index:end_index]
            # sparse_vector = dict(zip(sparse_indices, sparse_data))
            sparse_vector = {int(k): float(v) for k, v in zip(sparse_indices, sparse_data)}
            chunk["dense_vector"] = dense_vector
            chunk["sparse_vector"] = sparse_vector
        return chunks

    def _backup_data(self, backup_state: ImportGraphState) -> None:
        """
        备份数据
        Args:
            backup_state: ImportGraphState
        Returns:None
        """
        local_dir = backup_state.get("file_dir", "")
        if not local_dir:
            return
        os.makedirs(local_dir, exist_ok=True)  # exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks_item_name_embedded.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(backup_state.get("chunks"), f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")


if __name__ == '__main__':
    setup_logging()
    base_temp_dir = Path(r"G:\coder\git\project\shopkeeper_brain\test\temp_dir\万用表RS-12的使用\auto")
    input_path = base_temp_dir / "chunks_item_name.json"
    with open(input_path, "r", encoding="utf-8") as rf:
        content = json.load(rf)
    temp_state = {
        "chunks": content,
        "file_dir": base_temp_dir,
        "item_name": "万用表RS-12"
    }
    node_bge_embedding = BgeEmbeddingNode()
    proceed_result = node_bge_embedding(temp_state)
    # print(json.dumps(proceed_result, default=str, ensure_ascii=False, indent=4))
