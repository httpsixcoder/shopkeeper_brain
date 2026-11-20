from dataclasses import dataclass

from pymilvus import DataType, MilvusClient


@dataclass
class ItemNameSchemaBase:
    """存放所有公共的字段和配置，对应你的 Base"""
    pk_max_length: int = 100
    max_varchar_length: int = 512
    dense_dim: int = 1024


@dataclass
class ItemNameSchemaCreate(ItemNameSchemaBase):
    """建表的配置类"""

    def build_schema(self, milvus_client: MilvusClient):
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True,
                         max_length=self.pk_max_length)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=self.max_varchar_length)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=self.max_varchar_length)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        return schema

    def build_index_params(self, milvus_client: MilvusClient):
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_name="dense_index", index_type="AUTOINDEX",
                               metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector", index_name="sparse_index",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        return index_params
