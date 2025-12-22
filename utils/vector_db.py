from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)
from qdrant_client.http.exceptions import UnexpectedResponse, ApiException


class QdrantVectorDB:
    """
    基于QdrantClient的向量数据库操作封装类
    支持向量的增删改查、集合创建/删除/检查等操作
    """

    def __init__(
            self,
            host: str = "localhost",
            port: int = 6333,
            api_key: Optional[str] = None,
            https: bool = False,
            collection_name: str = "default_collection",
            vector_size: int = 768,
            distance: Distance = Distance.COSINE,
    ):
        """
        初始化Qdrant客户端

        Args:
            host: Qdrant服务地址
            port: Qdrant服务端口
            api_key: 认证API密钥（云服务需要）
            https: 是否使用HTTPS协议
            collection_name: 默认集合名称
            vector_size: 向量维度
            distance: 向量距离计算方式（COSINE/ EUCLID/ DOT）
        """
        # 初始化Qdrant客户端
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=https,
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        # 检查并创建默认集合（如果不存在）
        self.create_collection_if_not_exists()


    def create_collection_if_not_exists(self) -> bool:
        """
        检查集合是否存在，不存在则创建

        Returns:
            bool: 是否创建了新集合
        """
        try:
            # 检查集合是否存在
            if not self.client.collection_exists(collection_name=self.collection_name):
                # 创建集合
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance,
                    ),
                )
                print(f"集合 {self.collection_name} 创建成功")
                return True
            else:
                print(f"集合 {self.collection_name} 已存在")
                return False
        except Exception as e:
            print(f"创建集合失败: {e}")
            raise

    def create_collection(
            self,
            collection_name: str,
            vector_size: int,
            distance: Distance = Distance.COSINE,
    ) -> bool:
        """
        手动创建新集合

        Args:
            collection_name: 集合名称
            vector_size: 向量维度
            distance: 距离计算方式

        Returns:
            bool: 创建是否成功
        """
        try:
            if self.client.collection_exists(collection_name=collection_name):
                print(f"集合 {collection_name} 已存在，跳过创建")
                return False

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )
            print(f"集合 {collection_name} 创建成功")
            return True
        except Exception as e:
            print(f"创建集合 {collection_name} 失败: {e}")
            raise

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        删除指定集合

        Args:
            collection_name: 集合名称，默认使用初始化的集合名称

        Returns:
            bool: 删除是否成功
        """
        collection_name = collection_name or self.collection_name
        try:
            if self.client.collection_exists(collection_name=collection_name):
                self.client.delete_collection(collection_name=collection_name)
                print(f"集合 {collection_name} 删除成功")
                return True
            else:
                print(f"集合 {collection_name} 不存在")
                return False
        except Exception as e:
            print(f"删除集合 {collection_name} 失败: {e}")
            raise

    def add_vectors(
            self,
            vectors: Union[List[List[float]], np.ndarray],
            payloads: Optional[List[Dict]] = None,
            ids: Optional[List[int]] = None,
            collection_name: Optional[str] = None,
    ) -> List[int]:
        """
        向集合中添加向量

        Args:
            vectors: 向量列表（二维列表或numpy数组）
            payloads: 向量对应的元数据（可选）
            ids: 向量的唯一ID（可选，不指定则自动生成）
            collection_name: 集合名称（默认使用初始化的集合名称）

        Returns:
            List[int]: 添加的向量ID列表
        """
        collection_name = collection_name or self.collection_name
        try:
            # 转换numpy数组为列表
            if isinstance(vectors, np.ndarray):
                vectors = vectors.tolist()

            # 构建点结构列表
            points = []
            for idx, vector in enumerate(vectors):
                point_id = ids[idx] if ids else None
                payload = payloads[idx] if payloads else None
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            # 上传向量
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=points,
            )

            # 获取添加的ID列表
            added_ids = [point.id for point in points] if not ids else ids
            print(f"成功添加 {len(added_ids)} 个向量到集合 {collection_name}")
            return added_ids

        except Exception as e:
            print(f"添加向量失败: {e}")
            raise

    def delete_vectors(
            self,
            ids: List[int],
            collection_name: Optional[str] = None,
    ) -> bool:
        """
        根据ID删除向量

        Args:
            ids: 要删除的向量ID列表
            collection_name: 集合名称（默认使用初始化的集合名称）

        Returns:
            bool: 删除是否成功
        """
        collection_name = collection_name or self.collection_name
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=ids,
            )
            print(f"成功删除 {len(ids)} 个向量")
            return True
        except Exception as e:
            print(f"删除向量失败: {e}")
            raise

    def search_vectors(
            self,
            query_vector: Union[List[float], np.ndarray],
            top_k: int = 10,
            filter_conditions: Optional[Dict] = None,
            with_payload: bool = True,
            with_vector: bool = False,
            collection_name: Optional[str] = None,
            search_params: Optional[SearchParams] = None,
    ) -> List[Dict]:
        """
        纯向量相似性搜索（原生 API，无文本向量化）
        """
        collection_name = collection_name or self.collection_name
        try:
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()

            # 构建过滤条件
            query_filter = None
            if filter_conditions:
                must_conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter_conditions.items()
                ]
                query_filter = Filter(must=must_conditions)

            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=with_payload,
                search_params=search_params or SearchParams(hnsw_ef=100),
                query_filter=query_filter
            )

            # 格式化结果
            formatted = [
                {
                    "id": res.id,
                    "score": res.score,
                    "payload": res.payload,
                    "vector": res.vector if with_vector else None
                }
                for res in results.points
            ]

            print(f"搜索到 {len(formatted)} 个相似向量")
            return formatted

        except ApiException as e:
            raise RuntimeError(f"搜索失败: {e.status} - {e.reason}")
        except Exception as e:
            raise RuntimeError(f"搜索失败: {e}")

    def get_vector(
            self,
            vector_ids: list,
            with_payload: bool = True,
            with_vector: bool = True,
            collection_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        根据ID获取单个向量

        Args:
            vector_id: 向量ID
            with_payload: 是否返回元数据
            with_vector: 是否返回向量本身
            collection_name: 集合名称

        Returns:
            Optional[Dict]: 向量信息，包含id、vector、payload等
        """
        collection_name = collection_name or self.collection_name
        try:
            results = self.client.retrieve(
                collection_name=collection_name,
                ids=vector_ids,
                with_payload=with_payload,
            )

            if results:
                formatted = [
                    {
                        "id": res.id,
                        "payload": res.payload,
                        "vector": res.vector if with_vector else None
                    }
                    for res in results
                ]
                return formatted
            else:
                print(f"未找到ID为 {vector_ids} 的向量")
                return []

        except Exception as e:
            print(f"获取向量失败: {e}")
            raise

    def update_vector_payload(
            self,
            vector_id: int,
            payload: Dict,
            collection_name: Optional[str] = None,
    ) -> bool:
        """
        更新向量的元数据

        Args:
            vector_id: 向量ID
            payload: 新的元数据
            collection_name: 集合名称

        Returns:
            bool: 更新是否成功
        """
        collection_name = collection_name or self.collection_name
        try:
            self.client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[vector_id],
            )
            print(f"成功更新ID为 {vector_id} 的向量元数据")
            return True
        except Exception as e:
            print(f"更新向量元数据失败: {e}")
            raise

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict:
        """
        获取集合的统计信息（终极兼容版：适配所有Qdrant Client版本）

        Args:
            collection_name: 集合名称

        Returns:
            Dict: 集合统计信息（向量数量、配置等）
        """
        collection_name = collection_name or self.collection_name
        try:
            stats = self.client.get_collection(collection_name=collection_name)

            # 1. 兼容集合名称属性：collection_name (新版) / name (旧版)
            col_name = getattr(stats, "collection_name", getattr(stats, "name", collection_name))

            # 2. 兼容向量配置属性：vector (旧版单数) / vectors (新版复数)
            vector_config = None
            if hasattr(stats.config, "vector"):
                # 旧版本：vector 是单个 VectorParams 对象
                vector_config = stats.config.vector
            elif hasattr(stats.config, "vectors"):
                # 新版本：vectors 可能是字典（多向量）或单个对象
                vectors_attr = stats.config.vectors
                if isinstance(vectors_attr, dict):
                    # 多向量配置，取第一个默认向量
                    vector_config = next(iter(vectors_attr.values())) if vectors_attr else None
                else:
                    # 单向量配置
                    vector_config = vectors_attr
            else:
                # 极端情况：无法获取向量配置，使用初始化的默认值
                vector_config = VectorParams(size=self.vector_size, distance=self.distance)

            # 3. 提取向量配置信息（兼容配置为空的情况）
            vec_size = vector_config.size if vector_config else self.vector_size
            vec_distance = vector_config.distance if vector_config else self.distance

            # 4. 兼容统计数量属性
            vector_count = getattr(stats, "points_count", 0)
            indexed_vector_count = getattr(stats, "indexed_points_count", 0)

            return {
                "collection_name": col_name,
                "vector_count": vector_count,
                "indexed_vector_count": indexed_vector_count,
                "vector_size": vec_size,
                "distance": vec_distance,
                "status": getattr(stats, "status", "UNKNOWN"),
            }

        except UnexpectedResponse as e:
            if e.status_code == 404:
                print(f"集合 {collection_name} 不存在")
                return {}
            else:
                raise
        except Exception as e:
            print(f"获取集合统计信息失败: {e}")
            raise

    def close(self):
        """关闭客户端连接（QdrantClient为无状态，此方法主要用于资源清理）"""
        print("Qdrant客户端连接已关闭")
        # QdrantClient的连接是HTTP短连接，无需显式关闭，此处仅作占位
        pass


if __name__ == "__main__":
    # 初始化向量库
    vector_db = QdrantVectorDB(
        host="localhost",
        port=6333,
        collection_name="test_collection",
        vector_size=128,
    )

    # 生成测试向量
    test_vectors = np.random.rand(10, 128).tolist()
    test_payloads = [{"name": f"vector_{i}", "category": "test"} for i in range(10)]
    test_ids = list(range(10))

    # 添加向量
    added_ids = vector_db.add_vectors(
        vectors=test_vectors,
        payloads=test_payloads,
        ids=test_ids,
    )

    # 获取集合统计信息
    stats = vector_db.get_collection_stats()
    print("集合统计信息:", stats)

    # 搜索相似向量
    query_vector = np.random.rand(128).tolist()
    search_results = vector_db.search_vectors(
        query_vector=query_vector,
        top_k=5,
        filter_conditions={"category": "test"},
    )
    print("搜索结果:", search_results)

    # 获取单个向量
    vector_info = vector_db.get_vector(vector_ids=[0])
    print("向量信息:", vector_info)

    # 更新向量元数据
    vector_db.update_vector_payload(
        vector_id=0,
        payload={"name": "updated_vector", "category": "test"},
    )

    # 删除向量
    vector_db.delete_vectors(ids=[0, 1])

    # 删除集合
    vector_db.delete_collection()

    # 关闭客户端
    vector_db.close()
