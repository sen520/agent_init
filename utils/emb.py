import numpy as np
from qdrant_client import AsyncQdrantClient
import asyncio
from qdrant_client import QdrantClient

from grpc import StatusCode
from qdrant_client.grpc import PointId
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import ApiException
import time
from functools import lru_cache
import threading
import random
import os


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    获取Qdrant同步客户端（带连接池优化）
    """
    # 配置gRPC连接池参数
    grpc_options = {
        # 示例：设置最大接收消息大小为 200MB（解决大向量传输报错）
        "grpc.max_receive_message_length": 200 * 1024 * 1024,
        # 示例：设置最大发送消息大小为 200MB
        "grpc.max_send_message_length": 200 * 1024 * 1024,
        # 示例：设置连接超时（可选）
        "grpc.connect_timeout_ms": 5000,
    }

    # 创建客户端
    client = QdrantClient(
        host="localhost",  # Qdrant服务地址（集群可填多个，用逗号分隔）
        port=6333,  # gRPC端口
        grpc_options=grpc_options,
    )

    # 验证连接
    try:
        client.get_collection(collection_name="test")
        print("Qdrant客户端连接成功！")
    except ApiException as e:
        client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    return client


async def get_async_qdrant_client() -> AsyncQdrantClient:
    """
    获取Qdrant异步客户端（带连接池优化）
    """
    client = AsyncQdrantClient(
        host="localhost",
        port=6333,
        grpc_options={
            "grpc.max_concurrent_streams": 1000,
            "grpc.keepalive_time_ms": 30 * 1000,
        },
    )
    try:
        await client.get_collection(collection_name="test")
    except:
        await client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    # 验证连接
    try:
        await client.get_collection(collection_name="test")
        print("异步Qdrant客户端连接成功！")
    except Exception as e:
        print(f"异步连接失败：{e}")
        raise

    return client


def query_vector(client: QdrantClient, thread_id: int):
    """
    模拟单线程的向量查询请求
    """
    try:
        # 生成随机向量（768维，与集合配置一致）
        random_vector = [random.random() for _ in range(4)]
        # 执行查询
        start = time.time()
        response = client.query_points(
            collection_name="test_collection",
            query=random_vector,
            limit=10,
        )
        end = time.time()
        print(f"线程{thread_id}查询成功，耗时：{(end - start) * 1000:.2f}ms")
    except Exception as e:
        print(f"线程{thread_id}查询失败：{e}")


async def async_query_vector(client: AsyncQdrantClient, task_id: int):
    """
    异步执行向量查询
    """
    try:
        random_vector = [random.random() for _ in range(128)]
        start = time.time()
        response = await client.query_points(
            collection_name="test_collection",
            query=random_vector,
            limit=10,
        )
        end = time.time()
        print(f"任务{task_id}查询成功，耗时：{(end - start) * 1000:.2f}ms")
    except Exception as e:
        print(f"任务{task_id}查询失败：{e}")


async def main():
    """
    异步主函数：模拟1000个并发请求
    """
    # 获取异步客户端
    client = await get_async_qdrant_client()

    # 创建1000个协程任务
    tasks = [async_query_vector(client, i) for i in range(1000)]

    # 批量执行协程（控制并发数，避免压垮服务器）
    # 分批次执行，每批100个
    batch_size = 100
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*batch)
        print(f"批次{i // batch_size + 1}执行完成")


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"所有任务执行完成，总耗时：{end_time - start_time:.2f}s")

    # print('---------------------------------------')
    # client = get_qdrant_client()
    #
    # # 模拟100个并发请求（多线程）
    # threads = []
    # for i in range(100):
    #     t = threading.Thread(target=query_vector, args=(client, i))
    #     threads.append(t)
    #     t.start()
    #
    # # 等待所有线程完成
    # for t in threads:
    #     t.join()
    #
    # print("所有并发请求执行完成！")
