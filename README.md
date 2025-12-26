# agent_init

langgraph图调用简单原型

已完成：
  日志配置：日志打印定长，显示全部以及其他配置
  知识库解析成md, chunk切分, chunk提取qa

文件：
- emb_start 手动拉起一个小模型服务
- src 代码路径
  - utils
    - customModel
      - CustomModel  langgraph自定义模型服务
      - CustomEmbedding langgraph自定义embedding模型服务
    - logger
    - office_to_pdf 使用libreoffice将office文档转为pdf
    - vector_db  向量库qdrantdb
    - utils
      - unzip_file 解压zip
  - knowledge
    - converted 放置转化结果
    - files 待转化的文件
    - tmp office转pdf的结果
    - chunk chunk分割
      - 基于Token Size的分割  
      - 基于Markdown递归分割 
      - 基于语义的分割结果
    - convert mineru转化 各种文档转markdown, 暂支持 pdf, office, png等图片, zip
  - knowledge.py 知识库处理，文档转md -> chunk切分 -> qa提取 -> embedding -> 入库 -> 向量库查询

todo
- 基于langgraph的代码优化助手
  - 上下文管理，压缩，相关度
  - 工作流
  - sql工具使用
  - 知识库rerank
- 模块分离
  - config/ 配置
  - state/ state结构，schema
  - tools/ 工具定义，工具初始化
  - nodes/ 节点定义
  - graph/ 图定义，组装节点
  - logs/ 日志
  - knowledge/ 知识库处理

  

## mineru安装
参考 https://github.com/opendatalab/MinerU


pip install uv -i https://mirrors.aliyun.com/pypi/simple
uv pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple 

配置torch

- pip uninstall pytorch
- pip uninstall torchvision
- pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

模型下载位置 C:\Users\admin\.cache\modelscope

mineru.exe -p '.\0. 树莓派5新手入门手册.pdf' -o . --source modelscope

支持文件
```python
pdf_suffixes = ['.pdf']
office_suffixes = ['.ppt', '.pptx', '.doc', '.docx', 'xls', 'xlsx'] # 需要安装libreoffice【win, linux】, doc.spire【win】
image_suffixes = ['.png', '.jpeg', '.jpg']
compress = ['.zip', '.rar']

```

## 向量库

### qdrant
```shell
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/qdrant/qdrant:v1.16.2
```

### milvus

#### 可视化
```shell
docker run -d --name attu -p 8000:3000 -e MILVUS_URL={milvus server IP}:19530 zilliz/attu:latest

```


#### 部署
docker-compose up
 ```
version: '3.5'
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.18
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ./volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ./volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.6.4
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
    networks:
      - milvus-tier

networks:
  milvus-tier:
    name: milvus-tier
    driver: bridge

```