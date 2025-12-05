import json
import os
import re
import sqlite3
from loguru import logger
from typing import Optional, List, Any, Mapping, Dict

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.embeddings import Embeddings

from sentence_transformers import SentenceTransformer


def load_config():
    with open('config.json', 'r') as f:
        for k, v in json.load(f).items():
            os.environ[k] = v


class CustomModel(LLM):
    """自定义模型服务集成"""
    api_url: str
    model_name: str = "custom-model"
    temperature: float = 0.0
    max_tokens: int = 1000

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """调用自定义模型服务"""
        try:
            payload = {
                'model': self.model_name,
                'messages': [
                    {'role': 'system', 'content': prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stop": stop
            }
            response = requests.post(
                self.api_url + '/v1/chat/completions',
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            logger.debug(f'{prompt=}')
            response.raise_for_status()
            data = re.sub(r'<think>[\s\S\s]*</think>', '', response.json()['choices'][-1]['message']['content'], re.DOTALL).strip()
            logger.debug(f'{data=}')
            return data

        except Exception as e:
            return f"模型调用错误: {str(e)}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """返回识别参数"""
        return {
            "api_url": self.api_url,
            "model_name": self.model_name,
        }


class CustomEmbedding(Embeddings):
    """
    自定义 Embedding 模型（适配 Chroma 要求）
    支持：本地模型（sentence-transformers）、远程 API 模型、自研模型
    """

    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",  # 轻量级中文/英文通用模型
            use_local: bool = True,
            api_url: Optional[str] = None,  # 远程模型服务地址（如 FastAPI 部署）
            device: str = "cpu"  # cpu/cuda
    ):
        self.model_name = model_name
        self.use_local = use_local
        self.api_url = api_url
        self.device = device

        # 初始化本地 Embedding 模型
        if self.use_local:
            self.model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device
            )
            print(f"✅ 本地 Embedding 模型加载完成: {self.model_name}")
        elif self.api_url:
            print(f"✅ 远程 Embedding 服务配置完成: {self.api_url}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本嵌入（Chroma 用于文档向量生成）
        :param texts: 文本列表
        :return: 向量列表（每个文本对应一个 float 向量）
        """
        if self.use_local:
            # 本地模型生成向量（转为 list 避免 numpy 类型问题）
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,  # 归一化提升检索效果
                convert_to_numpy=True
            ).tolist()
        else:
            # 调用远程 Embedding API（适配自定义模型服务）
            import requests
            response = requests.post(
                url=f"{self.api_url}/emb",
                json={"texts": texts},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            embeddings = response.json()["embeddings"]

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        生成单条查询文本的嵌入（Chroma 用于检索匹配）
        :param text: 查询文本
        :return: 单个向量
        """
        return self.embed_documents([text])[0]


class SQLitePriceDB:
    def __init__(self, db_path: str = "./price_database.db"):
        """初始化SQLite数据库"""
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 启用行工厂，支持列名访问
        return conn

    def init_database(self):
        """初始化数据库和表结构"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # 创建价格表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                specification TEXT DEFAULT '标准版',
                price REAL NOT NULL,
                description TEXT DEFAULT '无',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_product_name ON price_table(product_name)
        ''')

        # 插入示例数据（如果表为空）
        cursor.execute('SELECT COUNT(*) FROM price_table')
        if cursor.fetchone()[0] == 0:
            sample_data = [
                ('产品A', '标准版', 99.00, '基础功能，适合个人使用'),
                ('产品A', '高级版', 199.00, '高级功能，适合团队使用'),
                ('产品B', '标准版', 149.00, '基础功能套餐'),
                ('产品B', '专业版', 299.00, '专业功能套餐'),
                ('产品C', '企业版', 499.00, '企业级解决方案'),
                ('产品D', '旗舰版', 999.00, '旗舰级全功能版本')
            ]
            cursor.executemany('''
                INSERT INTO price_table (product_name, specification, price, description)
                VALUES (?, ?, ?, ?)
            ''', sample_data)

        conn.commit()
        conn.close()

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行SQL查询并返回结果"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # 获取列名
            columns = [description[0] for description in cursor.description] if cursor.description else []
            # 将结果转换为字典列表
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            conn.close()
            return results

        except sqlite3.Error as e:
            return [{"error": f"SQL错误: {str(e)}"}]

    def execute_update(self, query: str, params: tuple = None) -> str:
        """执行更新操作（INSERT/UPDATE/DELETE）"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            conn.commit()
            affected_rows = cursor.rowcount
            conn.close()

            return f"操作成功，影响行数: {affected_rows}"

        except sqlite3.Error as e:
            return f"SQL错误: {str(e)}"


if __name__ == '__main__':
    api_url = 'http://192.168.153.1:5050'
    model_name = 'qwen/qwen3-8b'
    llm = CustomModel(api_url=api_url, model_name=model_name)
    result = llm.invoke('你是谁')
    print(result)

    emb = CustomEmbedding(model_name='Qwen3-Embedding-8B', use_local=False, api_url='http://127.0.0.1:5000')
    result = emb.embed_query('你是谁')
    print(result)
