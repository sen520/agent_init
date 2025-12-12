import json
import os
import re
from loguru import logger
from typing import Optional, List, Any, Mapping

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
            data = re.sub(r'<think>[\s\S\s]*</think>', '', response.json()['choices'][-1]['message']['content'],
                          re.DOTALL).strip()
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


if __name__ == '__main__':
    api_url = 'http://192.168.153.1:5050'
    model_name = 'qwen/qwen3-8b'
    llm = CustomModel(api_url=api_url, model_name=model_name)
    result = llm.invoke('你是谁')
    print(result)

    emb = CustomEmbedding(model_name='Qwen3-Embedding-8B', use_local=False, api_url='http://127.0.0.1:5000')
    result = emb.embed_query('你是谁')
    print(result)
