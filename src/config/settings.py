import os
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # LLM 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    # 向量库配置
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "chroma")  # chroma/faiss/pinecone
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "langgraph-demo")
    # 检索配置
    RETRIEVE_TOP_K: int = int(os.getenv("RETRIEVE_TOP_K", 3))


# 全局配置实例
settings = Settings()
