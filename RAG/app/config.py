from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "phi3.5"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_persist_dir: str = "/data/chroma"
    documents_dir: str = "/data/documents"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
