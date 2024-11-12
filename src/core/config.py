from typing import Optional
from pydantic import BaseModel
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

class VectorStoreSettings(BaseModel):
    provider: str = "chroma"
    persist_directory: str = str(PROJECT_ROOT / "chroma")
    collection_name: str = "default"
    distance_metric: str = "cosine"


class EmbeddingSettings(BaseModel):
    provider: str = "ollama"
    model_name: str = "nomic-embed-text"
    cache_size: int = 1000
    batch_size: int = 32


class LLMSettings(BaseModel):
    provider: str = "ollama"
    model_name: str = "mistral"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30


class CacheSettings(BaseModel):
    type: str = "lru"
    capacity: int = 1000
    default_ttl: int = 3600  # 1 hour


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


class AppConfig(BaseModel):
    vector_store: VectorStoreSettings = VectorStoreSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    llm: LLMSettings = LLMSettings()
    cache: CacheSettings = CacheSettings()
    logging: LoggingSettings = LoggingSettings()

    @classmethod
    def load_from_yaml(cls, path: str) -> 'AppConfig':
        yaml_path = Path(path)
        if not yaml_path.exists():
            return cls()

        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)

            # Converter caminhos relativos em absolutos
            if 'vector_store' in config_dict:
                persist_dir = config_dict['vector_store'].get('persist_directory', 'chroma')
                if not Path(persist_dir).is_absolute():
                    config_dict['vector_store']['persist_directory'] = str(
                        PROJECT_ROOT / persist_dir
                    )

            return cls(**config_dict)