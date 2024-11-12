from abc import abstractmethod
from typing import List, Optional
from src.core.interfaces import EmbeddingProvider
from functools import lru_cache


class BaseEmbeddingProvider(EmbeddingProvider):
    def __init__(self, cache_size: int = 1000):
        self.dimension: Optional[int] = None

    @lru_cache(maxsize=1000)
    async def get_embedding(self, text: str) -> List[float]:
        return await self._generate_embedding(text)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [await self.get_embedding(text) for text in texts]

    @abstractmethod
    async def _generate_embedding(self, text: str) -> List[float]:
        pass