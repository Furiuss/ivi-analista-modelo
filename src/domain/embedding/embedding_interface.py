from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        pass