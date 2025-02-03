from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from src.core.interfaces import Document


class VectorStore(ABC):
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        pass