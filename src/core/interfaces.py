from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunk_id: Optional[str] = None
    created_at: datetime = datetime.now()

class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        pass

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

class DocumentLoader(ABC):
    @abstractmethod
    async def load(self, path: str) -> List[Document]:
        pass

class TextSplitter(ABC):
    @abstractmethod
    def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 800,
        chunk_overlap: int = 80
    ) -> List[Document]:
        pass

class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        pass

class Cache(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        pass