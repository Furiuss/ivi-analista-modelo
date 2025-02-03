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