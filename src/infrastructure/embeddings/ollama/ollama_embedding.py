from src.infrastructure.embeddings.shared.base_embedding import BaseEmbeddingProvider
from langchain_community.embeddings.ollama import OllamaEmbeddings
from typing import List


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str = "nomic-embed-text"):
        super().__init__()
        self.embeddings = OllamaEmbeddings(model=model)

    async def _generate_embedding(self, text: str) -> List[float]:
        return await self.embeddings.aembed_query(text)