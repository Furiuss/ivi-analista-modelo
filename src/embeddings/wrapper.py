from typing import List, Union
from src.core.interfaces import EmbeddingProvider
import asyncio


class ChromaEmbeddingWrapper:
    """
    Wrapper para adaptar EmbeddingProvider à interface esperada pelo ChromaDB.
    """

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        Adapta a chamada assíncrona para síncrona, como esperado pelo ChromaDB.
        """
        if isinstance(input, str):
            return asyncio.run(self.embedding_provider.get_embeddings([input]))
        return asyncio.run(self.embedding_provider.get_embeddings(input))
