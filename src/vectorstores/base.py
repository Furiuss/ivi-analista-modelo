from typing import List, Dict, Any, Optional, Tuple
from src.core.interfaces import VectorStore, Document, EmbeddingProvider
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseVectorStore(VectorStore):
    def __init__(
            self,
            embedding_provider: EmbeddingProvider,
            distance_metric: str = "cosine"
    ):
        self.embedding_provider = embedding_provider
        self.distance_metric = distance_metric

    async def similarity_search(
            self,
            query: str,
            k: int = 5,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Realiza busca por similaridade com suporte a filtros.
        Retorna lista de tuplas (documento, score).
        """
        try:
            query_embedding = await self.embedding_provider.get_embedding(query)

            results = await self._similarity_search_by_vector(
                query_embedding,
                k=k,
                filter=filter
            )

            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    @abstractmethod
    async def _similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int = 5,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Implementação específica da busca por vetor de embedding."""
        pass

    async def add_documents(
            self,
            documents: List[Document],
            batch_size: int = 32
    ) -> List[str]:
        """
        Adiciona documentos em batch para melhor performance.
        """
        try:
            # Processa em batches
            all_ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # Gera embeddings para documentos que não têm
                texts_to_embed = [
                    doc.content for doc in batch
                    if doc.embedding is None
                ]

                if texts_to_embed:
                    embeddings = await self.embedding_provider.get_embeddings(
                        texts_to_embed
                    )

                    # Atualiza documentos com embeddings
                    embedding_idx = 0
                    for doc in batch:
                        if doc.embedding is None:
                            doc.embedding = embeddings[embedding_idx]
                            embedding_idx += 1

                # Adiciona batch ao store
                batch_ids = await self._add_documents_batch(batch)
                all_ids.extend(batch_ids)

            return all_ids

        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    @abstractmethod
    async def _add_documents_batch(
            self,
            documents: List[Document]
    ) -> List[str]:
        """Implementação específica da adição em batch."""
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> None:
        """Remove documentos por ID."""
        pass