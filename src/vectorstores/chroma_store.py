from src.core.interfaces import Document, EmbeddingProvider
from src.vectorstores.base import BaseVectorStore
from src.embeddings.wrapper import ChromaEmbeddingWrapper
import chromadb
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import uuid


logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    def __init__(
            self,
            persist_directory: str,
            embedding_provider: EmbeddingProvider,
            collection_name: str = "default",
            distance_metric: str = "cosine"
    ):
        # Chamar o construtor da classe base primeiro
        super().__init__(
            embedding_provider=embedding_provider,
            distance_metric=distance_metric
        )

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Garantir que o diretório existe
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Criar wrapper para o embedding provider
        self.embedding_wrapper = ChromaEmbeddingWrapper(embedding_provider)

        # Inicializar cliente e coleção
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_wrapper
        )

        logger.info(
            f"Initialized ChromaVectorStore at {persist_directory} "
            f"with collection {collection_name}"
        )

    async def _similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int = 5,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=filter,
                include=["documents", "metadatas", "distances"]
            )

            documents_with_scores = []
            for i in range(len(results['ids'][0])):
                doc = Document(
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    chunk_id=results['ids'][0][i]
                )
                # Converter distância em score de similaridade (0 a 1)
                distance = float(results['distances'][0][i])
                similarity_score = 1 / (1 + distance)  # Normalização

                documents_with_scores.append((doc, similarity_score))

            # Ordenar por score decrescente
            documents_with_scores.sort(key=lambda x: x[1], reverse=True)

            return documents_with_scores

        except Exception as e:
            logger.error(f"Error in similarity search by vector: {str(e)}")
            raise

    async def _add_documents_batch(
            self,
            documents: List[Document]
    ) -> List[str]:
        try:
            # Validar e preparar dados
            embeddings = []
            texts = []
            metadatas = []
            ids = []

            for doc in documents:
                if not doc.chunk_id:
                    doc.chunk_id = f"doc_{uuid.uuid4()}"

                embeddings.append(doc.embedding)
                texts.append(doc.content)
                metadatas.append(doc.metadata)
                ids.append(str(doc.chunk_id))  # Garantir que é string

            # Validar que temos todos os dados necessários
            if not all(ids) or len(ids) != len(documents):
                raise ValueError("Invalid or missing document IDs")

            # Adicionar ao Chroma
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(documents)} documents to collection")
            return ids

        except Exception as e:
            logger.error(f"Error adding document batch: {str(e)}")
            raise

    async def delete(self, ids: List[str]) -> None:
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from collection")

        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise

    async def clear(self) -> None:
        """Limpa toda a coleção e reinicializa"""
        try:
            logger.info(f"Clearing collection {self.collection_name}")

            # Deletar coleção existente
            try:
                self.client.delete_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"Error deleting collection: {str(e)}")

            # Recriar coleção
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_wrapper
            )

            # Reinicializar métricas da classe base
            # super()._initialize_metrics()

            logger.info("Collection cleared and reinitialized")

        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise

    def __repr__(self) -> str:
        return (
            f"ChromaVectorStore(directory='{self.persist_directory}', "
            f"collection='{self.collection_name}', "
            f"total_documents={self.total_documents})"
        )