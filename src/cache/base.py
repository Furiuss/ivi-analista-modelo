from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
from datetime import datetime

T = TypeVar('T')


class BaseCache(ABC, Generic[T]):
    """
    Classe base abstrata que define a interface e implementa funcionalidades básicas
    de cache com suporte a TTL e tipos genéricos.
    """

    def __init__(self):
        self._storage = {}
        self._expiry = {}

    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """
        Recupera um valor do cache.

        Args:
            key: Chave para buscar no cache

        Returns:
            O valor armazenado ou None se não encontrado ou expirado
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """
        Armazena um valor no cache.

        Args:
            key: Chave para armazenar o valor
            value: Valor a ser armazenado
            ttl: Tempo de vida em segundos (opcional)
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Remove um valor do cache.

        Args:
            key: Chave a ser removida
        """
        pass

    async def clear(self) -> None:
        """Limpa todo o cache."""
        self._storage.clear()
        self._expiry.clear()

    def _is_expired(self, key: str) -> bool:
        """
        Verifica se uma chave está expirada.

        Args:
            key: Chave a ser verificada

        Returns:
            True se a chave estiver expirada, False caso contrário
        """
        return key in self._expiry and datetime.now() > self._expiry[key]
