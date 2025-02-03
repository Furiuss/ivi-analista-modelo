from src.domain.cache.cache_interface import BaseCache
from typing import Optional, Any
from datetime import datetime, timedelta
from collections import OrderedDict
import time



class LRUCache(BaseCache[Any]):
    """
        Implementação de um cache LRU (Least Recently Used) com suporte a TTL.
        """

    def __init__(self, capacity: int = 1000):
        """
        Inicializa o cache LRU.

        Args:
            capacity: Capacidade máxima do cache
        """
        super().__init__()
        self.capacity = capacity
        self.cache = OrderedDict()
        self.ttls = {}
        self._expiry = {}

    async def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None

        # Verifica TTL
        if key in self.ttls and time.time() > self.ttls[key]:
            await self.delete(key)
            return None

        # Move para o fim (mais recentemente usado)
        self.cache.move_to_end(key)
        return self.cache[key]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        if ttl:
            self.ttls[key] = time.time() + ttl
            self._expiry[key] = datetime.now() + timedelta(seconds=ttl)

        # Remove o item menos recentemente usado se exceder a capacidade
        if len(self.cache) > self.capacity:
            oldest = next(iter(self.cache))
            await self.delete(oldest)

    async def delete(self, key: str) -> None:
        self.cache.pop(key, None)
        self.ttls.pop(key, None)
        self._expiry.pop(key, None)

    async def clear(self) -> None:
        await super().clear()
        self.cache.clear()
        self.ttls.clear()