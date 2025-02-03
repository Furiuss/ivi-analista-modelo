from abc import ABC, abstractmethod
from typing import Optional

class LLMProvider(ABC):
    """Interface para provedores de LLM no domínio"""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Gera texto a partir de um prompt"""
        pass

    @abstractmethod
    async def get_token_count(self, text: str) -> int:
        """Calcula a contagem de tokens para um texto"""
        pass