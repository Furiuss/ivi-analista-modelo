from typing import Optional
from src.core.interfaces import LLMProvider
from abc import abstractmethod
import logging
import asyncio
from datetime import datetime
import async_timeout

logger = logging.getLogger(__name__)


class BaseLLMProvider(LLMProvider):
    def __init__(
            self,
            model_name: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            timeout: int = 30
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._request_count = 0
        self._last_request_time = None

    async def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            **kwargs
    ) -> str:
        try:
            # Rate limiting
            if self._last_request_time:
                time_since_last = (datetime.now() - self._last_request_time).total_seconds()
                if time_since_last < 1.0:  # Limite de 1 requisição por segundo
                    await asyncio.sleep(1.0 - time_since_last)

            self._last_request_time = datetime.now()
            self._request_count += 1

            # Configurações para esta chamada
            effective_max_tokens = max_tokens or self.max_tokens
            effective_temperature = temperature or self.temperature

            # Timeout wrapper usando async_timeout
            try:
                async with async_timeout.timeout(self.timeout):
                    response = await self._generate(
                        prompt,
                        max_tokens=effective_max_tokens,
                        temperature=effective_temperature,
                        **kwargs
                    )

                return response

            except asyncio.TimeoutError:
                logger.error(f"LLM request timed out after {self.timeout} seconds")
                raise

        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise

    @abstractmethod
    async def _generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: float = 0.7,
            **kwargs
    ) -> str:
        """Implementação específica do provedor de LLM."""
        pass

    async def get_token_count(self, text: str) -> int:
        """Estima o número de tokens no texto."""
        # Implementação básica (pode ser sobrescrita por provedores específicos)
        return len(text.split())