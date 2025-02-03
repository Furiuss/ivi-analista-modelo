from abc import abstractmethod
from typing import Optional
import logging
from src.domain.llm.llm_interface import LLMProvider
from datetime import datetime
import asyncio
import async_timeout


logger = logging.getLogger(__name__)


class BaseLLMProvider(LLMProvider):

    def __init__(
            self,
            model_name: str,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            timeout: int = 30,
            max_retries: int = 3,
            retry_delay: float = 1.0
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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
            if self._last_request_time:
                time_since_last = (datetime.now() - self._last_request_time).total_seconds()
                if time_since_last < 1.0:
                    await asyncio.sleep(1.0 - time_since_last)

            self._last_request_time = datetime.now()
            self._request_count += 1

            effective_max_tokens = max_tokens or self.max_tokens
            effective_temperature = temperature or self.temperature

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
    async def _generate_implementation(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            **kwargs
    ) -> str:
        pass

    async def get_token_count(self, text: str) -> int:
        return len(text.split())