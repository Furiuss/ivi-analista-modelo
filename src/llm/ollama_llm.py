from typing import Optional, Dict, Any
from src.llm.base import BaseLLMProvider
from langchain_community.llms.ollama import Ollama
import logging
import asyncio

logger = logging.getLogger(__name__)


class OllamaLLMProvider(BaseLLMProvider):
    def __init__(
            self,
            model_name: str = "mistral",
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            timeout: int = 120,  # Timeout em segundos
            max_retries: int = 3,
            retry_delay: float = 1.0
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Configurar cliente Ollama
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            num_ctx=4096
        )

    async def _generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            **kwargs
    ) -> str:
        attempts = 0
        last_error = None

        while attempts < self.max_retries:
            try:
                response = await self.llm.ainvoke(
                    prompt,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    **kwargs
                )
                return response

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    f"Timeout on attempt {attempts + 1}/{self.max_retries}. "
                    f"Retrying in {self.retry_delay} seconds..."
                )

            except asyncio.CancelledError:
                logger.error("Operation was cancelled")
                raise

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Error on attempt {attempts + 1}/{self.max_retries}: {str(e)}. "
                    f"Retrying in {self.retry_delay} seconds..."
                )

            attempts += 1
            if attempts < self.max_retries:
                await asyncio.sleep(self.retry_delay)

        logger.error(f"Failed after {self.max_retries} attempts. Last error: {str(last_error)}")
        raise last_error

    async def get_token_count(self, text: str) -> int:
        """
        Implementação específica para o Ollama
        """
        # Estimativa aproximada - pode ser ajustada conforme necessário
        return len(text.split()) * 1.3