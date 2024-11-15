from dataclasses import dataclass
from typing import List, Optional
import asyncio
import argparse
import langdetect
from src.core.config import AppConfig
from src.embeddings.ollama import OllamaEmbeddingProvider
from src.utils.prompt_templates import PromptTemplates
from src.vectorstores.chroma_store import ChromaVectorStore
from src.llm.ollama_llm import OllamaLLMProvider
from src.cache.lru_cache import LRUCache
from src.utils.logging import setup_logger


@dataclass
class SearchResult:
    """Classe para armazenar resultados da busca"""
    context_texts: List[str]
    sources: List[str]

    def get_formatted_context(self) -> str:
        return "\n\n---\n\n".join(self.context_texts)

    def get_formatted_sources(self, language: str) -> str:

        header = 'Fontes consultadas' if language == 'pt' else 'Fuentes consultadas'
        return (f"{header}:\n{'-' * 40}\n"
                f"{chr(10).join('- ' + src for src in self.sources)}")


class QueryProcessor:
    """Classe principal para processamento de queries no sistema RAG"""

    def __init__(self, config_path: str):
        self.config = AppConfig.load_from_yaml(config_path)
        self.logger = setup_logger("query", level=self.config.logging.level)
        self.cache = LRUCache(capacity=self.config.cache.capacity)

        # Inicialização lazy dos providers
        self._embedding_provider = None
        self._vector_store = None
        self._llm_provider = None

    @property
    def embedding_provider(self):
        if self._embedding_provider is None:
            self._embedding_provider = OllamaEmbeddingProvider(
                model=self.config.embeddings.model_name
            )
        return self._embedding_provider

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore(
                persist_directory=self.config.vector_store.persist_directory,
                embedding_provider=self.embedding_provider,
                collection_name=self.config.vector_store.collection_name
            )
        return self._vector_store

    @property
    def llm_provider(self):
        if self._llm_provider is None:
            self._llm_provider = OllamaLLMProvider(
                model_name=self.config.llm.model_name,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout=180,
                max_retries=3
            )
        return self._llm_provider

    async def detect_language(self, query_text: str, forced_language: Optional[str] = None) -> str:
        """Detecta ou valida o idioma da query"""
        if forced_language:
            return forced_language

        try:
            detected = langdetect.detect(query_text)
            return 'pt' if detected not in ['pt', 'es'] else detected
        except:
            return 'pt'

    async def check_cache(self, query_text: str) -> Optional[str]:
        """Verifica se a resposta está em cache"""
        cached_response = await self.cache.get(query_text)
        if cached_response:
            self.logger.info("Cache hit")
            return cached_response
        return None

    async def search_documents(
            self,
            query_text: str,
            similarity_threshold: float
    ) -> Optional[SearchResult]:
        """Busca documentos relevantes no vector store"""
        results = await self.vector_store.similarity_search(query_text, k=5)

        filtered_results = [
            (doc, score) for doc, score in results
            if score >= float(similarity_threshold)
        ]

        if not filtered_results:
            return None

        context_texts = []
        sources = []

        for doc, score in filtered_results:
            context_texts.append(doc.content)
            title = doc.metadata.get('title', 'Título Desconhecido')
            doc_id = doc.metadata.get('doc_id', 'ID Desconhecido')
            source = f"{title} (ID: {doc_id}, Score: {score:.3f})"
            if source not in sources:
                sources.append(source)

        self.logger.info(f"Found {len(results)} relevant documents")
        return SearchResult(context_texts=context_texts, sources=sources)

    async def generate_response(
            self,
            prompt: str,
            system_prompt: str
    ) -> Optional[str]:
        """Gera resposta usando o LLM"""
        try:
            return await self.llm_provider.generate(
                f"{system_prompt}\n\n{prompt}"
            )
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return None

    def format_response(
            self,
            response: str,
            search_result: SearchResult,
            language: str
    ) -> str:
        """Formata a resposta final com as fontes"""
        return (
            f"{'Resposta' if language == 'pt' else 'Respuesta'}:\n"
            f"{response}\n\n"
            f"{search_result.get_formatted_sources(language)}"
        )

    async def process_query(
            self,
            query_text: str,
            language: Optional[str] = None,
            similarity_threshold: float = 0.004
    ) -> str:
        """Metodo principal para processar uma query"""
        try:
            # Verificar cache
            cached_response = await self.check_cache(query_text)
            if cached_response:
                return cached_response

            # Detectar idioma
            language = await self.detect_language(query_text, language)
            self.logger.info(f"Query language detected: {language}")

            # Buscar documentos
            search_result = await self.search_documents(query_text, similarity_threshold)
            if not search_result:
                return (
                    "Não encontrei documentos suficientemente relevantes para sua pergunta. "
                    "Por favor, reformule a pergunta ou busque por outro tópico."
                )

            # Preparar prompts
            template = PromptTemplates.get_template(language)
            prompt = template.format(
                context=search_result.get_formatted_context(),
                question=query_text
            )
            system_prompt = PromptTemplates.get_system_prompt()

            # Gerar resposta
            response = await self.generate_response(prompt, system_prompt)
            if not response:
                return "Desculpe, ocorreu um erro ao gerar a resposta. Por favor, tente novamente."

            # Formatar resposta final
            formatted_response = self.format_response(response, search_result, language)

            # Salvar no cache
            await self.cache.set(
                query_text,
                formatted_response,
                ttl=self.config.cache.default_ttl
            )

            return formatted_response

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"Ocorreu um erro ao processar sua pergunta: {str(e)}"


def main():
    """Função principal para execução via linha de comando"""
    parser = argparse.ArgumentParser(description="Query RAG system")
    parser.add_argument(
        "query_text",
        help="The question to ask"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--language",
        choices=['pt', 'es'],
        help="Force specific language"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.004,
        help="Force specific threshold"
    )

    args = parser.parse_args()

    try:
        processor = QueryProcessor(args.config)
        response = asyncio.run(
            processor.process_query(
                args.query_text,
                args.language,
                args.similarity_threshold
            )
        )
        print(response)
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário.")
    except Exception as e:
        print(f"Erro inesperado: {str(e)}")


if __name__ == "__main__":
    main()