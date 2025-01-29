"""
Sistema de Consulta RAG Interativo com cache e detecção de idioma
"""

import sys
import json
import asyncio
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict

import langdetect
from langdetect import LangDetectException

from src.core.config import AppConfig
from src.embeddings.ollama import OllamaEmbeddingProvider
from src.vectorstores.chroma_store import ChromaVectorStore
from src.llm.ollama_llm import OllamaLLMProvider
from src.cache.lru_cache import LRUCache
from src.utils.logging import setup_logger
from src.utils.prompt_templates import PromptTemplates, RoleEnum


@dataclass(frozen=True)
class SearchResult:
    """Resultado de uma busca no repositório de documentos"""
    context_texts: List[str]
    sources: List[str]

    def get_formatted_context(self) -> str:
        """Formata o contexto para inclusão no prompt"""
        return "\n\n---\n\n".join(self.context_texts)

    def get_formatted_sources(self, language: str) -> str:
        """Formata as fontes para exibição ao usuário"""
        headers = {
            'pt': 'Fontes consultadas',
            'es': 'Fuentes consultadas'
        }
        sources_list = chr(10).join(f'- {src}' for src in self.sources)
        return (
            f"{headers.get(language, 'Fontes')}:\n"
            f"{'-' * 40}\n"
            f"{sources_list}"
        )


class QueryProcessor:
    """Processador de consultas utilizando arquitetura RAG"""

    def __init__(self, config_path: str):
        self.config = AppConfig.load_from_yaml(config_path)
        self.logger = setup_logger("query_processor", level=self.config.logging.level)
        self.cache = LRUCache(capacity=self.config.cache.capacity)

        self._embedding_provider: Optional[OllamaEmbeddingProvider] = None
        self._vector_store: Optional[ChromaVectorStore] = None
        self._llm_provider: Optional[OllamaLLMProvider] = None

    @property
    def embedding_provider(self) -> OllamaEmbeddingProvider:
        """Provider de embeddings (inicialização lazy)"""
        if self._embedding_provider is None:
            self._embedding_provider = OllamaEmbeddingProvider(
                model=self.config.embeddings.model_name
            )
        return self._embedding_provider

    @property
    def vector_store(self) -> ChromaVectorStore:
        """Vector store (inicialização lazy)"""
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore(
                persist_directory=self.config.vector_store.persist_directory,
                embedding_provider=self.embedding_provider,
                collection_name=self.config.vector_store.collection_name
            )
        return self._vector_store

    @property
    def llm_provider(self) -> OllamaLLMProvider:
        """Provider do modelo LLM (inicialização lazy)"""
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
        """Detecta o idioma da consulta usando lib externa"""
        if forced_language:
            return forced_language

        try:
            detected = langdetect.detect(query_text)
            return 'pt' if detected not in ['pt', 'es'] else detected
        except LangDetectException as e:
            self.logger.warning(f"Erro na detecção de idioma: {e}. Usando 'pt' como fallback")
            return 'pt'

    async def _get_cached_response(self, query_text: str) -> Optional[str]:
        """Verifica o cache por uma resposta existente"""
        cached = await self.cache.get(query_text)
        if cached:
            self.logger.debug(f"Cache hit para query: {query_text[:50]}...")
            return cached
        return None

    async def _search_relevant_documents(self, query_text: str, threshold: float) -> Optional[SearchResult]:
        """Executa a busca por documentos relevantes"""
        results = await self.vector_store.similarity_search(query_text, k=5)

        filtered = [
            (doc, score)
            for doc, score in results
            if score >= threshold
        ]

        if not filtered:
            return None

        context_texts = []
        sources = []
        seen_sources = set()

        for doc, score in filtered:
            context_texts.append(doc.content)
            source_info = (
                f"{doc.metadata.get('title', 'Sem título')} "
                f"(ID: {doc.metadata.get('doc_id', 'N/A')}, "
                f"Score: {score:.3f})"
            )
            if source_info not in seen_sources:
                seen_sources.add(source_info)
                sources.append(source_info)

        self.logger.info(f"Documentos encontrados: {len(filtered)}/{len(results)}")
        return SearchResult(context_texts, sources)

    def _build_prompts(self, context: str, query: str) -> Dict[str, str]:
        """Constrói os prompts para o LLM"""
        user_context = {
            "phone": "62985761305",
            "name": "Messias",
            "client_id": 22
        }

        return {
            "system": PromptTemplates.get_prompt(RoleEnum.System),
            "user": PromptTemplates.get_prompt(RoleEnum.User).format(
                context=context,
                question=query,
                **user_context
            )
        }

    async def _generate_llm_response(self, prompts: Dict[str, str]) -> Optional[str]:
        """Gera a resposta através do LLM"""
        try:
            full_prompt = f"{prompts['system']}\n\n{prompts['user']}"
            return await self.llm_provider.generate(full_prompt)
        except Exception as e:
            self.logger.error(f"Falha na geração da resposta: {str(e)}")
            return None

    def _format_final_response(self, response: str, sources: SearchResult, language: str) -> str:
        """Formata a resposta final para exibição"""
        response_header = "Resposta" if language == 'pt' else "Respuesta"
        return (
            f"{response_header}:\n{response}\n\n"
            f"{sources.get_formatted_sources(language)}"
        )

    async def process_query(
            self,
            query_text: str,
            language: Optional[str] = None,
            similarity_threshold: float = 0.004
    ) -> str:
        """Fluxo principal de processamento de consultas"""
        self.logger.info(f"Processando consulta: {query_text[:50]}...")

        try:
            # Verificação de cache
            if cached := await self._get_cached_response(query_text):
                return cached

            # Detecção de idioma
            language = await self.detect_language(query_text, language)
            self.logger.debug(f"Idioma detectado: {language}")

            # Busca de documentos
            search_result = await self._search_relevant_documents(query_text, similarity_threshold)
            if not search_result:
                self.logger.warning("Nenhum documento relevante encontrado")
                return (
                    "Não encontrei documentos relevantes para sua pergunta. "
                    "Por favor, reformule a consulta."
                )

            # Construção e execução dos prompts
            prompts = self._build_prompts(
                search_result.get_formatted_context(),
                query_text
            )

            llm_response = await self._generate_llm_response(prompts)
            if not llm_response:
                return "Erro ao gerar resposta. Tente novamente mais tarde."

            # Formatação e cache
            formatted = self._format_final_response(llm_response, search_result, language)
            await self.cache.set(
                key=query_text,
                value=formatted,
                ttl=self.config.cache.default_ttl
            )

            return formatted

        except Exception as e:
            self.logger.exception(f"Erro crítico ao processar consulta: {str(e)}")
            return f"Erro interno: {str(e)}"


def run_interactive_session(processor: QueryProcessor, args: argparse.Namespace):
    """Executa sessão interativa de consultas via CLI"""
    print("\n🤖 Sistema de Consulta RAG Interativo")
    print("Digite 'sair' ou pressione Ctrl+C para encerrar\n")

    try:
        while True:
            try:
                query = input("👉 Sua pergunta: ").strip()

                if query.lower() in ('sair', 'exit', 'quit'):
                    print("\n👋 Até logo!")
                    break

                if not query:
                    continue

                response = asyncio.run(
                    processor.process_query(
                        query_text=query,
                        language=args.language,
                        similarity_threshold=args.similarity_threshold
                    )
                )

                print(f"\n{response}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Encerrando sessão...")
                break

    except Exception as e:
        print(f"⚠️ Erro inesperado: {str(e)}")
        sys.exit(1)


def main():
    """Ponto de entrada principal da aplicação"""
    parser = argparse.ArgumentParser(
        description="Sistema de Consulta RAG Interativo"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Caminho para o arquivo de configuração"
    )
    parser.add_argument(
        "--language",
        choices=['pt', 'es'],
        help="Forçar idioma específico (pt/es)"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.004,
        help="Limiar de similaridade para documentos"
    )

    args = parser.parse_args()

    try:
        processor = QueryProcessor(args.config)
        run_interactive_session(processor, args)
    except FileNotFoundError:
        print(f"❌ Arquivo de configuração não encontrado: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Falha na inicialização: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()