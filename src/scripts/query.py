import asyncio
import argparse
from src.core.config import AppConfig
from src.embeddings.ollama import OllamaEmbeddingProvider
from src.vectorstores.chroma_store import ChromaVectorStore
from src.llm.ollama_llm import OllamaLLMProvider
from src.cache.lru_cache import LRUCache
from src.utils.logging import setup_logger
import langdetect

PROMPT_TEMPLATE_PT = """
Por favor, responda à pergunta baseando-se EXCLUSIVAMENTE no seguinte contexto:

{context}

---

Regras OBRIGATÓRIAS:
1. Use APENAS as informações fornecidas no contexto acima
2. Se a informação NÃO estiver EXPLICITAMENTE no contexto, responda "Não encontrei informações sobre isso no contexto fornecido"
3. NÃO FAÇA SUPOSIÇÕES ou adicione informações externas
4. Se o contexto não for relacionado à pergunta, responda "O contexto fornecido não contém informações relacionadas a esta pergunta"
5. Mantenha a resposta focada apenas no que está documentado
6. NÃO RESPONDA do que o contexto fornecido se trata, apenas que a pergunta não está nele

Pergunta: {question}
"""

PROMPT_TEMPLATE_ES = """
Por favor, responde a la pregunta basándote EXCLUSIVAMENTE en el siguiente contexto:

{context}

---

Reglas OBLIGATORIAS:
1. Utiliza SOLO la información proporcionada en el contexto anterior
2. Si la información NO está EXPLÍCITAMENTE en el contexto, responde "No encontré información sobre esto en el contexto proporcionado"
3. NO HAGAS SUPOSICIONES ni agregues información externa
4. Si el contexto no está relacionado con la pregunta, responde "El contexto proporcionado no contiene información relacionada con esta pregunta"
5. Mantén la respuesta enfocada solo en lo que está documentado
6. NO RESPONDA sobre lo que trata el contexto proporcionado, solo que la pregunta no está en él.

Pregunta: {question}
"""


async def query(
        query_text: str,
        config_path: str,
        language: str = None,
        similarity_threshold: float = 0.004
):
    # Carregar configuração
    config = AppConfig.load_from_yaml(config_path)

    # Configurar logger
    logger = setup_logger("query", level=config.logging.level)

    try:
        # Detectar idioma se não especificado
        if not language:
            try:
                language = langdetect.detect(query_text)
                language = 'pt' if language == 'pt' else 'es' if language == 'es' else 'pt'
            except:
                language = 'pt'

        logger.info(f"Query language detected: {language}")

        # Inicializar componentes
        embedding_provider = OllamaEmbeddingProvider(
            model=config.embeddings.model_name
        )

        vector_store = ChromaVectorStore(
            persist_directory=config.vector_store.persist_directory,
            embedding_provider=embedding_provider,
            collection_name=config.vector_store.collection_name
        )

        llm_provider = OllamaLLMProvider(
            model_name=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=180,
            max_retries=3
        )

        # Verificar cache
        cache = LRUCache(capacity=config.cache.capacity)
        cached_response = await cache.get(query_text)
        if cached_response:
            logger.info("Cache hit")
            return cached_response

        # Buscar documentos relevantes
        results = await vector_store.similarity_search(query_text, k=5)

        # Filtrar resultados pelo threshold
        # filtered_results = [
        #     (doc, score) for doc, score in results
        #     if score >= float(similarity_threshold)
        # ]

        filtered_results = []
        for doc, score in results:
            try:
                score_float = float(score)
                # if score_float >= float(similarity_threshold):
                filtered_results.append((doc, score_float))
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting score to float: {str(e)}")
                continue

        if not filtered_results:
            return (
                "Não encontrei documentos suficientemente relevantes para sua pergunta. "
                "Por favor, reformule a pergunta ou busque por outro tópico."
            )
        # if not results:
        #     logger.warning("No relevant documents found")
        #     return "Não encontrei documentos relevantes para sua pergunta."

        # Preparar contexto - agora desempacotando as tuplas corretamente
        context_texts = []
        sources = []

        for doc, score in results:
            context_texts.append(doc.content)
            title = doc.metadata.get('title', 'Título Desconhecido')
            doc_id = doc.metadata.get('doc_id', 'ID Desconhecido')
            source = f"{title} (ID: {doc_id}, Score: {score:.3f})"
            if source not in sources:
                sources.append(source)

        context_text = "\n\n---\n\n".join(context_texts)

        logger.info(f"Found {len(results)} relevant documents")

        # Selecionar template baseado no idioma
        template = PROMPT_TEMPLATE_PT if language == 'pt' else PROMPT_TEMPLATE_ES

        # Preparar prompt
        prompt = template.format(context=context_text, question=query_text)

        # Sistema prompt mais restritivo
        system_prompt = """
            Você é um assistente técnico especializado em documentação de software.
            IMPORTANTE:
            1. Responda APENAS com informações presentes no contexto fornecido
            2. Se não houver informações suficientes, diga claramente
            3. NÃO USE conhecimento externo
            4. NÃO FAÇA suposições
            5. Seja direto e técnico em suas respostas
            6. Se a pergunta não estiver relacionada ao contexto, deixe isso claro
            7. NÃO RESPONDA do que o contexto fornecido se trata, apenas que a pergunta não está nele
            """
        # # Sistema prompt baseado no idioma
        # system_prompt = (
        #     "Você é um assistente técnico especializado em documentação de software. "
        #     f"Responda sempre em {'português' if language == 'pt' else 'español'}. "
        #     "Seja direto e técnico em suas respostas."
        # )

        # Gerar resposta
        try:
            response = await llm_provider.generate(
                f"{system_prompt}\n\n{prompt}"
            )
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Desculpe, ocorreu um erro ao gerar a resposta. Por favor, tente novamente."

        # Formatar resposta final
        formatted_response = (
            f"{'Resposta' if language == 'pt' else 'Respuesta'}:\n{response}\n\n"
            f"{'Fontes consultadas' if language == 'pt' else 'Fuentes consultadas'}:\n"
            f"{'-' * 40}\n"
            f"{chr(10).join('- ' + src for src in sources)}"
        )

        # Salvar no cache
        await cache.set(
            query_text,
            formatted_response,
            ttl=config.cache.default_ttl
        )

        return formatted_response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Ocorreu um erro ao processar sua pergunta: {str(e)}"


def main():
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
        help="Force specific threshold"
    )

    args = parser.parse_args()

    try:
        response = asyncio.run(query(args.query_text, args.config, args.language, args.similarity_threshold))
        print(response)
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário.")
    except Exception as e:
        print(f"Erro inesperado: {str(e)}")


if __name__ == "__main__":
    main()