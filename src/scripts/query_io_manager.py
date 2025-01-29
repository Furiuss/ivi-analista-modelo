import argparse
import sys
import io
import asyncio
import logging
import warnings


from src.scripts.query import QueryProcessor


class RAGCLIOutputManager:
    """Módulo para gerenciar a saída do CLI de forma mais limpa."""

    def __init__(self, processor):
        self.processor = processor
        self.setup_silent_logging()

    def setup_silent_logging(self):
        """Configura o logging para ignorar warnings."""
        logging.basicConfig(
            level=logging.ERROR,
            format="%(levelname)s: %(message)s",
            stream=sys.stdout
        )

        warnings.filterwarnings("ignore")

    async def process_query(self, query_text, language=None, similarity_threshold=0.004):
        """
        Processa a query e retorna apenas a resposta final, sem logs.
        """
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        try:
            response = await self.processor.process_query(
                query_text, language, similarity_threshold
            )
        except Exception as e:
            sys.stdout = old_stdout
            return f"Ocorreu um erro ao processar sua pergunta: {str(e)}"

        sys.stdout = old_stdout
        return response

def main():
    parser = argparse.ArgumentParser(description="Interactive RAG Query System")

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
        help="Force specific similarity threshold"
    )

    args = parser.parse_args()


    try:
        processor = QueryProcessor(args.config)
        output_manager = RAGCLIOutputManager(processor)

        print("🤖 Ivi RAG")
        print("Digite 'exit', 'quit', ou pressione Ctrl+C para terminar a sessão.\n")

        while True:
            try:
                query_text = input("👉 Faça sua pergunta: ").strip()

                if query_text.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Tchau!")
                    break

                if not query_text:
                    continue

                response = asyncio.run(
                    output_manager.process_query(
                        query_text,
                        args.language,
                        args.similarity_threshold
                    )
                )
                print("\n" + response + "\n")

            except KeyboardInterrupt:
                print("\n\n👋 Saindo da Ivi RAG System.")
                break

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
