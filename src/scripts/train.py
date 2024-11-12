import asyncio
import argparse
from pathlib import Path
from src.core.config import AppConfig, PROJECT_ROOT
from src.embeddings.ollama import OllamaEmbeddingProvider
from src.vectorstores.chroma_store import ChromaVectorStore
from src.document_processing.loader import JSONDocumentLoader
from src.document_processing.splitter import RecursiveTextSplitter
from src.utils.logging import setup_logger


async def train(config_path: str, data_path: str, reset: bool = False):
    # Carregar configuração
    config = AppConfig.load_from_yaml(config_path)

    # Configurar logger
    logger = setup_logger("train", level=config.logging.level)

    try:
        # Converter caminho de dados para absoluto
        data_dir = Path(data_path)
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_path

        logger.info(f"Using data directory: {data_dir}")
        logger.info(f"Using Chroma directory: {config.vector_store.persist_directory}")

        # Inicializar componentes
        embedding_provider = OllamaEmbeddingProvider(
            model=config.embeddings.model_name
        )

        vector_store = ChromaVectorStore(
            persist_directory=config.vector_store.persist_directory,
            embedding_provider=embedding_provider,
            collection_name=config.vector_store.collection_name
        )

        # Reset se solicitado
        if reset:
            logger.info("Resetting vector store...")
            await vector_store.clear()

        # Carregar e processar documentos
        loader = JSONDocumentLoader()
        splitter = RecursiveTextSplitter(
            chunk_size=800,
            chunk_overlap=80
        )

        # Processar cada arquivo no diretório
        data_dir = Path(data_path)
        for file_path in data_dir.glob("*.json"):
            logger.info(f"Processing {file_path}")

            # Carregar documento
            documents = await loader.load(str(file_path))

            # Dividir em chunks
            chunks = splitter.split_documents(documents)

            # Adicionar ao vector store
            await vector_store.add_documents(chunks)

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train RAG system")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store"
    )

    args = parser.parse_args()
    asyncio.run(train(args.config, args.data, args.reset))


if __name__ == "__main__":
    main()