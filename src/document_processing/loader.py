from typing import List, Dict, Any
from src.core.interfaces import Document, DocumentLoader
from datetime import datetime
import json
from pathlib import Path
from bs4 import BeautifulSoup
import logging
import hashlib

logger = logging.getLogger(__name__)


class JSONDocumentLoader(DocumentLoader):
    def __init__(self, content_field: str = "ContentHtml"):
        self.content_field = content_field

    def _generate_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Gera um ID único para o documento"""
        # Usar ID existente se disponível
        if 'doc_id' in metadata and metadata['doc_id']:
            return str(metadata['doc_id'])

        # Gerar hash do conteúdo + timestamp
        content_hash = hashlib.md5(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()

        return f"doc_{content_hash[:12]}"

    async def load(self, path: str) -> List[Document]:
        documents = []
        file_path = Path(path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

                article = json_data.get('Article', {})
                title = article.get('Title', '')

                html_content = article.get(self.content_field, '')
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.get_text(separator='\n', strip=True)

                categories = article.get('Categories', [])
                category_str = ';'.join([cat.get('Name', '') for cat in categories])

                # Gerar metadata com ID garantido
                metadata = {
                    "source": file_path.name,
                    "title": title,
                    "category": category_str,
                    "doc_id": str(article.get('Id', '')),
                    "created_date": str(article.get('CreatedDate', '')),
                    "updated_date": str(article.get('UpdatedDate', '')),
                    "created_at": datetime.now().isoformat()
                }

                # Garantir que temos um ID válido
                if not metadata["doc_id"]:
                    metadata["doc_id"] = self._generate_doc_id(text_content, metadata)

                doc = Document(
                    content=text_content,
                    metadata=metadata,
                    chunk_id=metadata["doc_id"]  # Usar o ID do documento como chunk_id inicial
                )
                documents.append(doc)

                logger.info(f"Loaded document {metadata['doc_id']} from {file_path.name}")

        except Exception as e:
            logger.error(f"Error loading document {path}: {str(e)}")
            raise

        return documents