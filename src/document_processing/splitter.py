from typing import List, Optional
from src.core.interfaces import Document, TextSplitter
import logging
import uuid

logger = logging.getLogger(__name__)


class RecursiveTextSplitter(TextSplitter):
    def __init__(
            self,
            chunk_size: int = 800,
            chunk_overlap: int = 80,
            length_function: callable = len,
            separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Gera um ID único para o chunk"""
        return f"{doc_id}_chunk_{chunk_index}"

    def split_documents(
            self,
            documents: List[Document],
            chunk_size: Optional[int] = None,
            chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        splits = []
        for doc in documents:
            chunks = self._split_text(doc.content, chunk_size, chunk_overlap)

            doc_id = doc.metadata.get('doc_id', str(uuid.uuid4()))

            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(doc_id, i)

                # Criar novo documento para cada chunk
                splits.append(Document(
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_size": len(chunk),
                        "total_chunks": len(chunks),
                        "is_chunk": True
                    },
                    chunk_id=chunk_id  # ID único garantido para cada chunk
                ))

        logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
        return splits

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Separa texto em chunks usando splitting recursivo."""
        if self.length_function(text) <= chunk_size:
            return [text]

        chunks = []
        for sep in self.separators:
            if sep == "":
                # If eu cheguei no fim dos separadores, separa por caractere
                return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

            splits = text.split(sep)
            if len(splits) > 1:
                # Processa cada split recursivamente
                good_splits = []
                current_chunk = []
                current_length = 0

                for split in splits:
                    split_length = self.length_function(split)
                    if current_length + split_length <= chunk_size:
                        current_chunk.append(split)
                        current_length += split_length
                    else:
                        if current_chunk:
                            good_splits.append(sep.join(current_chunk))
                        current_chunk = [split]
                        current_length = split_length

                if current_chunk:
                    good_splits.append(sep.join(current_chunk))

                return good_splits

        return chunks