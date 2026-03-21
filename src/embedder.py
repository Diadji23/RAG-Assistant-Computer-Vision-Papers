from typing import List
import ollama
import logging


logger = logging.getLogger(__name__)


class MyEmbeddings:
    def __init__(self, model: str = "mxbai-embed-large"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(texts)}")
            embedding = self._embed_text(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        """Internal method to embed text using Ollama"""
        try:
            response = ollama.embeddings(
                model=self.model,
                prompt=text[:8192]
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
