import os
import time
import logging
from typing import List
from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError

load_dotenv()
logger = logging.getLogger(__name__)


class AzureEmbeddings:
    def __init__(self, deployment: str = None, batch_size: int = 64):
        self.deployment = deployment or os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        self.batch_size = batch_size
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

    def _embed_batch(self, texts: List[str], retries: int = 5) -> List[List[float]]:
        for attempt in range(retries):
            try:
                response = self.client.embeddings.create(model=self.deployment, input=texts)
                return [item.embedding for item in response.data]
            except RateLimitError:
                wait = 2 ** attempt * 5
                logger.warning(f"Rate limited, retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError("Embedding failed after retries")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            logger.info(f"  Progress: {i}/{len(texts)}")
            embeddings.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]