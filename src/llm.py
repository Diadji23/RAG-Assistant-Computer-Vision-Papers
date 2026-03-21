import os
import ollama
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


load_dotenv()


class OllamaLLM:
    def __init__(self, model: str = None):
        self.model = model or os.getenv("LLM_MODEL", "mistral")

    def generate(self, prompt: str) -> str:

        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            return response.get('response', "")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise
