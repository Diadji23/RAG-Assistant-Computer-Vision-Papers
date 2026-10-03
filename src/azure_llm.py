import os
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
logger = logging.getLogger(__name__)


class AzureLLM:
    def __init__(self, deployment: str = None):
        self.deployment = deployment or os.getenv("AZURE_CHAT_DEPLOYMENT")
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure LLM error: {e}")
            raise