import os
import ollama
from typing import List
from dotenv import load_dotenv

load_dotenv()

class OllamaLLM:
    def __init__(self, model: str = None):
        self.model = model or os.getenv("LLM_MODEL", "mistral")
    
    def generate(self, prompt: str, context: List[str] = None) -> str:
        if context:
            context_str = "\n\n".join(context)
            full_prompt = f"Context:\n{context_str}\n\nQuestion:\n{prompt}\n\nAnswer using only the context."
        else:
            full_prompt = prompt
        
        try:
            response = ollama.generate(model=self.model, prompt=full_prompt)
            return response.get('response', "")
        except Exception as e:
            return f"Error: {str(e)}"