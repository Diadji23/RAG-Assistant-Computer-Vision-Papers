from typing import List
import requests
import time

class MyEmbeddings:
    def __init__(self, model: str = "mxbai-embed-large", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        print(f"Embedding {len(texts)} documents...")
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:  
                print(f"  Progress: {i}/{len(texts)}")
            embedding = self._embed_text(text)
            embeddings.append(embedding)
            time.sleep(0.1)  
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._embed_text(text)
    
    def _embed_text(self, text: str) -> List[float]:
        """Internal method to embed text using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text[:8192]  
                },
                timeout=30  
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            print(f"  Generated embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Fallback: return random embedding
            import random
            return [random.random() for _ in range(1024)]
        
