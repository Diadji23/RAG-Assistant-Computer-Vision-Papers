from src.embedder import MyEmbeddings
from unittest.mock import patch

def test_embed_query_returns_list():
    # On simule la réponse d'Ollama sans avoir besoin qu'il tourne
    with patch('ollama.embeddings') as mock:
        mock.return_value = {"embedding": [0.1, 0.2, 0.3]}
        
        embedder = MyEmbeddings()
        results = embedder.embed_query("test")
        
        assert isinstance(results, list)
        assert len(results) > 0