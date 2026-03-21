
from src.embedder import MyEmbeddings





def test_emed_query_returns_list():
    embedder = MyEmbeddings()
    results = embedder.embed_query("test")
    assert isinstance(results, list)
    assert results != []

