from src.pipeline import RAGPipeline 
from unittest.mock import MagicMock

retriever = MagicMock()
retriever.search.return_value = []  # simule un retriever vide

llm = MagicMock()
llm.generate.return_value = "test answer"  # simule une réponse



def test_pipeline_run():

    pipeline = RAGPipeline(retriever, llm)

    res= pipeline.run("test", k = 3)

    assert isinstance(res,dict)
    assert "answer" in res
    assert "sources" in res
