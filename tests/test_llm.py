
from unittest.mock import Mock
from llm import OllamaLLM

def test_llm_generation(monkeypatch):
    fake_ollama = Mock()
    fake_ollama.generate.return_value = {"response": "Hello from LLM"}

    monkeypatch.setattr("llm.ollama", fake_ollama)

    llm = OllamaLLM(model="mistral")
    res = llm.generate("hi")
    
    assert res == "Hello from LLM"
