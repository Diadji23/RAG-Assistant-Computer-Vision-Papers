# RAG Assistant - Computer Vision Papers

Question-Answering system for scientific papers using RAG (Retrieval-Augmented Generation).

## What it does

Ask questions about Computer Vision papers (YOLO, ResNet, Transformers, etc.) and get accurate answers with sources.

## Tech Stack

- **LLM:** Mistral (via Ollama)
- **Embeddings:** mxbai-embed-large
- **Vector DB:** ChromaDB
- **Framework:** LangChain
- **API:** FastAPI (prochainemnt)

## Quick Start

```bash
#  Install Ollama and models
ollama pull mistral
ollama pull mxbai-embed-large

#  Install dependencies
pip install -r requirements.txt

# Add PDFs to data/raw/

#  Run
python main.py
```

## Architecture

```
PDF Papers → Chunking → Embeddings → ChromaDB
                                         ↓
Question → Embedding → Search → Top-K chunks → LLM → Answer
```

## Example

```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline(retriever, llm)
result = rag.run("How does YOLO work?")
print(result['answer'])
# Output: "YOLO divides the image into a grid and predicts bounding boxes..."
```



## Project Structure

```
src/          # Core RAG components
api/          # FastAPI endpoints
tests/
data/raw/     # PDF papers
main.py       

```
