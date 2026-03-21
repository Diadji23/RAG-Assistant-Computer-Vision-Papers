# RAG Assistant — Computer Vision Papers 

> Ask questions about Computer Vision research papers and get accurate answers with sources — powered by local LLMs via Ollama.

---

## Demo

```
Q: How does YOLO detect objects?

A: YOLO detects objects by training a convolutional neural network to predict 
regions of interest directly, instead of using Selective Search. The entire 
model is trained jointly on a loss function that directly corresponds to 
detection performance, allowing real-time inference.

Sources: yolo.pdf (3 chunks retrieved)
```

---

## Tech Stack

| Component     | Technology              |
|---------------|-------------------------|
| LLM           | Mistral (via Ollama)    |
| Embeddings    | mxbai-embed-large       |
| Vector DB     | ChromaDB (persistent)   |
| API           | FastAPI                 |
| Infra         | Docker + docker-compose |
| CI/CD         | GitHub Actions          |

---

## Quick Start

### 1. Install Ollama and pull models
```bash
ollama pull mistral
ollama pull mxbai-embed-large
```

### 2. Clone and install dependencies
```bash
git clone https://github.com/Diadji23/RAG.git
cd RAG
pip install -r requirements.txt
```

### 3. Add papers and run
```bash
mkdir -p data/raw
# Add your PDF papers to data/raw/
# Example:
wget https://arxiv.org/pdf/1506.02640.pdf -O data/raw/yolo.pdf
wget https://arxiv.org/pdf/1512.03385.pdf -O data/raw/resnet.pdf

python main.py
```

### Or with Docker
```bash
docker-compose up
```

---

## Architecture

```
PDF Papers → Chunking (512 tokens) → Embeddings → ChromaDB
                                                       ↓
Question → Embedding → Similarity Search → Top-K Chunks → LLM → Answer + Sources
```

---

## Project Structure

```
rag-assistant/
├── .github/workflows/   # CI/CD — tests run on every push
├── src/
│   ├── embedder.py      # mxbai-embed-large via Ollama
│   ├── retriever.py     # ChromaDB vector store
│   ├── llm.py           # Mistral via Ollama
│   ├── pipeline.py      # RAG orchestration
│   └── loader.py        # PDF loading + chunking
├── api/
│   └── main.py          # FastAPI REST endpoint
├── tests/               # Unit tests with pytest
├── main.py              # Entry point + interactive mode
├── Dockerfile
└── docker-compose.yml
```

---

## Results

- **965 chunks** indexed from 5 Computer Vision papers (YOLO, ResNet, CLIP, Transformers)
- Honest retrieval — model explicitly states when information is not in context (no hallucination)
- Average response time: ~2s per query (local CPU)

---

## API Usage

```bash
# Start the API
uvicorn api.main:app --reload

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does YOLO detect objects?"}'
```

---

## Papers Supported

- YOLO: You Only Look Once
- ResNet: Deep Residual Learning
- CLIP: Contrastive Language-Image Pretraining
- Attention Is All You Need (Transformers)

---

## Author

**Papa Diadji BOYE** — ML Engineer  
[GitHub](https://github.com/Diadji23) · [LinkedIn](https://www.linkedin.com/in/papa-diadji-boye/)