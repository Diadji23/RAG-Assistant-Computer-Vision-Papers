import os
import logging
from dotenv import load_dotenv

load_dotenv(".env.local")

from src.loader import DataLoader
from src.embedder import MyEmbeddings
from src.retriever import Retriever
from src.llm import OllamaLLM
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

def get_components():
    if os.getenv("PROVIDER", "ollama") == "azure":
        from src.azure_embedder import AzureEmbeddings
        from src.azure_llm import AzureLLM
        logger.info("Provider: Azure")
        return AzureEmbeddings(), AzureLLM()
    logger.info("Provider: Ollama")
    return MyEmbeddings(model="mxbai-embed-large"), OllamaLLM(model="mistral")


def main():

    logger.info("RAG SYSTEM DEMO - Computer Vision Papers")
    logger.info("\n[1/4] Loading PDFs")
    loader = DataLoader(data_dir="data/raw")
    documents = loader.load_pdfs()
    documents = loader.chunk_documents(documents)

    embeddings, llm = get_components()

    logger.info("\n[2/4] Building vector index")
    if os.getenv("RETRIEVER", "chroma") == "azure_search":
        from src.azure_search_retriever import AzureSearchRetriever
        retriever = AzureSearchRetriever(embeddings)
        if retriever.count() == 0:
            retriever.build(documents)
        else:
            logger.info(f"Using existing Azure index: {retriever.count()} docs")
    else:
        retriever = Retriever(embeddings)
        retriever.build(documents)

    logger.info("\n[3/4] LLM ready")

    # Create pipeline
    logger.info("\n[4/4] Creating RAG pipeline...")
    rag = RAGPipeline(retriever=retriever, llm=llm)

    logger.info("\n" + "=" * 70)
    logger.info("READY - Testing with sample questions")

    # Test questions
    questions = [
        "How does YOLO detect objects?",
        "What are skip connections in ResNet?",
        "Explain the transformer architecture",
    ]

    for i, question in enumerate(questions, 1):
        logger.info(f"\n[Question {i}/{len(questions)}]")
        logger.info(f"Q: {question}")
        logger.info("-" * 70)

        result = rag.run(question, k=3)

        logger.info(f"A: {result['answer']}\n")
        logger.info(f"Sources: {', '.join(result['sources'])}")
        logger.info(f"Retrieved: {result['num_sources']} chunks")
        logger.info("=" * 70)

        if i < len(questions):
            logger.info("\n[Press Enter for next question...]\n")

    # Interactive mode
    logger.info("INTERACTIVE MODE - Type 'quit' to exit")
    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            logger.info("Goodbye!")
            break

        if not question:
            continue

        result = rag.run(question, k=3)
        logger.info(f"\nAnswer: {result['answer']}")
        logger.info(f"Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    data_dir = Path("data/raw")
    if not data_dir.exists() or not list(data_dir.glob("*.pdf")):
        print("ERROR: No PDFs found in data/raw/")
        print("\nDownload some papers:")
        print("mkdir -p data/raw")
        print("cd data/raw")
        print("wget https://arxiv.org/pdf/1506.02640.pdf -O yolo.pdf")
        print("wget https://arxiv.org/pdf/1512.03385.pdf -O resnet.pdf")
        sys.exit(1)

    main()