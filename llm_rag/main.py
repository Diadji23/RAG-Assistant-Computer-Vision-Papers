from src.data_loader import DataLoader
from src.embeddings import MyEmbeddings
from src.retriever import Retriever
from src.llm import OllamaLLM
from src.rag_pipeline import RAGPipeline


def main():

    print("RAG SYSTEM DEMO - Computer Vision Papers")
    print("\n[1/4] Loading PDFs")
    loader = DataLoader(data_dir="data/raw")
    documents = loader.load_pdfs()

    print("\n[2/4] Building vector index")
    embeddings = MyEmbeddings(model="mxbai-embed-large")
    retriever = Retriever(embeddings)
    retriever.build(documents)
    print(f"Index built with {len(documents)} chunks")
    
    
    print("\n[3/4] Initializing LLM")
    llm = OllamaLLM(model="mistral")
    
    # Create pipeline
    print("\n[4/4] Creating RAG pipeline...")
    rag = RAGPipeline(retriever=retriever, llm=llm)
    
    print("\n" + "=" * 70)
    print("READY - Testing with sample questions")
    
    # Test questions
    questions = [
        "How does YOLO detect objects?",
        "What are skip connections in ResNet?",
        "Explain the transformer architecture",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        print(f"Q: {question}")
        print("-" * 70)
        
        result = rag.run(question, k=3)
        
        print(f"A: {result['answer']}\n")
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Retrieved: {result['num_sources']} chunks")
        print("=" * 70)
        
        if i < len(questions):
            input("\n[Press Enter for next question...]\n")
    
    # Interactive mode
    print("INTERACTIVE MODE - Type 'quit' to exit")    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        result = rag.run(question, k=3)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")


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