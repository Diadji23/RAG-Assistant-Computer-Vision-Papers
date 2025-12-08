import chromadb
from typing import List
from langchain_core.documents import Document


class Retriever:
    """Retriever avec ChromaDB (plus rapide que FAISS)"""
    
    def __init__(self, embeddings, collection_name: str = "papers"):
        self.embeddings = embeddings
        self.collection_name = collection_name
        
        # Client persistent
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "CV Papers"}
        )
        
        print(f"ChromaDB initialized: {self.collection.count()} existing docs")
    
    def build(self, documents: List[Document], force_rebuild: bool = False):
        """Build index (incremental, rapide)"""
        
        current_count = self.collection.count()
        
        if current_count > 0 and not force_rebuild:
            print(f"Index already built with {current_count} docs")
            return
        
        if force_rebuild and current_count > 0:
            print("Clearing existing index...")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "CV Papers"}
            )
        
        print(f"Building index with {len(documents)} documents...")
        
        # Prepare data
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            
            self.collection.add(
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
            
            print(f"  Added batch {i//batch_size + 1} ({end}/{len(documents)})")
        
        print(f"Index built: {self.collection.count()} documents")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search similar documents"""
        
        if self.collection.count() == 0:
            raise ValueError("Index empty. Call build() first.")
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Convert to LangChain Documents
        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=results['metadatas'][0][i]
            )
            documents.append(doc)
        
        return documents
    
    def clear_cache(self):
        """Clear index"""
        self.client.delete_collection(name=self.collection_name)
        print("Index cleared")