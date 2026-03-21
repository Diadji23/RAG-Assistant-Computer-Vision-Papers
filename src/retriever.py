import chromadb
from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """Retriever avec ChromaDB"""

    def __init__(self, embedder, collection_name: str = "papers"):
        self.embedder = embedder
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "CV papers"},
        )
        logger.info(
            f"ChromaDB initialized: {self.collection.count()} existing docs"
        )

    def build(self, documents: List[Document]):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]

        logger.info(f"Building index with {len(documents)} documents...")
        embeddings = self.embedder.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f"Index built: {self.collection.count()} documents")

    def search(self, query: str, k: int = 3) -> List[Document]:
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        documents = []
        for i in range(len(results["ids"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            documents.append(doc)

        return documents
