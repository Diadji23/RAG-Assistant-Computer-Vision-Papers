import os
import logging
from typing import List
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from langchain_core.documents import Document

load_dotenv(".env.local")
logger = logging.getLogger(__name__)


class AzureSearchRetriever:
    def __init__(self, embedder, index_name: str = None, dimensions: int = 1536):
        self.embedder = embedder
        self.index_name = index_name or os.getenv("AZURE_SEARCH_INDEX", "papers")
        self.dimensions = dimensions
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        self.index_client = SearchIndexClient(endpoint, credential)
        self.search_client = SearchClient(endpoint, self.index_name, credential)
        self._create_index_if_needed()

    def _create_index_if_needed(self):
        if self.index_name in list(self.index_client.list_index_names()):
            return
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page", type=SearchFieldDataType.Int32),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.dimensions,
                vector_search_profile_name="default",
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
        )
        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        self.index_client.create_index(index)
        logger.info(f"Index '{self.index_name}' created")

    def count(self) -> int:
        return self.search_client.get_document_count()

    def build(self, documents: List[Document], batch_size: int = 100):
        texts = [doc.page_content for doc in documents]
        logger.info(f"Building index with {len(documents)} documents...")
        embeddings = self.embedder.embed_documents(texts)

        records = [
            {
                "id": f"doc_{i}",
                "content": text,
                "source": doc.metadata.get("source", "unknown"),
                "page": int(doc.metadata.get("page", 0)),
                "embedding": emb,
            }
            for i, (doc, text, emb) in enumerate(zip(documents, texts, embeddings))
        ]
        for i in range(0, len(records), batch_size):
            self.search_client.upload_documents(records[i:i + batch_size])
            logger.info(f"  Uploaded: {min(i + batch_size, len(records))}/{len(records)}")

    def search(self, query: str, k: int = 3) -> List[Document]:
        vector_query = VectorizedQuery(
            vector=self.embedder.embed_query(query),
            k_nearest_neighbors=k,
            fields="embedding",
        )
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=k,
            select=["content", "source", "page"],
        )
        return [
            Document(
                page_content=r["content"],
                metadata={"source": r["source"], "page": r["page"], "score": r["@search.score"]},
            )
            for r in results
        ]