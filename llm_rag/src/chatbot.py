

from src.embeddings import MyEmbeddings
from src.retriever import Retriever
from src.llm import OllamaLLM 
from langchain_core.documents import Document

class RAGChatbot:
    def __init__(self, llm_model: str = "mxbai-13b"):
        # Embeddings
        self.embedding_model = MyEmbeddings(model="mxbai-embed-large")
        
        # Retriever
        self.retriever = Retriever(self.embedding_model)

        # LLM
        self.llm = OllamaLLM(model=llm_model)

    def build_index(self, documents: list[Document]):
        """
        Construire la base de données vectorielle FAISS avec les documents.
        """
        self.retriever.build(documents)

    def ask(self, query: str, k: int = 3) -> str:
        """
        Pose une question au chatbot et renvoie la réponse générée.
        """
        # 1️ Récupérer les documents les plus pertinents
        docs = self.retriever.search(query, k=k)
        
        if not docs:
            return "Désolé, aucun document pertinent trouvé."
        
        # Préparer le contexte pour le LLM
        context = "\n".join([doc.page_content for doc in docs])
        
        #  Générer la réponse
        prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {query}"
        answer = self.llm.generate(prompt)
        return answer
