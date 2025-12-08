from typing import List
from langchain_core.documents import Document

class RAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def run(self, query: str, k: int = 5) -> dict:
        # Retrieve
        docs = self.retriever.search(query, k=k)
        
        if not docs:
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "sources": []
            }
            
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        
        # Generate
        prompt = f"""You are a computer vision expert.
Answer the question using ONLY the following context.
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

        answer = self.llm.generate(prompt)        
        return {
            "query": query,
            "answer": answer,
            "sources": [doc.metadata.get('source', 'unknown') for doc in docs],
            "num_sources": len(docs)
        }
    
