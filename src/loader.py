import os 
from pathlib import Path 
from typing import List 
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from tqdm import tqdm 
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter



logger = logging.getLogger(__name__)


class DataLoader: 
    """ load computer vision documents """
    def __init__(self, data_dir ="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents = True , exist_ok = True)

    def load_pdfs(self , pdf_path= None): 
        if pdf_path: 
            loader = PyPDFLoader(pdf_path) 
            return loader.load()
    
        loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        return loader.load()
    
    def load_text_files(self):
        loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        return loader.load()
    

    def load_all_documents(self) -> List[Document]:
        """Charge tous les types de documents"""
        all_docs = []
        
        print("  PDFs...")
        try:
            all_docs.extend(self.load_pdfs())
        except Exception as e:
            print(f" Erreur PDFs: {e}")
        
        print("text files")
        try:
            all_docs.extend(self.load_text_files())
        except Exception as e:
            print(f"  Erreur texte: {e}")
        
        print(f"Total: {len(all_docs)} documents loaded")
        return all_docs
    

    def get_document_stats(self , documents) :
        total_chars = sum(len(doc.page_content) for doc in documents) 

        return { 
            "nb_documents": len(documents) , 
            "total_characters" : total_chars , 
            "avg_doc_legnths" : total_chars // len(documents) if documents else 0
        }



    def chunk_documents(self, documents, chunk_size=512, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)


