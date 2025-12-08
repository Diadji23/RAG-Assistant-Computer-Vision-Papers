
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import DataLoader

def test_data_text_files(): 
    loader = DataLoader("data/raw")
    docs = loader.load_pdfs()
    assert len(docs) > 0 
    assert hasattr(docs[0], "page_content") 


test_data_text_files()