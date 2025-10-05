"""
Loader for City of Adelaide documents (PDF, MD, JSON)
"""
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from pathlib import Path

def load_documents(folder_path: str):
    """Load all supported files from a directory into LangChain documents."""
    docs = []
    folder = Path(folder_path)

    for file in folder.glob("*"):
        suffix = file.suffix.lower()
        if suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif suffix in [".md", ".txt"]:
            docs.extend(TextLoader(str(file), encoding="utf-8").load())
        elif suffix == ".json":
            docs.extend(JSONLoader(str(file), jq_schema=".", text_content=False).load())
    return docs

if __name__ == "__main__":
    from pprint import pprint
    pprint(load_documents("../data/pdfs"))
