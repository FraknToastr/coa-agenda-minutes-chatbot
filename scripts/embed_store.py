"""
Embed and store documents in a Chroma vectorstore
"""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loader import load_documents
from pathlib import Path

VECTOR_DIR = Path("../vectorstore")
DATA_DIRS = ["../data/pdfs", "../data/markdown", "../data/json"]

def build_vectorstore():
    # Load all docs
    docs = []
    for d in DATA_DIRS:
        docs.extend(load_documents(d))

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    # Embed
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Store
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=str(VECTOR_DIR))
    db.persist()
    print(f"✅ Vectorstore built and saved to {VECTOR_DIR}")

if __name__ == "__main__":
    build_vectorstore()
