"""
Auto-update script to add new files and refresh embeddings
"""
from loader import load_documents
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

DATA_DIRS = ["../data/pdfs", "../data/markdown", "../data/json"]
VECTOR_DIR = "../vectorstore"

def update_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    new_docs = []
    for d in DATA_DIRS:
        new_docs.extend(load_documents(d))

    split_new = splitter.split_documents(new_docs)
    db.add_documents(split_new)
    db.persist()
    print("✅ Vectorstore updated with new documents.")

if __name__ == "__main__":
    update_vectorstore()
