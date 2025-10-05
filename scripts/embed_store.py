"""
Embed and store City of Adelaide Council documents in a local Chroma vectorstore.

This version runs fully offline:
- Uses HuggingFace sentence-transformer embeddings
- No OpenAI dependencies
- Persists embeddings locally under ../vectorstore
"""

from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loader import load_documents


# === CONFIGURATION ============================================================
# Where to store the Chroma database
VECTOR_DIR = Path("../vectorstore")

# Source folders containing Council documents
DATA_DIRS = [
    "../data/pdfs",
    "../data/markdown",
    "../data/json"
]

# Embedding model (local, runs via sentence-transformers)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
# ==============================================================================


def build_vectorstore():
    print("🏗️  Building local vectorstore...")
    all_docs = []

    # Load all supported document types
    for folder in DATA_DIRS:
        print(f"📂 Loading from: {folder}")
        docs = load_documents(folder)
        print(f"   → {len(docs)} documents loaded.")
        all_docs.extend(docs)

    if not all_docs:
        print("⚠️  No documents found! Please add PDFs/MD/JSON to the data folders.")
        return

    # Split into manageable text chunks
    print("✂️  Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    split_docs = splitter.split_documents(all_docs)
    print(f"   → {len(split_docs)} chunks created.")

    # Create local embeddings
    print(f"🧠 Generating embeddings using '{EMBEDDING_MODEL}' ...")
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Build or overwrite Chroma database
    print("💾 Creating Chroma vectorstore...")
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_fn,
        persist_directory=str(VECTOR_DIR)
    )
    db.persist()

    print("✅ Vectorstore built and saved locally at:", VECTOR_DIR.resolve())


if __name__ == "__main__":
    build_vectorstore()
