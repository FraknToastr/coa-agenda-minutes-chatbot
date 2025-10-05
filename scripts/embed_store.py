"""
Embed and store City of Adelaide Council documents in a local Chroma vectorstore.
(Updated for LangChain 1.0 modular packages)

This version:
- Runs fully offline
- Uses HuggingFace embeddings (no API key needed)
- Stores vectors persistently in ChromaDB
"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loader import load_documents


# === CONFIGURATION ============================================================
VECTOR_DIR = Path("../vectorstore")  # Where to store Chroma database
DATA_DIRS = [
    "../data/pdfs",
    "../data/markdown",
    "../data/json",
]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
# ==============================================================================


def build_vectorstore():
    """Load documents, embed them, and persist locally in Chroma."""
    print("🏗️  Building local vectorstore...")

    # Load all documents from multiple folders
    all_docs = []
    for folder in DATA_DIRS:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"⚠️  Skipping missing folder: {folder}")
            continue

        docs = load_documents(folder)
        print(f"📂 Loaded {len(docs)} docs from {folder_path.name}")
        all_docs.extend(docs)

    if not all_docs:
        print("⚠️  No documents found! Add files under /data/pdfs, /markdown, or /json.")
        return

    # Split into manageable chunks
    print("✂️  Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    split_docs = splitter.split_documents(all_docs)
    print(f"   → {len(split_docs)} chunks created.")

    # Create local embeddings (runs offline)
    print(f"🧠 Generating embeddings using '{EMBEDDING_MODEL}' ...")
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create persistent Chroma vectorstore
    print("💾 Creating Chroma vectorstore...")
    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_fn,
        persist_directory=str(VECTOR_DIR)
    )
    db.persist()

    print(f"✅ Vectorstore built successfully and saved at: {VECTOR_DIR.resolve()}")


if __name__ == "__main__":
    build_vectorstore()
