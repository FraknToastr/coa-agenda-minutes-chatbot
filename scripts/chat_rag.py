"""
City of Adelaide Agenda & Minutes Chatbot
(Local version – updated for LangChain 1.0 modular packages)

This script:
- Uses Ollama for a local LLM (llama3, mistral, phi3, etc.)
- Loads documents from a local Chroma vector store
- Embeds text via HuggingFace (runs offline)
- Supports conversational memory
"""

from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# === CONFIGURATION ============================================================
VECTOR_DIR = Path("../vectorstore")               # Where Chroma DB is stored
MODEL_NAME = "llama3"                             # Any local Ollama model (e.g. llama3, mistral, phi3, gpt-oss:20b)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_K = 5                                       # Number of retrieved chunks per query
# ==============================================================================


def create_chatbot():
    """Create a conversational RAG chatbot using local models only."""
    print(f"🤖 Starting local chatbot using model: {MODEL_NAME}")
    print("📚 Loading local vectorstore...")

    # Local embeddings
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Load persisted Chroma vectorstore
    db = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embedding_fn)

    # Local LLM through Ollama
    llm = OllamaLLM(model=MODEL_NAME, temperature=0)

    # Short-term memory for the chat session
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Build the conversational RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": CHUNK_K}),
        memory=memory
    )

    print("✅ Local chatbot ready!")
    return chain


# === INTERACTIVE CLI LOOP =====================================================
if __name__ == "__main__":
    chatbot = create_chatbot()
    print("\n🏛️ City of Adelaide Agenda & Minutes Chatbot\n")
    print("Type your question (or 'exit' to quit):\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            response = chatbot.invoke({"question": query})
            print("Bot:", response["answer"], "\n")
        except Exception as e:
            print("⚠️  Error during response:", e, "\n")
