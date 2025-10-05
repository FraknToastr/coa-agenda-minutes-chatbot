"""
Conversational Retrieval-Augmented Chatbot (local version)
City of Adelaide Agenda & Minutes Chatbot

This version:
- Uses Ollama to run a local LLM (e.g. llama3, mistral, phi3)
- Retrieves relevant document chunks from Chroma
- Keeps short-term chat memory for context
"""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path


# === CONFIGURATION ============================================================
VECTOR_DIR = Path("../vectorstore")  # where Chroma is stored
MODEL_NAME = "llama3"                # or "mistral", "phi3", etc.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ==============================================================================


def create_chatbot():
    print(f"🤖 Starting local chatbot using model: {MODEL_NAME}")
    print("📚 Loading local vectorstore...")

    # Load embeddings and Chroma store
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embedding_fn)

    # Create local language model
    llm = OllamaLLM(model=MODEL_NAME, temperature=0)

    # Add conversation memory for continuity
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Build the retrieval-based conversation chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        memory=memory
    )

    print("✅ Local chatbot ready!")
    return chain


if __name__ == "__main__":
    chatbot = create_chatbot()
    print("\n🏛️ City of Adelaide Agenda & Minutes Chatbot\n")
    print("Type your question (or 'exit' to quit):\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        response = chatbot.invoke({"question": query})
        print("Bot:", response["answer"], "\n")
