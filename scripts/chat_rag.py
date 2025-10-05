"""
Conversational Retrieval-Augmented Chatbot
"""
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma

VECTOR_DIR = "../vectorstore"

def create_chatbot():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    db = Chroma(persist_directory=VECTOR_DIR, embedding_function=None)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        memory=memory
    )

if __name__ == "__main__":
    chatbot = create_chatbot()
    print("City of Adelaide Council Chatbot ready! Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break
        ans = chatbot.invoke({"question": q})
        print("Bot:", ans["answer"], "\n")
