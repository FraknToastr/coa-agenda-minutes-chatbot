import streamlit as st
from scripts.chat_rag import create_chatbot

st.set_page_config(page_title="City of Adelaide Chatbot", page_icon="🏛️", layout="wide")
st.title("🏛️ City of Adelaide Agenda & Minutes Chatbot")

chatbot = create_chatbot()

if "history" not in st.session_state:
    st.session_state["history"] = []

query = st.text_input("Ask a question about Council meetings or decisions:")

if st.button("Send") and query:
    response = chatbot.invoke({"question": query})
    st.session_state["history"].append((query, response["answer"]))

for user, bot in st.session_state["history"]:
    st.markdown(f"**You:** {user}")
    st.markdown(f"**Bot:** {bot}")
