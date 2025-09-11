import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from supabase import create_client, Client
import os
import datetime

# Initialize connection to Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "<YOUR_SUPABASE_URL>")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "<YOUR_SUPABASE_KEY>")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helper function to add messages safely ---
def add_message(role: str, content: str):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": role, "content": content})

# --- Chat History Functions ---
def save_chat(user_id: str, question: str, answer: str):
    timestamp = datetime.datetime.utcnow().isoformat()
    supabase.table("chats").insert({
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "timestamp": timestamp
    }).execute()

def load_chat_history(user_id: str):
    response = supabase.table("chats").select("*").eq("user_id", user_id).order("timestamp").execute()
    return response.data if response.data else []

# --- App Logic ---
st.title("📚 DocQnA App with OAuth & Chat History")

if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# OAuth Login Section
if not st.session_state.user:
    st.info("Login with Google to save your chats.")
    login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to=http://localhost:8501"
    st.markdown(f"[Login with Google]({login_url})")

    # Handle redirect params after OAuth
    query_params = st.query_params
    if "access_token" in query_params and "refresh_token" in query_params:
        access_token = query_params["access_token"]
        refresh_token = query_params["refresh_token"]
        session = supabase.auth.set_session(access_token, refresh_token)
        if session and session.user:
            st.session_state.user = session.user
else:
    st.success(f"Welcome {st.session_state.user.email}!")
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # Show chat history from Supabase
    history = load_chat_history(st.session_state.user.id)
    if history:
        st.subheader("💬 Chat History")
        for chat in history:
            with st.container():
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.caption(f"{chat['timestamp']}")
    else:
        st.info("No past chats found.")

# --- Chat Interface ---
st.subheader("Ask a Question about your Documents")
question = st.text_input("Enter your question")
if st.button("Submit") and question:
    # Dummy Answer Logic (Replace with actual QA pipeline)
    answer = f"This is a dummy answer to: {question}"

    add_message("user", question)
    add_message("assistant", answer)

    st.markdown(f"**Answer:** {answer}")

    # Save chat only if logged in
    if st.session_state.user:
        save_chat(st.session_state.user.id, question, answer)
