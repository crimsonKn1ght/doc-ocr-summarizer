import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from supabase import create_client, Client
import os
import datetime

# Initialize connection to Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "<YOUR_SUPABASE_URL>")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "<YOUR_SUPABASE_KEY>")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Authentication Functions ---
def login_user(email: str, password: str):
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return user
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None

def signup_user(email: str, password: str):
    try:
        user = supabase.auth.sign_up({"email": email, "password": password})
        return user
    except Exception as e:
        st.error(f"Signup failed: {e}")
        return None

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
st.title("📚 DocQnA App with Login & Chat History")

if "user" not in st.session_state:
    st.session_state.user = None

# Login / Signup form
if not st.session_state.user:
    tab1, tab2 = st.tabs(["Login", "Signup"])
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                user = login_user(email, password)
                if user:
                    st.session_state.user = user.user
                    st.success("Logged in successfully!")
                    st.rerun()
    with tab2:
        with st.form("signup_form"):
            email = st.text_input("New Email")
            password = st.text_input("New Password", type="password")
            submit = st.form_submit_button("Signup")
            if submit:
                user = signup_user(email, password)
                if user:
                    st.success("Account created! Please log in.")
else:
    st.success(f"Welcome {st.session_state.user.email}!")
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # Show chat history
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
        st.markdown(f"**Answer:** {answer}")

        # Save chat
        save_chat(st.session_state.user.id, question, answer)
