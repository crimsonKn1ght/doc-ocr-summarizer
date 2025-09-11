import os
import io
import hashlib
import json
import streamlit as st
from typing import List
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# LangChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Supabase
from supabase import create_client

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="📄 DocQ&A", page_icon="📄", layout="wide")

# -----------------------------
# Supabase Setup
# -----------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
REDIRECT_URL = st.secrets.get("REDIRECT_URL")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# OAuth Login Handling
# -----------------------------
st.sidebar.header("🔑 Login")

if "session" not in st.session_state:
    st.session_state.session = None
if "messages" not in st.session_state:
    st.session_state.messages = []

user_id = None
user_email = None

if supabase:
    login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={REDIRECT_URL}"
    query_params = st.query_params
    access_token = query_params.get("access_token")
    refresh_token = query_params.get("refresh_token")

    if access_token and refresh_token and st.session_state.session is None:
        try:
            session = supabase.auth.set_session(access_token, refresh_token)
            st.session_state.session = session
        except Exception as e:
            st.warning(f"Auth session failed: {e}")

    logged_in = (
        st.session_state.session is not None
        and getattr(st.session_state.session, "user", None) is not None
    )

    if logged_in:
        user = st.session_state.session.user
        user_id = user.id
        user_email = user.email
        st.sidebar.success(f"✅ Logged in as {user_email}")
        if st.sidebar.button("🚪 Logout"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state.session = None
            st.query_params.clear()
            st.rerun()
    else:
        st.sidebar.markdown(f"[Login with Google]({login_url})")
else:
    st.sidebar.warning("⚠️ Supabase not configured")

# -----------------------------
# OCR and Text Extraction
# -----------------------------
def ocr_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return ""

def extract_text_from_pdf(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
        if use_ocr:
            for img_meta in page.get_images(full=True):
                xref = img_meta[0]
                base_image = doc.extract_image(xref)
                text += ocr_image(base_image["image"])
    return text

def extract_text_from_docx(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    doc = DocxDocument(file_bytes)
    text = "\n".join(para.text for para in doc.paragraphs)
    if use_ocr:
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                text += ocr_image(rel.target_part.blob)
    return text

def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    return file_bytes.read().decode("utf-8")

def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

# -----------------------------
# TF-IDF Embeddings
# -----------------------------
from langchain.embeddings.base import Embeddings

class TFIDFEmbeddings(Embeddings):
    def __init__(self, max_features=384):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )
        self.is_fitted = False
        self.dimension = max_features

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
        vectors = self.vectorizer.transform(texts).toarray()
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            return [0.0] * self.dimension
        v = self.vectorizer.transform([text]).toarray()[0]
        return v.tolist()

# -----------------------------
# Document Manager
# -----------------------------
class DocumentManager:
    def __init__(self):
        self.documents = []
        self.processed_files = {}
        self.embeddings = TFIDFEmbeddings()
        self.vectordb = None
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the following context to answer the question. "
                "If you cannot find the answer in the context, say "
                "\"I cannot find the answer in the provided documents.\""
                "\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        )

    def add_file(self, filename: str, content: str, file_hash: str, file_size: int):
        if file_hash in self.processed_files:
            return False, f"File '{filename}' already processed"
        doc = Document(
            page_content=content,
            metadata={"source": filename, "file_hash": file_hash, "file_size": file_size},
        )
        self.documents.append(doc)
        self.processed_files[file_hash] = {
            "name": filename,
            "size": file_size,
        }
        return True, f"Successfully processed '{filename}'"

    def _rebuild_vectordb(self):
        if not self.documents:
            self.vectordb = None
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(self.documents)
        all_texts = [c.page_content for c in chunks]
        self.embeddings = TFIDFEmbeddings()
        _ = self.embeddings.embed_documents(all_texts)
        self.vectordb = FAISS.from_documents(chunks, self.embeddings)

    def answer_question(self, question: str) -> str:
        if not self.documents or not self.vectordb:
            return "No documents uploaded yet."
        docs = self.vectordb.similarity_search(question, k=5)
        if not docs:
            return "I cannot find any relevant information."
        context = "\n\n".join(
            [f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in docs]
        )
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        return chain.run(context=context, question=question)

# -----------------------------
# Chat Persistence
# -----------------------------
def save_chat(user_id, messages):
    if user_id and supabase:
        try:
            supabase.table("chats").upsert({"user_id": user_id, "messages": messages}).execute()
        except Exception as e:
            st.warning(f"Failed to save chat: {e}")

def load_chat(user_id):
    if user_id and supabase:
        try:
            res = supabase.table("chats").select("messages").eq("user_id", user_id).execute()
            if res.data:
                return res.data[0]["messages"]
        except Exception as e:
            st.warning(f"Failed to load chat: {e}")
    return []

# -----------------------------
# Initialize
# -----------------------------
if "doc_manager" not in st.session_state:
    st.session_state.doc_manager = DocumentManager()

if user_id:
    st.session_state.messages = load_chat(user_id)

if "staged_files" not in st.session_state:
    st.session_state.staged_files = []

# -----------------------------
# Main UI
# -----------------------------
st.title("📄 DocQ&A — Your AI Assistant")

# Top-right user info
if user_email:
    st.markdown(
        f"<div style='text-align:right; font-size:14px;'>👤 {user_email}</div>",
        unsafe_allow_html=True,
    )

# Sidebar: Upload documents
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)
use_ocr = st.sidebar.checkbox("Enable OCR for images", True)

if uploaded_files:
    for f in uploaded_files:
        data = f.getvalue()
        file_hash = get_file_hash(data)
        if file_hash not in st.session_state.doc_manager.processed_files:
            ext = os.path.splitext(f.name)[1].lower()
            text = ""
            if ext == ".pdf":
                text = extract_text_from_pdf(io.BytesIO(data), use_ocr)
            elif ext == ".docx":
                text = extract_text_from_docx(io.BytesIO(data), use_ocr)
            elif ext == ".txt":
                text = extract_text_from_txt(io.BytesIO(data))
            if text.strip():
                st.session_state.doc_manager.add_file(f.name, text, file_hash, f.size)
    st.session_state.doc_manager._rebuild_vectordb()
    st.success("✅ Documents processed")

# -----------------------------
# Chat Interface
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.doc_manager.answer_question(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    if user_id:
        save_chat(user_id, st.session_state.messages)
