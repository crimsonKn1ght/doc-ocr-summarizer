import os
import tempfile
from typing import List, Tuple
import io
import json

import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PIL import Image
import pytesseract
from supabase import create_client

# --- Supabase Setup ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")

st.write("Supabase URL loaded:", bool(SUPABASE_URL))
st.write("Supabase Key loaded:", bool(SUPABASE_KEY))

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY in Streamlit Cloud → Settings → Secrets")
    st.stop()

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.success("✅ Supabase client initialized successfully")
except Exception as e:
    st.error(f"❌ Failed to init Supabase client: {type(e).__name__} - {e}")
    st.stop()

# --- Auth Section ---
st.sidebar.header("Login")

# OAuth login URL (Google)
login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={st.secrets.get('REDIRECT_URL')}"

if "session" not in st.session_state:
    st.session_state.session = None

# If not logged in, show login button
if not st.session_state.session:
    st.sidebar.markdown(f"[🔑 Login with Google]({login_url})")
    st.stop()

# If logged in, extract user id
session = supabase.auth.get_session()
if session and session.user:
    user_id = session.user.id   # UUID of logged-in user
    st.sidebar.success(f"Logged in as {session.user.email}")
else:
    st.sidebar.error("Not logged in")
    st.stop()

# --- OCR Function ---
def ocr_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.warning(f"OCR failed for an image: {e}")
        return ""

# --- Text Extraction ---
def extract_text_from_pdf(pdf_path: str, use_ocr: bool = True) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        if use_ocr:
            for img_meta in page.get_images(full=True):
                xref = img_meta[0]
                base_image = doc.extract_image(xref)
                text += ocr_image(base_image["image"])
    return text

def extract_text_from_docx(docx_path: str, use_ocr: bool = True) -> str:
    from docx import Document
    doc = Document(docx_path)
    text = "\n".join(para.text for para in doc.paragraphs)
    if use_ocr:
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                text += ocr_image(rel.target_part.blob)
    return text

def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def get_text_from_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], use_ocr: bool) -> List[Tuple[str, str]]:
    extracted_texts = []
    for uploaded_file in uploaded_files:
        try:
            suffix = f".{uploaded_file.name.split('.')[-1]}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                tmp_path = tmpfile.name

            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".pdf":
                text = extract_text_from_pdf(tmp_path, use_ocr)
            elif ext == ".docx":
                text = extract_text_from_docx(tmp_path, use_ocr)
            elif ext == ".txt":
                text = extract_text_from_txt(tmp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue

            if text.strip():
                extracted_texts.append((uploaded_file.name, text))
            os.remove(tmp_path)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    return extracted_texts

# --- VectorDB Builder ---
def build_vectordb(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", encode_kwargs={"batch_size": 32})
    return FAISS.from_documents(chunks, embeddings)

# --- QA System ---
class DocumentQA:
    def __init__(self, texts: List[Tuple[str, str]]):
        self.documents = [Document(page_content=text, metadata={"source": src}) for src, text in texts]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_documents(self.documents)
        self.vectordb = build_vectordb(self.chunks)
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context to answer the question:
Context: {context}

Question: {question}

Answer:"""
        )

    def add_documents(self, texts: List[Tuple[str, str]]):
        new_docs = [Document(page_content=text, metadata={"source": src}) for src, text in texts]
        self.documents.extend(new_docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_chunks = splitter.split_documents(new_docs)
        self.chunks.extend(new_chunks)
        self.vectordb.add_documents(new_chunks)

    def answer_question(self, q: str) -> str:
        docs = self.vectordb.similarity_search(q, k=3)
        context = "\n\n".join([f"Source: {d.metadata['source']}\n{d.page_content}" for d in docs])
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        return chain.run(context=context, question=q)

# --- Chat History Save/Load ---
def save_chat(user_id, messages):
    supabase.table("chats").upsert({
        "user_id": user_id,
        "messages": messages
    }).execute()

def load_chat(user_id):
    res = supabase.table("chats").select("messages").eq("user_id", user_id).execute()
    return res.data[0]["messages"] if res.data else []

# --- Streamlit Interface ---
st.set_page_config(page_title="Document Q&A with Login", page_icon="🔑", layout="wide")
st.title("🔑 Document Q&A Assistant with Supabase")

# Load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat(user_id)

# Upload & Process Files
with st.sidebar:
    st.header("Upload Documents")
    use_ocr = st.checkbox("Enable OCR", True)
    files = st.file_uploader("Upload", type=["pdf","docx","txt"], accept_multiple_files=True)
    if "qa" not in st.session_state: st.session_state.qa = None
    if files:
        texts = get_text_from_files(files, use_ocr)
        if texts:
            if st.session_state.qa is None:
                st.session_state.qa = DocumentQA(texts)
            else:
                st.session_state.qa.add_documents(texts)
            st.success(f"Added {len(texts)} docs.")

# Chat Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Ask about your docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    if st.session_state.qa:
        with st.spinner("Thinking..."):
            ans = st.session_state.qa.answer_question(prompt)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"): st.markdown(ans)
            save_chat(user_id, st.session_state.messages)
    else:
        with st.chat_message("assistant"):
            msg = "Please upload docs first!"
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
