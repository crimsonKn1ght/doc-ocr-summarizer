import os
import io
from typing import List, Tuple

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

# --- Streamlit Page Config ---
st.set_page_config(page_title="DocQ&A", page_icon="📄", layout="wide")

# --- Supabase Setup ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
REDIRECT_URL = st.secrets.get("REDIRECT_URL")  # must be your deployed Streamlit URL

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- OAuth Login Handling ---
st.sidebar.header("Login")

login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={REDIRECT_URL}"

query_params = st.query_params
access_token = query_params.get("access_token")
refresh_token = query_params.get("refresh_token")

if access_token and refresh_token:
    session = supabase.auth.set_session(
        {"access_token": access_token, "refresh_token": refresh_token}
    )
    st.session_state.session = session

logged_in = (
    "session" in st.session_state
    and st.session_state.session
    and st.session_state.session.user
)

if logged_in:
    user = st.session_state.session.user
    user_id = user.id
    st.sidebar.success(f"✅ Logged in as {user.email}")
    if st.sidebar.button("🚪 Logout"):
        supabase.auth.sign_out()
        st.session_state.session = None
        st.query_params.clear()
        st.rerun()
else:
    st.sidebar.markdown(f"[🔑 Login with Google]({login_url})")
    user_id = None  # no Supabase link

# --- OCR Function ---
def ocr_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.warning(f"OCR failed for an image: {e}")
        return ""

# --- Text Extraction ---
def extract_text_from_pdf(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    import fitz  # PyMuPDF
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
    from docx import Document
    doc = Document(file_bytes)
    text = "\n".join(para.text for para in doc.paragraphs)
    if use_ocr:
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                text += ocr_image(rel.target_part.blob)
    return text

def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    return file_bytes.read().decode("utf-8")

def get_text_from_files(uploaded_files, use_ocr: bool) -> List[Tuple[str, str]]:
    extracted_texts = []
    for uploaded_file in uploaded_files:
        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            file_bytes = io.BytesIO(uploaded_file.getvalue())

            if ext == ".pdf":
                text = extract_text_from_pdf(file_bytes, use_ocr)
            elif ext == ".docx":
                text = extract_text_from_docx(file_bytes, use_ocr)
            elif ext == ".txt":
                text = extract_text_from_txt(file_bytes)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue

            if text.strip():
                extracted_texts.append((uploaded_file.name, text))

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    return extracted_texts

# --- VectorDB Builder ---
def build_vectordb(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},      # 👈 Force CPU
        encode_kwargs={"batch_size": 32}
    )
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
            template="""Use the following context to answer the question:

Context:
{context}

Question:
{question}

Answer:"""
        )

    def add_documents(self, texts: List[Tuple[str, str]]):
        new_docs = [Document(page_content=text, metadata={"source": src}) for src, text in texts]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_chunks = splitter.split_documents(new_docs)
        self.documents.extend(new_docs)
        self.chunks.extend(new_chunks)
        self.vectordb.add_documents(new_chunks)

    def answer_question(self, q: str) -> str:
        docs = self.vectordb.similarity_search(q, k=3)
        context = "\n\n".join([f"Source: {d.metadata['source']}\n{d.page_content}" for d in docs])
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        return chain.run(context=context, question=q)

# --- Chat History Persistence ---
def save_chat(user_id, messages):
    if user_id:
        supabase.table("chats").upsert({"user_id": user_id, "messages": messages}).execute()
    else:
        st.session_state._local_messages = messages  # in-memory only

def load_chat(user_id):
    if user_id:
        res = supabase.table("chats").select("messages").eq("user_id", user_id).execute()
        return res.data[0]["messages"] if res.data else []
    else:
        return getattr(st.session_state, "_local_messages", [])

# --- Main Interface ---
st.title("📄 DocQ&A – Your AI Assistant")

# Load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat(user_id)

# Sidebar: Upload docs
with st.sidebar:
    st.header("📂 Upload Documents")
    use_ocr = st.checkbox("Enable OCR", True)
    files = st.file_uploader("Upload", type=["pdf","docx","txt"], accept_multiple_files=True)
    if "qa" not in st.session_state:
        st.session_state.qa = None
    if files:
        texts = get_text_from_files(files, use_ocr)
        if texts:
            if st.session_state.qa is None:
                st.session_state.qa = DocumentQA(texts)
            else:
                st.session_state.qa.add_documents(texts)
            st.success(f"✅ Added {len(texts)} docs.")

# Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.qa:
        with st.spinner("Thinking..."):
            ans = st.session_state.qa.answer_question(prompt)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.markdown(ans)
            save_chat(user_id, st.session_state.messages)
    else:
        with st.chat_message("assistant"):
            msg = "📂 Please upload documents first!"
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
