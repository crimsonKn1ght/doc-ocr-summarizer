import os
import io
import json
import uuid
import hashlib
import streamlit as st
from PIL import Image
import pytesseract
from supabase import create_client
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ----------------- CONFIG ----------------- #
st.set_page_config(page_title="DocQ&A", page_icon="📄", layout="wide")

SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
REDIRECT_URL = st.secrets.get("REDIRECT_URL")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# ----------------- UTILS ----------------- #
def show_spacing(px=12):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

def ocr_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return ""

def extract_text_from_pdf(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
        if use_ocr:
            for img_meta in page.get_images(full=True):
                base_image = doc.extract_image(img_meta[0])
                text += ocr_image(base_image["image"])
    return text

def extract_text_from_docx(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(file_bytes)
    text = "\n".join(para.text for para in doc.paragraphs)
    if use_ocr:
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                text += ocr_image(rel.target_part.blob)
    return text

def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    return file_bytes.read().decode("utf-8")

# ----------------- TFIDF EMBEDDINGS ----------------- #
class TFIDFEmbeddings:
    def __init__(self, max_features=384):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        self.is_fitted = False
        self.dimension = max_features

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
        vectors = self.vectorizer.transform(texts)
        dense_vectors = vectors.toarray()
        result = []
        for vector in dense_vectors:
            if len(vector) < self.dimension:
                padded = np.pad(vector, (0, self.dimension - len(vector)), 'constant')
                result.append(padded.tolist())
            else:
                result.append(vector[:self.dimension].tolist())
        return result

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            return [0.0] * self.dimension
        dense_vector = self.vectorizer.transform([text]).toarray()
        if len(dense_vector) < self.dimension:
            padded = np.pad(dense_vector, (0, self.dimension - len(dense_vector)), 'constant')
            return padded.tolist()
        else:
            return dense_vector[:self.dimension].tolist()

# ----------------- DOCUMENT MANAGER ----------------- #
class DocumentManager:
    def __init__(self):
        self.documents = []
        self.processed_files = {}
        self.embeddings = TFIDFEmbeddings()
        self.vectordb = None
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
        )

    def add_file(self, filename: str, content: str, file_hash: str, file_size: int):
        if file_hash in self.processed_files:
            return False, f"File '{filename}' already processed (duplicate content detected)"
        doc = Document(page_content=content, metadata={
            "source": filename,
            "file_hash": file_hash,
            "file_size": file_size
        })
        self.documents.append(doc)
        self.processed_files[file_hash] = {
            "name": filename,
            "size": file_size,
            "processed_time": str(st.session_state.get('current_time', 'unknown'))
        }
        return True, f"Successfully processed '{filename}'"

    def _rebuild_vectordb(self):
        if not self.documents:
            self.vectordb = None
            return
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(self.documents)
            all_texts = [c.page_content for c in chunks]
            self.embeddings = TFIDFEmbeddings()
            _ = self.embeddings.embed_documents(all_texts)
            self.vectordb = FAISS.from_documents(chunks, self.embeddings)
        except Exception as e:
            st.error(f"Failed to build vector database: {e}")
            self.vectordb = None

    def answer_question(self, question: str) -> str:
        if not self.documents:
            return "No documents uploaded."
        if not self.vectordb:
            return "Document search not ready."
        try:
            docs = self.vectordb.similarity_search(question, k=5)
            if not docs:
                return "I cannot find any relevant information in the documents."
            context = "\n\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in docs])
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            return chain.run(context=context, question=question)
        except Exception as e:
            return f"Error: {str(e)}"

    def get_document_list(self):
        return [(doc.metadata['source'], doc.metadata.get('file_size', 0)) for doc in self.documents]

    def remove_file(self, filename: str):
        self.documents = [doc for doc in self.documents if doc.metadata['source'] != filename]
        file_hash_to_remove = None
        for file_hash, file_info in self.processed_files.items():
            if file_info['name'] == filename:
                file_hash_to_remove = file_hash
                break
        if file_hash_to_remove:
            del self.processed_files[file_hash_to_remove]
        self._rebuild_vectordb() if self.documents else setattr(self, 'vectordb', None)

    def clear_all(self):
        self.documents = []
        self.processed_files = {}
        self.vectordb = None

# ----------------- CHAT HANDLING ----------------- #
def save_chat(user_id, conv_id, messages):
    if not supabase or not user_id: return
    supabase.table("conversations").upsert({
        "user_id": user_id,
        "conversation_id": conv_id,
        "messages": json.dumps(messages)
    }).execute()

def load_chat(user_id, conv_id):
    if not supabase or not user_id: return []
    res = supabase.table("conversations").select("messages").eq("user_id", user_id).eq("conversation_id", conv_id).execute()
    if res.data and len(res.data) > 0:
        return json.loads(res.data[0]["messages"])
    return []

def list_conversations(user_id):
    if not supabase or not user_id: return []
    res = supabase.table("conversations").select("conversation_id").eq("user_id", user_id).execute()
    return [c["conversation_id"] for c in res.data] if res.data else []

# ----------------- SESSION STATE ----------------- #
if "doc_manager" not in st.session_state:
    st.session_state.doc_manager = DocumentManager()

if "staged_files" not in st.session_state:
    st.session_state.staged_files = []

if "user" not in st.session_state:
    st.session_state.user = None

# ----------------- LOGIN ----------------- #
query_params = st.query_params
if "error" in query_params:
    st.error("Login failed. Please try again.")
elif "code" in query_params and st.session_state.user is None:
    try:
        session = supabase.auth.exchange_code_for_session(query_params["code"][0])
        st.session_state.user = session.user
        st.query_params.clear()
        st.rerun()
    except Exception:
        st.error("Could not log you in. Please try again later.")

user = st.session_state.user

if not user:
    # Show login card
    st.markdown("""
    <div style="display:flex; justify-content:center; align-items:center; height:80vh;">
      <div style="text-align:center; padding:2rem; border-radius:1rem; box-shadow:0 4px 12px rgba(0,0,0,0.1); background:white; max-width:400px;">
        <h2>👋 Welcome to DocQ&A</h2>
        <p>Sign in to save your chats and documents</p>
        <a href="{login_url}">
          <button style="padding:0.8rem 1.5rem; border:none; border-radius:0.5rem; background:#4285F4; color:white; font-size:1rem; cursor:pointer;">
            <img src="https://www.svgrepo.com/show/475656/google-color.svg" width="20" style="vertical-align:middle; margin-right:8px;"/>
            Sign in with Google
          </button>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ----------------- MAIN APP ----------------- #
    st.sidebar.success(f"Logged in as {user.email}")
    if st.sidebar.button("Logout"):
        st.session_state.pop("user")
        st.session_state.pop("messages", None)
        st.rerun()

    # Conversation management
    if "conv_id" not in st.session_state:
        # Create new conversation by default
        st.session_state.conv_id = str(uuid.uuid4())
        if user:
            st.session_state.messages = []

    # Sidebar: list previous conversations
    if user:
        convs = list_conversations(user.id)
        st.sidebar.subheader("💬 Conversations")
        for c in convs:
            if st.sidebar.button(f"Conversation {c[:6]}", key=c):
                st.session_state.conv_id = c
                st.session_state.messages = load_chat(user.id, c)
        if st.sidebar.button("➕ New Conversation"):
            st.session_state.conv_id = str(uuid.uuid4())
            st.session_state.messages = []

    # ----------------- DOCUMENT UPLOAD ----------------- #
    st.sidebar.subheader("📂 Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Choose files", type=["pdf","docx","txt"], accept_multiple_files=True)
    use_ocr = st.sidebar.checkbox("Enable OCR", True)

    for f in uploaded_files:
        data = f.getvalue()
        file_hash = get_file_hash(data)
        if file_hash in st.session_state.doc_manager.processed_files: continue
        ext = os.path.splitext(f.name)[1].lower()
        text = ""
        if ext == ".pdf":
            text = extract_text_from_pdf(io.BytesIO(data), use_ocr)
        elif ext == ".docx":
            text = extract_text_from_docx(io.BytesIO(data), use_ocr)
        elif ext == ".txt":
            text = extract_text_from_txt(io.BytesIO(data))
        success, msg = st.session_state.doc_manager.add_file(f.name, text, file_hash, f.size)
        st.success(msg) if success else st.info(msg)
    st.session_state.doc_manager._rebuild_vectordb()

    # ----------------- CHAT ----------------- #
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me about your documents..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Real answer
        answer = st.session_state.doc_manager.answer_question(prompt)
        st.session_state.messages.append({"role":"assistant","content":answer})
        with st.chat_message("assistant"): st.markdown(answer)

        if user:
            save_chat(user.id, st.session_state.conv_id, st.session_state.messages)
