import os
import io
import hashlib
import json
import datetime
from typing import List, Tuple, Dict, Any

import streamlit as st
from PIL import Image
import pytesseract

# Optional libs
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

# LangChain & FAISS (optional; fallbacks available)
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.embeddings.base import Embeddings
    # Prefer ChatGroq if available — otherwise fallback to OpenAI if configured
    try:
        from langchain_groq import ChatGroq
    except Exception:
        ChatGroq = None
    try:
        from langchain import OpenAI
    except Exception:
        OpenAI = None
except Exception:
    Document = None
    RecursiveCharacterTextSplitter = None
    FAISS = None
    PromptTemplate = None
    LLMChain = None
    Embeddings = None
    ChatGroq = None
    OpenAI = None

# sklearn fallback for embeddings
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
except Exception:
    TfidfVectorizer = None
    np = None

# Supabase client
try:
    from supabase import create_client
except Exception:
    create_client = None

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="DocQnA — Mini GPT", page_icon="📄", layout="wide")

# ---------------- Secure config (no secrets printed) ----------------
# Provide keys via Streamlit secrets or environment variables. Do NOT hardcode.
SUPABASE_URL = st.secrets.get("SUPABASE_URL") if st.secrets else os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") if st.secrets else os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if st.secrets else os.getenv("OPENAI_API_KEY")
REDIRECT_URL = st.secrets.get("REDIRECT_URL") if st.secrets else os.getenv("REDIRECT_URL")

supabase = None
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.warning("Supabase initialization failed — authentication will be disabled.")

# ---------------- Utilities ----------------

def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()


def ocr_image(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def extract_text_from_pdf(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required to extract PDF text. Install pymupdf.")
    file_bytes.seek(0)
    doc = fitz.open(stream=file_bytes.read(), filetype='pdf')
    text_parts = []
    for page in doc:
        try:
            text_parts.append(page.get_text())
        except Exception:
            pass
        if use_ocr:
            try:
                for img_meta in page.get_images(full=True):
                    xref = img_meta[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image.get('image')
                    if img_bytes:
                        text_parts.append(ocr_image(img_bytes))
            except Exception:
                pass
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    if DocxDocument is None:
        raise RuntimeError("python-docx required to extract .docx. Install python-docx.")
    file_bytes.seek(0)
    doc = DocxDocument(file_bytes)
    text = "\n".join(p.text for p in doc.paragraphs)
    if use_ocr:
        try:
            for rel in doc.part.rels.values():
                if hasattr(rel, 'target_part') and getattr(rel.target_part, 'blob', None):
                    text += "\n" + ocr_image(rel.target_part.blob)
        except Exception:
            pass
    return text


def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    file_bytes.seek(0)
    return file_bytes.read().decode('utf-8', errors='replace')

# ---------------- Embeddings (TF-IDF fallback) ----------------
class TFIDFEmbeddings(Embeddings if Embeddings is not None else object):
    def __init__(self, max_features: int = 384):
        if TfidfVectorizer is None or np is None:
            raise RuntimeError('scikit-learn and numpy required for TFIDFEmbeddings')
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1,2))
        self.dimension = max_features
        self.is_fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
        vectors = self.vectorizer.transform(texts).toarray()
        out = []
        for v in vectors:
            if len(v) < self.dimension:
                v = np.pad(v, (0, self.dimension - len(v)), 'constant')
            out.append(v[:self.dimension].tolist())
        return out

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            return [0.0] * self.dimension
        v = self.vectorizer.transform([text]).toarray()[0]
        if len(v) < self.dimension:
            v = np.pad(v, (0, self.dimension - len(v)), 'constant')
        return v[:self.dimension].tolist()

# ---------------- Document Manager ----------------
class DocumentManager:
    def __init__(self):
        self.documents: List[Any] = []
        self.processed_files: Dict[str, Dict[str, Any]] = {}
        self.embeddings = TFIDFEmbeddings()
        self.vectordb = None
        self.llm = None
        # prefer ChatGroq then OpenAI
        if ChatGroq:
            try:
                self.llm = ChatGroq(model_name='llama-3.3-70b-versatile', temperature=0)
            except Exception:
                self.llm = None
        elif OpenAI and OPENAI_API_KEY:
            try:
                os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
                self.llm = OpenAI()  # default params
            except Exception:
                self.llm = None
        self.prompt_template = PromptTemplate(input_variables=['context','question'], template=(
            "Use the following context to answer the question. If the answer is not in the context, "
            "reply: 'I cannot find the answer in the provided documents.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")) if PromptTemplate else None

    def add_file(self, filename: str, content: str, file_hash: str, file_size: int) -> Tuple[bool, str]:
        if file_hash in self.processed_files:
            return False, f"Duplicate: '{filename}'"
        doc = Document(page_content=content, metadata={'source': filename, 'file_hash': file_hash, 'file_size': file_size})
        self.documents.append(doc)
        self.processed_files[file_hash] = {'name': filename, 'size': file_size}
        return True, f"Processed '{filename}'"

    def rebuild_index(self):
        if not self.documents:
            self.vectordb = None
            return
        if RecursiveCharacterTextSplitter is None or FAISS is None:
            st.warning('LangChain/FAISS not available; search will not work well. Install langchain and langchain-community.')
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(self.documents)
        all_texts = [c.page_content for c in chunks]
        self.embeddings = TFIDFEmbeddings()
        _ = self.embeddings.embed_documents(all_texts)
        self.vectordb = FAISS.from_documents(chunks, self.embeddings)

    def answer_question(self, question: str) -> str:
        if not self.documents:
            return "No documents uploaded. Please upload files first."
        if not self.vectordb:
            return "Index not built. Please process uploaded files."
        try:
            docs = self.vectordb.similarity_search(question, k=5)
            if not docs:
                return "I cannot find any relevant information in the uploaded documents."
            context = "\n\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in docs])
            if self.llm and self.prompt_template and LLMChain:
                chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
                return chain.run(context=context, question=question)
            else:
                snippets = "\n---\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content[:700]}..." for d in docs])
                return f"(LLM unavailable) Relevant snippets:\n\n{snippets}"
        except Exception as e:
            return f"Error during retrieval/answering: {e}"

    def get_document_list(self) -> List[Tuple[str,int]]:
        out = []
        for d in self.documents:
            m = getattr(d, 'metadata', {})
            out.append((m.get('source','unknown'), m.get('file_size',0)))
        return out

    def remove_file(self, filename: str):
        self.documents = [d for d in self.documents if d.metadata.get('source') != filename]
        to_remove = None
        for h, info in list(self.processed_files.items()):
            if info.get('name') == filename:
                to_remove = h
                break
        if to_remove:
            del self.processed_files[to_remove]
        if self.documents:
            self.rebuild_index()
        else:
            self.vectordb = None

    def clear_all(self):
        self.documents = []
        self.processed_files = {}
        self.vectordb = None

# ---------------- Chat helpers & Supabase persistence ----------------

def add_message(role: str, content: str):
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({'role': role, 'content': content, 'time': datetime.datetime.utcnow().isoformat()})


def save_session_messages(user_id: str):
    if not supabase or not user_id:
        return
    try:
        supabase.table('chats').upsert({'user_id': user_id, 'messages': st.session_state.messages}).execute()
    except Exception as e:
        st.warning('Failed to save chats to Supabase.')


def load_session_messages(user_id: str) -> List[Dict[str,Any]]:
    if not supabase or not user_id:
        return []
    try:
        res = supabase.table('chats').select('messages').eq('user_id', user_id).execute()
        if res.data and len(res.data) > 0:
            return res.data[0].get('messages', []) or []
    except Exception:
        pass
    return []

# ---------------- Initialize session state ----------------
if 'doc_manager' not in st.session_state:
    st.session_state.doc_manager = DocumentManager()
if 'staged_files' not in st.session_state:
    st.session_state.staged_files = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user' not in st.session_state:
    st.session_state.user = None

# ---------------- OAuth & Login UI ----------------
logged_in = False
user_id = None
if supabase and SUPABASE_URL and SUPABASE_KEY:
    login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={REDIRECT_URL or ''}"
    params = st.query_params
    access_token = params.get('access_token')
    refresh_token = params.get('refresh_token')
    if access_token and refresh_token and st.session_state.get('session') is None:
        try:
            session = supabase.auth.set_session({'access_token': access_token, 'refresh_token': refresh_token})
            st.session_state.session = session
        except Exception:
            st.warning('Failed to restore session from tokens in URL.')
    logged_in = st.session_state.get('session') is not None and getattr(st.session_state.get('session'), 'user', None) is not None
    if logged_in:
        st.session_state.user = st.session_state.session.user
        user_id = st.session_state.user.id
        # Merge or load remote messages
        if user_id and not st.session_state.messages:
            loaded = load_session_messages(user_id)
            if loaded:
                st.session_state.messages = loaded
else:
    login_url = None

# ---------------- Layout: cool UI ----------------
st.markdown("<style>.stApp { background: linear-gradient(180deg, #0f172a 0%, #021124 100%); color: #e2e8f0; } .sidebar .stButton button { background: #0ea5a4; }</style>", unsafe_allow_html=True)
st.title("📄 DocQnA — Mini ChatGPT")

left, right = st.columns([3,1])
with left:
    # Top bar with login status
    cols = st.columns([4,1])
    with cols[0]:
        st.markdown("### Chat")
    with cols[1]:
        if logged_in and st.session_state.user:
            st.success(f"{st.session_state.user.email}")
            if st.button('Logout'):
                try:
                    supabase.auth.sign_out()
                except Exception:
                    pass
                st.session_state.session = None
                st.session_state.user = None
                st.experimental_set_query_params()
                st.experimental_rerun()
        else:
            if login_url:
                st.markdown(f"[🔑 Login with Google]({login_url})")
            else:
                st.info('Login disabled (Supabase not configured)')

    st.divider()
    # Render chat messages
    for msg in st.session_state.messages:
        role = msg.get('role','assistant')
        time = msg.get('time','')
        if role == 'user':
            with st.chat_message('user'):
                st.markdown(msg.get('content',''))
                if time:
                    st.caption(time)
        else:
            with st.chat_message('assistant'):
                st.markdown(msg.get('content',''))
                if time:
                    st.caption(time)

    # Chat input
    user_prompt = st.chat_input('Ask me about your documents...')
    if user_prompt:
        add_message('user', user_prompt)
        # Get answer from doc manager
        answer = st.session_state.doc_manager.answer_question(user_prompt)
        add_message('assistant', answer)
        with st.chat_message('assistant'):
            st.markdown(answer)
        # Save only if logged in
        if logged_in and user_id:
            save_session_messages(user_id)

with right:
    st.markdown('### 📂 Upload & Manage Documents')
    uploaded = st.file_uploader('Upload files (pdf, docx, txt)', type=['pdf','docx','txt'], accept_multiple_files=True)
    use_ocr = st.checkbox('Enable OCR for images', value=True)

    # Immediate processing of uploaded files
    if uploaded:
        for f in uploaded:
            try:
                b = f.getvalue()
                h = get_file_hash(b)
                if h in st.session_state.doc_manager.processed_files:
                    st.info(f"Already processed: {f.name}")
                    continue
                # extract text immediately
                bio = io.BytesIO(b)
                txt = ''
                _, ext = os.path.splitext(f.name)
                ext = ext.lower()
                if ext == '.pdf':
                    txt = extract_text_from_pdf(bio, use_ocr)
                elif ext == '.docx':
                    txt = extract_text_from_docx(bio, use_ocr)
                elif ext == '.txt':
                    txt = extract_text_from_txt(bio)
                if txt.strip():
                    success, msg = st.session_state.doc_manager.add_file(f.name, txt, h, f.size)
                    if success:
                        st.success(msg)
                    else:
                        st.info(msg)
                else:
                    st.warning(f"No text found in {f.name}")
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")
        # rebuild index right away
        st.session_state.doc_manager.rebuild_index()

    st.divider()
    st.markdown('#### Loaded documents')
    docs = st.session_state.doc_manager.get_document_list()
    if docs:
        for i, (name, size) in enumerate(docs,1):
            cols = st.columns([4,1])
            cols[0].write(f"{i}. {name} ({(size or 0)/(1024*1024):.2f} MB)")
            if cols[1].button('Remove', key=f"rm_{i}"):
                st.session_state.doc_manager.remove_file(name)
                st.experimental_rerun()
        if st.button('Clear all'):
            st.session_state.doc_manager.clear_all()
            st.experimental_rerun()
    else:
        st.info('No documents uploaded yet')

st.divider()
st.caption('Notes: - Uploading documents processes them immediately. - Login with Google stores chat history in Supabase. - No secrets are printed in the UI; configure keys via Streamlit secrets or environment variables.')
