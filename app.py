import os
import io
import hashlib
import json
from typing import List, Tuple, Dict, Any

import streamlit as st
from PIL import Image
import pytesseract

# Optional imports that may not be installed in every environment
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

# LangChain / FAISS / embeddings
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.embeddings.base import Embeddings
    from langchain_groq import ChatGroq
except Exception:
    Document = None
    RecursiveCharacterTextSplitter = None
    FAISS = None
    PromptTemplate = None
    LLMChain = None
    Embeddings = None
    ChatGroq = None

# Sci-kit learn for TF-IDF fallback
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

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="DocQnA Mini GPT", page_icon="📄", layout="wide")

# ---------------- Supabase / OAuth setup ----------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") if st.secrets else os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") if st.secrets else os.getenv("SUPABASE_KEY")
REDIRECT_URL = st.secrets.get("REDIRECT_URL") if st.secrets else os.getenv("REDIRECT_URL")

supabase = None
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.warning(f"Supabase init failed: {e}")

# ---------------- Utilities ----------------

def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()


def ocr_image(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def extract_text_from_pdf(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required to extract PDF text. Install pymupdf.")
    file_bytes.seek(0)
    doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
    text = ""
    for page in doc:
        try:
            text += page.get_text()
        except Exception:
            pass
        if use_ocr:
            try:
                images = page.get_images(full=True)
                for img_meta in images:
                    xref = img_meta[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image.get("image")
                    if img_bytes:
                        text += "\n" + ocr_image(img_bytes)
            except Exception:
                pass
    return text


def extract_text_from_docx(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    if DocxDocument is None:
        raise RuntimeError("python-docx is required to extract docx text. Install python-docx.")
    file_bytes.seek(0)
    doc = DocxDocument(file_bytes)
    text = "\n".join([p.text for p in doc.paragraphs])
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

# ---------------- Simple TF-IDF Embeddings ----------------
class TFIDFEmbeddings(Embeddings if Embeddings is not None else object):
    def __init__(self, max_features: int = 384):
        if TfidfVectorizer is None or np is None:
            raise RuntimeError("scikit-learn and numpy required for TFIDFEmbeddings")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1,2),
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        self.dimension = max_features
        self.is_fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_fitted:
            self.vectorizer.fit(texts)
            self.is_fitted = True
        vectors = self.vectorizer.transform(texts).toarray()
        # Ensure fixed dimension
        out = []
        for v in vectors:
            if len(v) < self.dimension:
                v = np.pad(v, (0, self.dimension - len(v)), 'constant')
            out.append(v[:self.dimension].tolist())
        return out

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            return [0.0]*self.dimension
        vec = self.vectorizer.transform([text]).toarray()[0]
        if len(vec) < self.dimension:
            vec = np.pad(vec, (0, self.dimension - len(vec)), 'constant')
        return vec[:self.dimension].tolist()

# ---------------- Document Manager ----------------
class DocumentManager:
    def __init__(self):
        self.documents: List[Any] = []
        self.processed_files: Dict[str, Dict[str, Any]] = {}
        self.embeddings = TFIDFEmbeddings()
        self.vectordb = None
        # LLM: ChatGroq if available
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0) if ChatGroq else None
        self.prompt_template = PromptTemplate(
            input_variables=["context","question"],
            template=(
                "Use the following context to answer the question. If the answer is not in the context, "
                "say 'I cannot find the answer in the provided documents.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        ) if PromptTemplate else None

    def add_file(self, filename: str, content: str, file_hash: str, file_size: int) -> Tuple[bool,str]:
        if file_hash in self.processed_files:
            return False, f"File '{filename}' already processed (duplicate)"
        doc = Document(page_content=content, metadata={"source":filename, "file_hash":file_hash, "file_size":file_size})
        self.documents.append(doc)
        self.processed_files[file_hash] = {"name": filename, "size": file_size}
        return True, f"Successfully processed '{filename}'"

    def _rebuild_vectordb(self):
        if not self.documents:
            self.vectordb = None
            return
        if RecursiveCharacterTextSplitter is None or FAISS is None:
            st.warning("LangChain text splitter or FAISS not installed. Document search won't work.")
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(self.documents)
        all_texts = [c.page_content for c in chunks]
        self.embeddings = TFIDFEmbeddings()
        _ = self.embeddings.embed_documents(all_texts)
        self.vectordb = FAISS.from_documents(chunks, self.embeddings)

    def answer_question(self, question: str) -> str:
        if not self.documents:
            return "No documents uploaded. Upload files and try again."
        if not self.vectordb:
            return "Document index not built yet. Please process the uploaded files."
        if not getattr(self.embeddings, 'is_fitted', False):
            return "Embeddings initializing. Please wait and try again."
        try:
            docs = self.vectordb.similarity_search(question, k=5)
            if not docs:
                return "I cannot find any relevant information in the documents to answer your question."
            context = "\n\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in docs])
            # Use LLM if available
            if self.llm and self.prompt_template and LLMChain:
                chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
                return chain.run(context=context, question=question)
            else:
                # Fallback: return concatenated snippets
                snippets = "\n---\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content[:800]}..." for d in docs])
                return f"(LLM unavailable) Relevant snippets:\n\n{snippets}"
        except Exception as e:
            return f"Error during retrieval/answering: {e}"

    def get_document_count(self) -> int:
        return len(self.documents)

    def get_document_list(self) -> List[Tuple[str,int]]:
        return [(doc.metadata.get('source','unknown'), doc.metadata.get('file_size',0)) for doc in self.documents]

    def get_processed_files_info(self) -> Dict[str,Any]:
        return self.processed_files

    def remove_file(self, filename: str):
        self.documents = [d for d in self.documents if d.metadata.get('source') != filename]
        remove_hash = None
        for h, info in list(self.processed_files.items()):
            if info.get('name') == filename:
                remove_hash = h
                break
        if remove_hash:
            del self.processed_files[remove_hash]
        if self.documents:
            self._rebuild_vectordb()
        else:
            self.vectordb = None

    def clear_all(self):
        self.documents = []
        self.processed_files = {}
        self.vectordb = None

# ---------------- Chat / Supabase persistence helpers ----------------
def add_message(role: str, content: str):
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({'role': role, 'content': content})


def save_messages_to_supabase(user_id: str):
    if not supabase or not user_id:
        return
    try:
        # Upsert by user_id: store entire session messages array
        supabase.table('chats').upsert({
            'user_id': user_id,
            'messages': st.session_state.messages
        }).execute()
    except Exception as e:
        st.warning(f"Failed to save chats to Supabase: {e}")


def load_messages_from_supabase(user_id: str) -> List[Dict[str,str]]:
    if not supabase or not user_id:
        return []
    try:
        res = supabase.table('chats').select('messages').eq('user_id', user_id).execute()
        if res.data and len(res.data) > 0:
            return res.data[0].get('messages', []) or []
    except Exception as e:
        st.warning(f"Failed to load chats from Supabase: {e}")
    return []

# ---------------- Initialize session state ----------------
if 'doc_manager' not in st.session_state:
    st.session_state.doc_manager = DocumentManager()
if 'staged_files' not in st.session_state:
    st.session_state.staged_files = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ---------------- OAuth handling ----------------
user_id = None
if supabase and SUPABASE_URL and SUPABASE_KEY:
    # Build login URL
    login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={REDIRECT_URL or 'http://localhost:8501'}"
    params = st.experimental_get_query_params()
    # Supabase may return access_token & refresh_token in query params depending on redirect setup
    access_token = params.get('access_token', [None])[0]
    refresh_token = params.get('refresh_token', [None])[0]

    if 'session' not in st.session_state:
        st.session_state.session = None

    if access_token and refresh_token and st.session_state.session is None:
        try:
            # set_session accepts dict with access_token and refresh_token
            session = supabase.auth.set_session({
                'access_token': access_token,
                'refresh_token': refresh_token
            })
            st.session_state.session = session
        except Exception as e:
            st.warning(f"Auth session failed: {e}")

    logged_in = st.session_state.session is not None and getattr(st.session_state.session, 'user', None) is not None
    if logged_in:
        user = st.session_state.session.user
        user_id = user.id
        # Load messages from supabase if session messages empty (merge gracefully)
        if user_id and (not st.session_state.messages):
            loaded = load_messages_from_supabase(user_id)
            if loaded:
                st.session_state.messages = loaded
else:
    login_url = None
    logged_in = False

# ---------------- UI Layout ----------------
st.title("📄 DocQnA — Mini ChatGPT with Documents")

# Left: main chat and QA
col1, col2 = st.columns([3,1])
with col1:
    # Top-right login status
    with st.container():
        if logged_in and st.session_state.session:
            st.success(f"✅ Logged in as {st.session_state.session.user.email}")
            if st.button("Logout"):
                try:
                    supabase.auth.sign_out()
                except Exception:
                    pass
                st.session_state.session = None
                st.experimental_set_query_params()
                st.experimental_rerun()
        else:
            if login_url:
                st.markdown(f"[🔑 Login with Google]({login_url})")
            else:
                st.info("Supabase not configured. Login disabled.")

    st.divider()

    # Render chat messages
    for msg in st.session_state.messages:
        role = msg.get('role','assistant')
        with st.chat_message(role):
            st.markdown(msg.get('content',''))

    # Chat input
    prompt = st.chat_input("Ask me about your documents...")
    if prompt:
        add_message('user', prompt)
        # Answer using DocumentManager
        answer = st.session_state.doc_manager.answer_question(prompt)
        add_message('assistant', answer)
        # Display assistant message
        with st.chat_message('assistant'):
            st.markdown(answer)
        # Save to Supabase only if logged in
        if logged_in and user_id:
            save_messages_to_supabase(user_id)

with col2:
    st.header("📂 Upload & Manage Documents")
    uploaded_files = st.file_uploader("Upload files (pdf, docx, txt)", type=['pdf','docx','txt'], accept_multiple_files=True)
    use_ocr = st.checkbox("Enable OCR for images", value=True)

    # Stage files
    if uploaded_files:
        processed_hashes = set(st.session_state.doc_manager.get_processed_files_info().keys())
        staged_hashes = {f.get('file_hash') for f in st.session_state.staged_files}
        for f in uploaded_files:
            try:
                data = f.getvalue()
                file_hash = get_file_hash(data)
                if file_hash in processed_hashes or file_hash in staged_hashes:
                    st.info(f"Skipping duplicate: {f.name}")
                    continue
                name = f.name
                _, ext = os.path.splitext(name)
                st.session_state.staged_files.append({
                    'name': name,
                    'bytes': data,
                    'size': f.size,
                    'ext': ext.lower(),
                    'file_hash': file_hash
                })
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")

    if st.session_state.staged_files:
        st.write(f"**Staged files:** {len(st.session_state.staged_files)}")
        for i, sf in enumerate(st.session_state.staged_files,1):
            st.write(f"{i}. {sf['name']} ({sf['size']/(1024*1024):.2f} MB)")

        if st.button("Process staged files"):
            progress = st.progress(0.0)
            total = len(st.session_state.staged_files)
            processed = 0
            for i, sf in enumerate(list(st.session_state.staged_files)):
                try:
                    progress.progress(i/total)
                    bio = io.BytesIO(sf['bytes'])
                    text = ''
                    if sf['ext'] == '.pdf':
                        text = extract_text_from_pdf(bio, use_ocr)
                    elif sf['ext'] == '.docx':
                        text = extract_text_from_docx(bio, use_ocr)
                    elif sf['ext'] == '.txt':
                        text = extract_text_from_txt(bio)
                    if text.strip():
                        success, msg = st.session_state.doc_manager.add_file(sf['name'], text, sf['file_hash'], sf['size'])
                        if success:
                            st.success(msg)
                            processed += 1
                        else:
                            st.info(msg)
                    else:
                        st.warning(f"No text found in {sf['name']}")
                except Exception as e:
                    st.error(f"Error processing {sf['name']}: {e}")
            # rebuild index
            st.session_state.doc_manager._rebuild_vectordb()
            progress.progress(1.0)
            st.success(f"Processed {processed}/{total} files")
            st.session_state.staged_files = []
            st.experimental_rerun()

    st.divider()
    st.subheader("Loaded Documents")
    doc_list = st.session_state.doc_manager.get_document_list()
    if doc_list:
        for i, (name, size) in enumerate(doc_list,1):
            size_mb = size/(1024*1024) if size else 0
            cols = st.columns([4,1])
            cols[0].write(f"{i}. {name} ({size_mb:.2f} MB)")
            if cols[1].button("Remove", key=f"rm_{i}"):
                st.session_state.doc_manager.remove_file(name)
                st.success(f"Removed {name}")
                st.experimental_rerun()
        if st.button("Clear all documents"):
            st.session_state.doc_manager.clear_all()
            st.success("Cleared all documents")
            st.experimental_rerun()
    else:
        st.info("No documents loaded yet.")

# ---------------- Footer / tips ----------------
st.divider()
st.caption("Notes: 1) Login with Google saves/loads chat history to Supabase. 2) Document search uses TF-IDF+FAISS and a local LLM (ChatGroq) when available; otherwise snippets are returned.")
