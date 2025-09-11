# app.py
import os
import io
import hashlib
import datetime
from typing import List, Dict, Any, Tuple

import streamlit as st
from PIL import Image
import pytesseract

# Optional libraries (graceful fallback)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
except Exception:
    TfidfVectorizer = None
    np = None

# LangChain and vectorstore
try:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.embeddings.base import Embeddings
    # optional LLMs
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

# Supabase client
try:
    from supabase import create_client
except Exception:
    create_client = None

# -----------------------
# Config & secrets
# -----------------------
st.set_page_config(page_title="DocQnA — Mini GPT", page_icon="📄", layout="wide")

SUPABASE_URL = st.secrets.get("SUPABASE_URL") if st.secrets else os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") if st.secrets else os.getenv("SUPABASE_KEY")
REDIRECT_URL = st.secrets.get("REDIRECT_URL") if st.secrets else os.getenv("REDIRECT_URL")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") if st.secrets else os.getenv("OPENAI_API_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.warning("Supabase client init failed. Auth disabled.")
        supabase = None
else:
    supabase = None

# -----------------------
# Helpers: text extraction, hashing
# -----------------------
def get_file_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def ocr_image(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def extract_text_from_pdf(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed.")
    file_bytes.seek(0)
    doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
    parts = []
    for page in doc:
        try:
            parts.append(page.get_text())
        except Exception:
            pass
        if use_ocr:
            try:
                imgs = page.get_images(full=True)
                for m in imgs:
                    xref = m[0]
                    base = doc.extract_image(xref)
                    im = base.get("image")
                    if im:
                        parts.append(ocr_image(im))
            except Exception:
                pass
    return "\n".join(parts)


def extract_text_from_docx(file_bytes: io.BytesIO, use_ocr: bool = True) -> str:
    if DocxDocument is None:
        raise RuntimeError("python-docx not installed.")
    file_bytes.seek(0)
    doc = DocxDocument(file_bytes)
    text = "\n".join(p.text for p in doc.paragraphs if p.text)
    if use_ocr:
        try:
            for rel in doc.part.rels.values():
                if hasattr(rel, "target_part") and getattr(rel.target_part, "blob", None):
                    text += "\n" + ocr_image(rel.target_part.blob)
        except Exception:
            pass
    return text


def extract_text_from_txt(file_bytes: io.BytesIO) -> str:
    file_bytes.seek(0)
    return file_bytes.read().decode("utf-8", errors="replace")


# -----------------------
# TF-IDF embeddings fallback (when no managed embeddings)
# -----------------------
class TFIDFEmbeddings(Embeddings if Embeddings is not None else object):
    def __init__(self, max_features: int = 384):
        if TfidfVectorizer is None or np is None:
            raise RuntimeError("scikit-learn and numpy required for TFIDFEmbeddings.")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=(1, 2)
        )
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
                v = np.pad(v, (0, self.dimension - len(v)), "constant")
            out.append(v[: self.dimension].tolist())
        return out

    def embed_query(self, text: str) -> List[float]:
        if not self.is_fitted:
            return [0.0] * self.dimension
        vec = self.vectorizer.transform([text]).toarray()[0]
        if len(vec) < self.dimension:
            vec = np.pad(vec, (0, self.dimension - len(vec)), "constant")
        return vec[: self.dimension].tolist()


# -----------------------
# Document Manager: store docs, build index, retrieve, call LLM
# -----------------------
class DocumentManager:
    def __init__(self):
        self.documents: List[Any] = []
        self.processed_files: Dict[str, Dict[str, Any]] = {}
        self.embeddings = TFIDFEmbeddings()
        self.vectordb = None
        # prefer ChatGroq, else OpenAI if configured
        self.llm = None
        if ChatGroq is not None:
            try:
                self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
            except Exception:
                self.llm = None
        elif OpenAI is not None and OPENAI_API_KEY:
            try:
                os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
                self.llm = OpenAI()  # default
            except Exception:
                self.llm = None

        self.prompt_template = (
            PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "Use the following context to answer the question. If the answer is not in the context, "
                    "reply: 'I cannot find the answer in the provided documents.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                ),
            )
            if PromptTemplate is not None
            else None
        )

    def add_file(self, filename: str, content: str, file_hash: str, file_size: int) -> Tuple[bool, str]:
        if file_hash in self.processed_files:
            return False, f"File '{filename}' already processed"
        if Document is None:
            # minimal fallback doc object
            doc = type("SimpleDoc", (), {"page_content": content, "metadata": {"source": filename, "file_size": file_size}})()
        else:
            doc = Document(page_content=content, metadata={"source": filename, "file_size": file_size})
        self.documents.append(doc)
        self.processed_files[file_hash] = {"name": filename, "size": file_size}
        return True, f"Successfully processed '{filename}'"

    def rebuild_index(self):
        if not self.documents:
            self.vectordb = None
            return
        if RecursiveCharacterTextSplitter is None or FAISS is None:
            st.warning("LangChain text splitter or FAISS not available; search quality will be limited.")
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(self.documents)
        all_texts = [c.page_content for c in chunks]
        self.embeddings = TFIDFEmbeddings()
        _ = self.embeddings.embed_documents(all_texts)
        self.vectordb = FAISS.from_documents(chunks, self.embeddings)

    def answer_question(self, question: str) -> str:
        if not self.documents:
            return "No documents uploaded. Please upload files."
        if not self.vectordb:
            return "Index not built yet. Upload documents or click 'rebuild index' if needed."
        try:
            docs = self.vectordb.similarity_search(question, k=5)
        except Exception as e:
            return f"Search failed: {e}"
        if not docs:
            return "I cannot find any relevant information in the uploaded documents."
        context = "\n\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in docs])
        # Use LLM if available
        if self.llm and self.prompt_template and LLMChain:
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            try:
                return chain.run(context=context, question=question)
            except Exception as e:
                return f"LLM generation error: {e}"
        # fallback: return helpful snippets
        snippets = "\n---\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content[:700]}..." for d in docs])
        return f"(LLM unavailable) Relevant snippets:\n\n{snippets}"


# -----------------------
# Chat persistence helpers (Supabase)
# -----------------------
def save_session_messages(user_id: str, messages: List[Dict[str, Any]]):
    if not supabase or not user_id:
        return
    try:
        # Upsert entire messages array by user_id
        supabase.table("chats").upsert({"user_id": user_id, "messages": messages}).execute()
    except Exception:
        st.warning("Failed to save messages to Supabase.")


def load_session_messages(user_id: str) -> List[Dict[str, Any]]:
    if not supabase or not user_id:
        return []
    try:
        res = supabase.table("chats").select("messages").eq("user_id", user_id).execute()
        if res.data and len(res.data) > 0:
            return res.data[0].get("messages", []) or []
    except Exception:
        pass
    return []


# -----------------------
# Initialize session state
# -----------------------
if "doc_manager" not in st.session_state:
    st.session_state.doc_manager = DocumentManager()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user" not in st.session_state:
    st.session_state.user = None
if "staged_files" not in st.session_state:
    st.session_state.staged_files = []


# -----------------------
# OAuth handling (redirect in same tab)
# -----------------------
login_url = None
logged_in = False
user_id = None
user_email = None

if supabase and SUPABASE_URL and SUPABASE_KEY:
    login_url = f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={REDIRECT_URL or ''}"
    params = st.query_params
    access_token = params.get("access_token")
    refresh_token = params.get("refresh_token")
    # tokens can be strings or lists depending on query_parsing; handle both
    if isinstance(access_token, list):
        access_token = access_token[0] if access_token else None
    if isinstance(refresh_token, list):
        refresh_token = refresh_token[0] if refresh_token else None

    if access_token and refresh_token and st.session_state.get("session") is None:
        try:
            session = supabase.auth.set_session({"access_token": access_token, "refresh_token": refresh_token})
            st.session_state.session = session
        except Exception:
            st.warning("Failed to set session from redirect tokens.")

    logged_in = st.session_state.get("session") is not None and getattr(st.session_state.get("session"), "user", None) is not None
    if logged_in:
        st.session_state.user = st.session_state.session.user
        user_id = st.session_state.user.id
        user_email = st.session_state.user.email
        # load messages (if session empty, load remote)
        if user_id and not st.session_state.messages:
            loaded = load_session_messages(user_id)
            if loaded:
                st.session_state.messages = loaded
else:
    login_url = None


# -----------------------
# UI: Login card when anonymous, otherwise main layout
# -----------------------
STYLES = """
<style>
/* simple cool gradient background and button */
body { background: linear-gradient(180deg,#071026 0%, #071b2a 100%); color: #e6eef7; }
.login-card {
  margin: 3rem auto;
  max-width: 520px;
  background: rgba(255,255,255,0.04);
  border-radius: 12px;
  padding: 28px;
  text-align:center;
  box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
.login-btn {
  display:inline-flex; align-items:center; gap:10px;
  background: linear-gradient(90deg,#4285F4,#34A853); color: white;
  border:none; padding: 12px 18px; border-radius:8px; font-weight:600;
}
.small-muted { color: #a9b8c9; font-size:13px; }
.card-sub { color: #cfe7ff; opacity:0.9; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

if not logged_in:
    # CENTERED LOGIN CARD
    st.markdown(
        """
        <div class="login-card">
            <h1 style="margin-bottom:6px;">👋 Welcome to <strong>DocQnA</strong></h1>
            <p class="card-sub">Upload documents, ask questions — save history when you sign in.</p>
            <div style="margin-top:18px;">
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if login_url:
            # Render a button that navigates current tab to Supabase auth (same tab)
            st.markdown(
                f"""<a href="{login_url}" style="text-decoration:none">
                    <button class="login-btn"> 
                        <img src="https://www.svgrepo.com/show/475656/google-color.svg" width="20" /> 
                        Sign in with Google
                    </button></a>""",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:12px'></div>")
            st.markdown('<div class="small-muted">Signing in saves your chat history and lets you access it across devices.</div>', unsafe_allow_html=True)
        else:
            st.warning("Login is not available because Supabase is not configured.")
    st.markdown("</div></div>", unsafe_allow_html=True)
    # but still show the app below so anonymous users can use it
    st.markdown("---")

# Main app layout (works for both states)
left, right = st.columns([3, 1])

with left:
    # Header row: show user on top-right area
    header_cols = st.columns([8, 2])
    header_cols[0].markdown("### 📄 DocQnA — Ask questions over your documents")
    if logged_in and user_email:
        header_cols[1].markdown(f"**{user_email}**")
    else:
        header_cols[1].markdown("")

    st.divider()

    # Chat window messages
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

    # Chat input
    prompt = st.chat_input("Ask me about your documents...")
    if prompt:
        # append user message
        st.session_state.messages.append({"role": "user", "content": prompt, "time": datetime.datetime.utcnow().isoformat()})
        with st.chat_message("user"):
            st.markdown(prompt)

        # produce answer via DocumentManager
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                answer = st.session_state.doc_manager.answer_question(prompt)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "time": datetime.datetime.utcnow().isoformat()})

        # save if logged in
        if logged_in and user_id:
            save_session_messages(user_id, st.session_state.messages)

with right:
    st.markdown("### 📂 Upload & Manage Documents")
    uploaded = st.file_uploader("Upload (pdf, docx, txt) — processed immediately", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    use_ocr = st.checkbox("Enable OCR for images", value=True)

    # Immediate processing for newly uploaded files
    if uploaded:
        processed = 0
        for f in uploaded:
            try:
                b = f.getvalue()
                h = get_file_hash(b)
                if h in st.session_state.doc_manager.processed_files:
                    st.info(f"Skipping duplicate: {f.name}")
                    continue
                bio = io.BytesIO(b)
                ext = os.path.splitext(f.name)[1].lower()
                txt = ""
                if ext == ".pdf":
                    txt = extract_text_from_pdf(bio, use_ocr)
                elif ext == ".docx":
                    txt = extract_text_from_docx(bio, use_ocr)
                elif ext == ".txt":
                    txt = extract_text_from_txt(bio)
                if txt.strip():
                    ok, msg = st.session_state.doc_manager.add_file(f.name, txt, h, f.size)
                    if ok:
                        st.success(msg)
                        processed += 1
                    else:
                        st.info(msg)
                else:
                    st.warning(f"No text found in {f.name}")
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")
        # Rebuild index so queries can use newly processed docs immediately
        st.session_state.doc_manager.rebuild_index()
        if processed:
            st.success(f"Processed {processed} files and rebuilt index")

    st.divider()
    st.markdown("#### Loaded Documents")
    docs = st.session_state.doc_manager.get_document_list()
    if docs:
        for i, (name, size) in enumerate(docs, start=1):
            c1, c2 = st.columns([4,1])
            c1.write(f"{i}. {name} ({(size or 0)/(1024*1024):.2f} MB)")
            if c2.button("Remove", key=f"rm_{i}"):
                st.session_state.doc_manager.remove_file(name)
                st.experimental_rerun()
        if st.button("Clear all documents"):
            st.session_state.doc_manager.clear_all()
            st.experimental_rerun()
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # Past chats (when logged in)
    if logged_in and user_id:
        st.markdown("#### 💬 Past Chats (your account)")
        history = load_session_messages(user_id)
        if history:
            for idx, item in enumerate(history[::-1]):  # newest first
                q = item.get("content", "") if item.get("role") == "user" else None
                # We stored full message pairs, show snippet of most recent user question/assistant answer pair
                # But the 'messages' array may contain many messages; we'll find last user message before assistant
                # For display, try to render the last user->assistant pair
                user_q = None
                assistant_a = None
                # iterate backward to find last user then assistant pair
                for m in reversed(item.get("messages", [item])) if isinstance(item, dict) and item.get("messages") else reversed(history):
                    # this is a compatibility guard; in our scheme we store messages array at top-level
                    pass
            # Simpler display: query supabase table for rows with messages and show their 'messages' top-level
            rows = supabase.table("chats").select("messages").eq("user_id", user_id).execute().data if supabase else []
            # Show recent row(s)
            for r in rows[::-1]:
                msgs = r.get("messages", [])
                # find last user prompt
                last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
                last_assistant = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
                label = (last_user.get("content")[:60] + "...") if last_user else "Conversation"
                if st.button(label, key=f"history_{label}"):
                    # load full messages
                    st.session_state.messages = msgs
                    st.experimental_rerun()
        else:
            st.info("No saved chats yet. Your chats will appear here after you sign in and ask questions.")
    else:
        st.markdown("Sign in to save chat history and view it here.")

st.divider()
st.caption(
    "Notes: Uploading documents processes them immediately. Login saves your chat history to Supabase. "
    "Make sure you configured SUPABASE_URL, SUPABASE_KEY, REDIRECT_URL in Streamlit secrets or env vars."
)
